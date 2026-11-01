/**
 * Generate and verify Session v4 fixtures with the pinned TypeScript implementation.
 *
 * Usage:
 *   pnpm --dir <deepseek-harness> exec tsx <this-file> generate <upstream> <output>
 *   pnpm --dir <deepseek-harness> exec tsx <this-file> verify <upstream> <root> <compression> <id>
 */

import { createHash } from 'node:crypto'
import { execFile } from 'node:child_process'
import { mkdir, rm, writeFile } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)

function moduleUrl(upstream: string, relative: string): string {
  return pathToFileURL(join(upstream, relative)).href
}

async function generate(upstream: string, output: string): Promise<void> {
  const format = await import(moduleUrl(
    upstream,
    'packages/session/session-persistence-jsonl/src/format.ts',
  ))
  const zstd = await import(moduleUrl(
    upstream,
    'packages/session/session-persistence-jsonl/src/zstd.ts',
  ))
  const header = {
    version: 4,
    id: 'dsh-v4-fixture',
    createdAt: 1_733_000_000_000,
    cwd: '/fixture',
    isSeeded: false,
  }
  const events = [
    { type: 'turn/start', seq: 0, time: 1_733_000_000_001, data: { turn: 1 } },
    {
      type: 'turn/end',
      seq: 1,
      time: 1_733_000_000_002,
      data: { turn: 1, reason: { kind: 'completed' } },
    },
  ]
  const headerLine = `${JSON.stringify(format.toHeaderLine(header))}\n`
  const eventBody = `${format.eventLines(events)}\n`

  await rm(output, { recursive: true, force: true })
  const artifacts: Array<{ compression: 'none' | 'zstd', path: string, sha256: string }> = []
  for (const compression of ['none', 'zstd'] as const) {
    const root = join(output, compression)
    const path = format.logPath(root, header.cwd, header.id, compression)
    await mkdir(dirname(path), { recursive: true })
    const content = compression === 'none'
      ? Buffer.from(headerLine + eventBody)
      : Buffer.concat([
          await zstd.compressZstdFrame(headerLine),
          await zstd.compressZstdFrame(eventBody),
        ])
    await writeFile(path, content)
    artifacts.push({
      compression,
      path: path.slice(output.length + 1),
      sha256: createHash('sha256').update(content).digest('hex'),
    })
  }
  const { stdout: commitOutput } = await execFileAsync(
    'git',
    ['-C', upstream, 'rev-parse', 'HEAD'],
  )
  const commit = commitOutput.trim()
  await writeFile(join(output, 'MANIFEST.json'), `${JSON.stringify({
    upstreamCommit: commit,
    sessionId: header.id,
    eventCount: events.length,
    artifacts,
  }, null, 2)}\n`)
}

async function verify(
  upstream: string,
  root: string,
  compression: 'none' | 'zstd',
  id: string,
): Promise<void> {
  const { Context } = await import(moduleUrl(upstream, 'vendor/cordis/src/index.ts'))
  const persistence = await import(moduleUrl(
    upstream,
    'packages/session/session-persistence-jsonl/src/index.ts',
  ))
  const ctx = new Context()
  try {
    await ctx.plugin(persistence, { root, compression })
    const service = ctx.get('sessionPersistence')
    if (service === undefined) throw new Error('sessionPersistence was not mounted')
    const handle = await service.open(id, 'read')
    try {
      const result = await handle.read()
      if (result.events.length !== 2) {
        throw new Error(`expected 2 events, received ${result.events.length}`)
      }
    } finally {
      await handle.close()
    }
  } finally {
    await ctx.fiber.dispose()
  }
}

async function main(): Promise<void> {
  const [command, upstreamArg, ...args] = process.argv.slice(2)
  if (command === undefined || upstreamArg === undefined) {
    throw new Error('missing command or upstream path')
  }
  const upstream = resolve(upstreamArg)
  if (command === 'generate') {
    if (args[0] === undefined) throw new Error('missing fixture output path')
    await generate(upstream, resolve(args[0]))
  } else if (command === 'verify') {
    const [root, compression, id] = args
    if (root === undefined || (compression !== 'none' && compression !== 'zstd') || id === undefined) {
      throw new Error('verify requires root, compression and session id')
    }
    await verify(upstream, resolve(root), compression, id)
  } else {
    throw new Error(`unknown command: ${command}`)
  }
}

void main()

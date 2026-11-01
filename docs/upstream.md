# Tracking upstream DeepSeek Harness

The pinned upstream TypeScript source is the specification for this package.

## The pin

[upstream.json](upstream.json) records one upstream commit for the whole package, and
maps every upstream package in scope to its Python module, release milestone and
porting status (`planned`, `in-progress`, `ported`). It also lists the modules that
have no upstream package (`new_modules`).

`watched` lists unmapped upstream packages that milestone `0.1` packages import, such
as `util/brand`, `util/timeout` and `typert/protocol`. Python inlines or omits their
behavior, so they have no module, but their commits are reported like a mapped
package's. The v0–v3 session-format migrations are left out on purpose.

Only milestone `0.1` belongs to the `deskagents` distribution. Later mapped
packages are retained as the source inventory for the external plugin families in
[plugin-roadmap.md](plugin-roadmap.md).

Current pin: [`ce7073d`](https://github.com/deepseek-ai/deepseek-harness/tree/ce7073d51254c26cd63b22a60835b8f948deb447)
(2026-10-30).

Reference checkout: `/Users/iali/workplace/projects/personal/deepseek-harness`, or
wherever `HARNESS_UPSTREAM` points. It is read through git only.

## Module names

Each upstream package becomes one Python module:

| Upstream package | Rule | Example |
|---|---|---|
| `packages/core/<pkg>` | `deskagents.core.<pkg>` | `core/agent-loop` → `deskagents.core.agent_loop` |
| `packages/<group>/<group>` (the Service Definition) | `deskagents.<group>` | `fs/fs` → `deskagents.fs` |
| `packages/<group>/<group>-<rest>` | `deskagents.<group>.<rest>` | `fs/fs-local` → `deskagents.fs.local` |
| any other `packages/<group>/<pkg>` | `deskagents.<group>.<pkg>` | `fs/tool-fs` → `deskagents.fs.tool_fs` |

Dashes become underscores. Seams keep upstream's three roles: the group's package
holds the Service Definition, provider modules sit beside it, and consumers (usually
`tool_*` modules) depend on the definition only.

## Pinned assets

`tests/upstream_assets/` holds unmodified copies of upstream files that tests read:

- `persistence-schema.json`: upstream's normalized type inventory of the session format,
  with a SHA-256 digest per type. Its own `formatVersion` (2) is the inventory layout,
  not the session format.
- `v4.json`: the finalized session format v4 baseline.

At `ce7073d` the inventory has 65 roots against the baseline's 62: 8 event types have
changed digests since v4 was finalized (including `user/message` and
`developer/message`) and 3 are new (`job/registered`, `job/settled`,
`working-directory/change`). Session code targets the inventory at the pin.

`MANIFEST.json` records each file's upstream path, the commit and its checksum.

`tests/fixtures/dsh_v4/` contains plaintext and Zstandard Session v4 artifacts written
by the pinned TypeScript implementation. Its manifest records the pin and artifact
checksums. Regenerate them with `scripts/dsh_session_fixtures.ts`; use
`scripts/write_interop_session.py` plus that script's `verify` command for the reverse
Python-to-`dsh` check.

## Vendored Cordis

The pin also records the versions of the two vendored upstream packages that
[deepseek-cordis](https://github.com/akkefa/deepseek-cordis) ports: `vendor/cordis`
(`@deepseek-ai/cordis`) and `vendor/loader` (`@deepseek-ai/cordis-plugin-loader`).
At `ce7073d` they are `4.0.6-alpha.1` and `1.0.7-alpha.1`. Upstream's
`vendor/cordis-rc11-upgrade-plan.md` proposes moving them to Cordis `4.0.0-rc.11`
and Loader `1.0.0-rc.8` in staged PRs. When a vendored version moves, deepseek-cordis
follows first, then this package raises its `deepseek-cordis` range.

## Watching upstream

`.github/workflows/upstream.yml` runs daily and on demand. It takes a blobless clone
of upstream master, verifies the pin with `--check`, and runs the change report with
`--exit-code`. When anything changed it opens one issue titled "Upstream DeepSeek
Harness changed since the pin", or refreshes its body and comments when upstream moves
again. Close the issue when the pin moves. The workflow uses the job's own token
with `issues: write`; no secret is configured.

`.github/dependabot.yml` opens a pull request for each new `deepseek-cordis` release,
including releases outside the pinned range.

Scheduled workflows run only from the default branch.

## Moving the pin

When the watch opens an issue, or at the start of each milestone:

1. Fetch upstream and run `uv run python scripts/upstream_changes.py --to origin/master`.
   It lists commits since the pin for every mapped package, new upstream packages in
   mapped groups, pinned assets that changed, and vendored Cordis packages that
   changed, with their old and new versions.
2. Write `docs/upstream-audit-<date>.md`: for each change, adopt, defer, or record an
   intentional Python difference.
3. Port the adopted changes with their tests, update the pin and vendored versions in
   `upstream.json`, refresh the assets and `MANIFEST.json`, and update spec checksums in
   `compatibility-cases.json`.
4. Run `uv run python scripts/upstream_changes.py --check`.

## Intentional Phase 3 omissions

Upstream tests left out of the Phase 3 port, by reason. Individual cases skipped inside
ported spec files follow the same reasons.

| Reason | Upstream spec files |
|---|---|
| v0–v3 decoding and migration (Python reads v4 only) | `session-format-v1-to-v2/tests/migration.spec.ts` and 13 of 15 `validation.spec.ts` cases; `session-format-v3-to-v4/tests/{migration,interrupted-turn,catalog-completion,corpus}.spec.ts` and the migration cases of `tool-role`, `sources`, `content`, `canonical`, `message-sources`, `retired-content`; `session-format-catalog/tests/{children,preset-migration}.spec.ts` and 12 of 14 `catalog.spec.ts` cases; `session-persistence-jsonl/tests/{catalog-migration,migration-refusal,migration-verifier,migration-verifier-real,multi-edge-publication,v2-ptc-migration,v2-system-migration,v3-restart-migration,content-admission,generation}.spec.ts` and 20 `jsonl.spec.ts` generation cases |
| `AbortSignal` (Python cancels tasks instead) | 3 `jsonl.spec.ts` durability cases |
| Node-only runtime (worker threads, Loader e2e, koffi bindings, `.e2e.ts`/`.perf.ts`) | `built-migration-worker.e2e.ts`, `teardown-loader.e2e.ts`, 8 `win32.spec.ts` cases; `session-projection/tests/loader-compatibility.e2e.ts`. `lease.two-process.e2e.ts` is replaced by a Python two-process test. |
| JavaScript-only semantics (thenable adoption, non-Error throws, type-level tests) | 2 `session-projection/tests/compatibility.spec.ts` cases, `registration.type-test.ts`, 1 live-write contract case, 1 `developer-history.spec.ts` case |
| Needs a package from a later phase | 1 `session-format-v3-to-v4/tests/fork.spec.ts` case (token meter, tool-result pruner), 1 `catalog.spec.ts` case (image-offload projection) |

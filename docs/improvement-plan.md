# deskagents phased improvement plan

Prepared: 2026-11-01. Based on the readiness audit of checkout `d3cdcaa`, against DeepSeek Harness `ce7073d51254c26cd63b22a60835b8f948deb447`.

## Objective and scope

Deliver a reliable, installable Python core runtime with documented compatibility against the pinned DeepSeek Harness, then publish an initial alpha and progress to 0.1.0.

Keep filesystem/shell/sandbox execution, skills, web, MCP, attachments, compaction, subagents, telemetry, UI/server integrations, and product-specific providers in separate plugin families. Their absence does not prevent the core release. Legacy session migration and advanced Node/grouped bundle behavior remain explicit exclusions.

This document plans the work; it does not implement changes or authorize publication. Existing build tooling and architecture should be retained. Prefer targeted fixes over a broad rewrite.

## Implementation status — 2026-11-01

| Phase | Current result |
|---|---|
| 0 | Contract and defect register established; credential-free origin configured. |
| 1 | Request, replay and model fixes implemented and verified with mocked HTTP. |
| 2 | Stream state, SSE, cache usage, provider errors and agent cancellation verified. |
| 3 | Single-call/composite API and six deferred cases implemented; plugin lifecycle verified. |
| 4 | Literal upstream inventory, durable resume, two-way session checks and clean wheel installs pass. Windows CI added; remote platform runs pending. Alpha verification omissions explicitly deferred. |
| 5 | Runnable examples/docs and 0.1.0a1 artifacts prepared and checked. Bounded live checker added; credentials unavailable, so live calls remain unverified. |
| 6 | Manual TestPyPI workflow and publisher instructions ready. Account setup, uploads, public-index installation and release decision remain outstanding. |

Local validation: 2,177 passed, 2 Windows-only skips; Ruff, mypy, distribution
and strict Twine checks pass. No release commit/tag/upload has been made.

## Phase 0 — Establish the release contract

Repository preparation implemented on 2026-11-01: see [release-contract.md](release-contract.md) for the baseline, defect register, and compatibility matrix. Adapter fixes are now resolved. Credential rotation remains a maintainer prerequisite before release.

**Purpose:** make the intended behavior and remaining work measurable.

- Recheck the working tree and upstream pin when implementation begins. Keep this improvement cycle on the audited pin; handle newer upstream changes as a separate decision.
- Record the eight reproduced adapter defects as regression cases, grouped into request/model handling and streaming/usage.
- Define a compatibility matrix for each in-scope core package: implemented behavior, applicable upstream cases, intentional adaptations, and deferred behavior. Distinguish implementation status from verification status.
- Correct stale naming claims, old checkout paths, and inconsistent counts in the README, plan, and audit documents.
- Keep the production release trigger explicit: pushing a matching `v*` tag invokes the existing release workflow.
- Before any release push, rotate the previously identified embedded Git token and replace the local remote with a credential-free URL. This is maintainer account work; do not copy the token into patches or reports.

**Completion criteria:** every confirmed finding has a tracked implementation/test destination; the release scope is unambiguous; the existing suite remains green before behavioral edits.

## Phase 1 — Repair DeepSeek requests, replay, and model metadata

**Purpose:** make valid core conversations reach the provider correctly.

Primary implementation areas: `llm/deepseek`, `llm/deepseek_api_key`, and their integration with `llm/runtime.py` and `llm/content.py`.

1. Implement developer tool-addition/removal serialization using upstream wire references and ordering.
2. Preserve initial system prompts and subsequent in-history prompt updates, including placement relative to user/tool-result turns.
3. Validate replay envelope kind/version, source and target model identity, block count/types, and signature types. Discard incompatible metadata while preserving durable content.
4. Skip empty assistant text and keep replay indexing aligned with the original content positions.
5. Normalize model listings and exact model resolution consistently; ID-only configuration must produce usable names.
6. Supply the correct reasoning capabilities/defaults and enforce the disabled-thinking policy. Ensure runtime preparation and serialization agree about valid efforts.
7. Compare immediate tool-result validation and ordering with the pinned serializer; add missing checks for duplicate/unresolved tool calls where applicable to the supported core path.

**Validation:** port applicable upstream serializer/replay/model-info cases; add runtime-level tests for README-style registration, model listing, reasoning selection, tool registry changes, system updates, and switching models within one adapter. Use mocked HTTP to verify the actual outgoing body, not only isolated helper output.

**Completion criteria:** audit defects 1–6 are resolved; a multi-turn mocked DeepSeek conversation survives tool/prompt changes and model switching; the full existing suite passes.

**Compatibility note:** malformed replay will be dropped and invalid request histories may be rejected earlier. Record these observable changes.

## Phase 2 — Repair streaming and token accounting

**Purpose:** make provider completion, failure, and usage reporting trustworthy.

1. Port upstream stream-state validation: required start, valid unique block indices, allowed delta types, block closure, settlement ordering, and terminal completion.
2. Validate streamed tool JSON and preserve the upstream max-token truncation exception.
3. Reject empty successful responses and premature stream endings with the appropriate failure categories.
4. Preserve cumulative input/output/cache-read/cache-write counters and upstream total-token semantics. Reject invalid numeric fields.
5. Review SSE framing and provider error translation against the pinned upstream implementation; add focused cases for any confirmed divergence.
6. Verify cancellation and stream closure through the actual runtime/agent path, including a stalled mocked transport, so resources close and partial output is handled consistently.

**Validation:** recorded/mocked SSE cases for valid text, reasoning, tools, usage, malformed ordering, truncation, cancellation, and transport failure. Verify errors are normalized at the runtime boundary rather than exposed as accidental KeyError/TypeError exceptions.

**Completion criteria:** audit defects 7–8 are resolved; malformed streams cannot become successful turns; cache usage survives through token-meter consumers; both provider suites and the full core suite pass.

**Compatibility note:** usage totals may increase where cache tokens were previously omitted; malformed streams previously accepted will fail explicitly.

## Phase 3 — Complete the core tool execution paths

**Purpose:** support external plugins that depend on upstream single-call and nested tool behavior.

1. Define the public Python equivalent of upstream's single-call pipeline.
2. Share policy, dispatch, post-processing, materialization, and finalization logic between single-call and batch execution, preserving their upstream differences.
3. Implement nested composite dispatch, including nested conclusions, presentation metadata, deferred contexts, and exception behavior.
4. Verify pre-aborted calls, guard/approval denial, replacement signals, cancellation drain, result notification, and cleanup ownership.
5. Resolve the six deferred catalog cases: implement applicable behavior or document a specific, justified exclusion. Do not relabel missing behavior as ported.
6. Extend the public plugin contract with a small external plugin exercising nested dispatch and unload/remount behavior.

**Validation:** port the relevant upstream cases and run batch scheduling, cancellation, session repair, and plugin-lifecycle regression suites. Test nested execution for duplicate/missing results and deadlocks when concurrency limits are reached.

**Completion criteria:** supported nested/single-call paths preserve upstream results, contexts, and lifecycle behavior; existing batch behavior remains green; deferred cases are explicitly accounted for.

**Compatibility note:** keep the existing batch API working. Document any necessary public API addition or behavioral correction.

## Phase 4 — Establish compatibility and platform evidence

**Purpose:** make the compatibility claims match demonstrated behavior.

1. Inventory the remaining applicable upstream cases for agent loop, agent service, tools, system prompt, LLM runtime, and DeepSeek adapter. Classify Node/UI/attachment-specific cases separately.
2. Port high-risk cases first: resume and request reconstruction, interruption, retry, owner disposal, tool history changes, and provider failure. Fix newly demonstrated defects within core scope.
3. Run an integrated scenario through bundle loading, provider dispatch, tool execution, JSONL checkpoint, disposal, resume, and a subsequent request.
4. Verify Python-written plaintext and Zstandard sessions with the pinned TypeScript reader, in addition to the committed fixtures read by Python. Preserve the immutable generation-file contract.
5. Add Windows CI and platform-correct virtual-environment executable paths. Exercise the two Windows lease tests there. Retain Linux Python 3.11–3.13 and macOS coverage.
6. Validate a fresh core-only install plus each provider extra; check that core imports do not require HTTP dependencies.

**Completion criteria:** supported-platform CI is green; the integrated durable-session scenario and two-way format checks pass; each applicable case is ported or explicitly deferred; no known unaddressed data-loss, deadlock, or false-success defect remains in the release path.

**Release scope rule:** exhaustive upstream parity is not required for an alpha, but its omissions must be named. Before 0.1.0, any remaining applicable core behavior must either be implemented or removed from the promised compatibility contract.

## Phase 5 — Finish the user experience and release candidate

**Purpose:** let a new user install the package and compose a useful durable agent.

1. Replace the partial README snippet with a runnable `asyncio.run(main())` example, environment-based credentials, one completed turn, and cleanup.
2. Add a durable-agent example composing JSONL persistence, retry, token metering, provider registration, and resume. Explain which services the minimal bundle enables.
3. Document model configuration, supported content, cancellation, error reporting, and intentional exclusions. Keep API examples consistent with the built wheel.
4. Run bounded, opt-in live DeepSeek and OpenAI-compatible smoke tests with supplied credentials: simple reply, tool round trip, subsequent reasoning replay where supported, and cancellation. Keep deterministic failure cases mocked. Record endpoint/model/date and results without credentials or sensitive transcripts.
5. Prepare `0.1.0a1` as the initial candidate, update the changelog and metadata, refresh `uv.lock`, and run locked synchronization.
6. Build a clean wheel/sdist pair; run distribution and strict Twine checks; install and exercise the wheel outside the repository.

**Completion criteria:** published documentation examples run against the candidate wheel; live smoke results are recorded or explicitly listed as unverified; package checks pass; the candidate's compatibility statement matches Phase 4 evidence.

## Phase 6 — Rehearse and publish

**Purpose:** verify the distribution path before the first production release.

1. Prepare a separate manually triggered TestPyPI workflow/environment; production `v*` tags must continue to mean production releases.
2. Configure the TestPyPI publisher and production PyPI pending publisher. For the current production workflow, verify owner `akkefa`, repository `deskagents`, workflow `release.yml`, environment `pypi`, and project name `deskagents` against the final release configuration.
3. Publish the approved candidate to TestPyPI. Download that exact wheel and install it in a fresh environment with dependencies from normal PyPI; run the quick start and metadata checks.
4. Present the final version, commit, artifacts, validation results, and known limitations for the maintainer's release decision.
5. After explicit publication authorization, push the approved release commit and exact matching tag, observe the production workflow, and verify installation of the exact version from PyPI.
6. Address alpha feedback with tested patches; promote to `0.1.0` only once the promised core behavior and supported-platform gates are met.

**Completion criteria:** the approved version installs from public PyPI with the documented extras, examples work, project links render correctly, and remaining limitations are visible to users.

Account configuration, live credentials, and publication authorization are external dependencies. Complete all available implementation and release preparation before requesting the specific missing input. A TestPyPI or PyPI upload is a separate action from preparing this plan.

## Execution discipline

- Sequence: Phase 0 → 1 → 2 → 3 → 4 → 5 → 6.
- Make small, reviewable changes grouped by behavior. Reuse the existing architecture and packaging workflows.
- For each behavioral fix: identify the pinned upstream behavior, reproduce the discrepancy, implement the change, and add meaningful regression coverage.
- Keep deterministic CI network-free. Live smoke tests are explicit and bounded.
- Run relevant tests during development; run the full suite, Ruff, formatting, mypy, and required package checks at phase gates.
- Update the compatibility catalog, changelog, and public limitations when behavior changes. Do not equate total test counts with compatibility coverage.
- Preserve published v4 data semantics. Any format-impacting change requires explicit interoperability evidence.
- Keep changes uncommitted and unpublished until requested; phase completion does not itself require a commit or release.

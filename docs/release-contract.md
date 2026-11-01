# Core release contract

Baseline: 2026-11-01, checkout `d3cdcaa`, upstream
`ce7073d51254c26cd63b22a60835b8f948deb447`. This improvement cycle keeps the pin
unchanged. Recheck upstream separately before proposing a pin update.

## Promise and release gates

The distribution targets a standalone Python core: scope/session v4, persistence,
projections, prompts, tools, agent loop, bundle composition, the LLM seam, retry,
token metering, and DeepSeek/OpenAI-compatible providers. It is an alpha candidate and
runnable; it does not yet promise complete parity for these areas.

The initial candidate is an alpha after the known adapter defects are fixed and
remaining limitations documented. Before 0.1.0, every promised core behavior must
be implemented and verified, or explicitly removed from the compatibility promise.
Use the [improvement plan](improvement-plan.md) for the ordered acceptance gates.

Execution/sandbox, attachments, skills, web, MCP, compaction, subagents, telemetry,
UI/host/API/SDK, Azure execution and product-specific integrations remain external.
Legacy v0–v3 migration and Node/HMR/grouped-expression bundles remain excluded.
These exclusions do not block the core release.

## Baseline evidence and remaining verification

Before Phase 0 edits, macOS/Python 3.11.15: **2,118 passed, 2 Windows-only skips**.
The inventory checker verified 92 mapped packages, pinned assets and 2 vendored
versions. Of the mapped packages, only 26 are milestone 0.1 core packages; three
additional Python-specific modules are also milestone 0.1.

The 2026-11-01 audit additionally passed lint, formatting, mypy, wheel/sdist,
distribution/Twine checks, locked installation, and a fresh core-only wheel import.
Current candidate validation on macOS/Python 3.11.15: **2,177 passed, 2
Windows-only skips**, no xfails; Ruff, formatting and mypy pass. Clean core,
DeepSeek-extra and OpenAI-extra wheel environments run the minimal and durable
examples. Fresh plaintext and Zstandard sessions pass both interoperability
directions against the pinned TypeScript runtime. Windows CI is now configured;
remote Linux/Windows CI and authenticated provider calls remain unverified locally.

## Resolved adapter defects

DS-01 through DS-08 now pass in
`tests/llm/deepseek/test_release_regressions.py`, without expected-failure markers.
Request fixes cover tool references, positional prompt updates, model-safe replay,
reasoning metadata, listing names and empty text. Stream fixes enforce start/state
validation and retain cache counters. Additional mocked HTTP and agent cancellation
tests live in `test_protocol.py`. Python defaults and API changes are documented in
[model-adapters.md](model-adapters.md).

## Per-module compatibility matrix

Implementation status comes from [upstream.json](upstream.json). Evidence counts
come from [compatibility-cases.json](compatibility-cases.json), grouped by upstream
package. Counts are a dated inventory, not code coverage or proof of completeness.
A `ported` implementation status does not mean every applicable case is verified.
Unmarked Python tests also exist; absence of source-indexed cases is not absence of tests.

Catalog snapshot: 62 adapted, 31 contract, 51 not-applicable, 855 ported.

| Module | Implementation status | Catalogued evidence | Adaptations / limitations / unverified work |
|---|---|---|---|
| `deskagents.core.scope` | ported | 14 ported, 7 adapted | No additional exclusion established in this phase; catalog adaptations remain authoritative. Verification is not exhaustive. |
| `deskagents.core.session` | ported | 261 ported, 21 adapted, 42 not-applicable | No additional exclusion established in this phase; catalog adaptations remain authoritative. Verification is not exhaustive. |
| `deskagents.core.system_prompt` | ported | 1 ported | Prompt assembly exists; broader cases inventoried as deferred-review. |
| `deskagents.core.tools` | in-progress | 140 ported, 27 adapted, 8 not-applicable | Single-call and nested composite dispatch implemented; Python batching adaptations documented in catalog. |
| `deskagents.core.tool_executor_native` | ported | No source-indexed cases | No additional exclusion established in this phase; catalog adaptations remain authoritative. Verification is not exhaustive. |
| `deskagents.core.agent` | in-progress | 5 ported | Focused lifecycle/initiator cases; remaining literal upstream cases inventoried as deferred-review. |
| `deskagents.core.agent_loop` | in-progress | 26 ported | Focused turn/cancellation cases; remaining literal upstream cases inventoried as deferred-review. |
| `deskagents.core.agent_default_model` | ported | No source-indexed cases | No additional exclusion established in this phase; catalog adaptations remain authoritative. Verification is not exhaustive. |
| `deskagents.core.agent_tool_presentation` | ported | No source-indexed cases | No additional exclusion established in this phase; catalog adaptations remain authoritative. Verification is not exhaustive. |
| `deskagents.llm` | ported | 15 ported | Attachment projection external; provider behavior has separate evidence. |
| `deskagents.llm.retry` | ported | 8 ported | No additional exclusion established in this phase; catalog adaptations remain authoritative. Verification is not exhaustive. |
| `deskagents.llm.token_meter` | ported | 13 ported | Attachment pricing uses structural heuristics until an external provider supplies projections. |
| `deskagents.llm.deepseek` | in-progress | 6 ported | DS-01–DS-08 fixed; files/images external. Remaining literal upstream cases inventoried as deferred-review. |
| `deskagents.llm.deepseek_api_key` | ported | No source-indexed cases | Provider registration exists; mocked runtime request validation passes; live endpoint check pending. |
| `deskagents.llm.deepseek_llm_api_extensions` | ported | 1 ported | No additional exclusion established in this phase; catalog adaptations remain authoritative. Verification is not exhaustive. |
| `deskagents.util.values` | in-progress | No source-indexed cases | No additional exclusion established in this phase; catalog adaptations remain authoritative. Verification is not exhaustive. |
| `deskagents.session.format` | ported | 17 ported, 7 adapted | No additional exclusion established in this phase; catalog adaptations remain authoritative. Verification is not exhaustive. |
| `deskagents.session.format_v1_to_v2` | in-progress | 13 ported | Only current-format framing; legacy migration excluded. |
| `deskagents.session.format_v3_to_v4` | in-progress | 118 ported | Only v4 validation/codec; legacy migration excluded. |
| `deskagents.session.format_catalog` | in-progress | 5 ported | v4 admitted; v0–v3 migration excluded. |
| `deskagents.session.persistence` | ported | 18 ported, 1 not-applicable, 31 contract | No additional exclusion established in this phase; catalog adaptations remain authoritative. Verification is not exhaustive. |
| `deskagents.session.persistence_jsonl` | ported | 131 ported | v4 plaintext/Zstandard fixtures; Windows lease execution still needs Windows CI. |
| `deskagents.session.projection` | ported | 60 ported | No additional exclusion established in this phase; catalog adaptations remain authoritative. Verification is not exhaustive. |
| `deskagents.session.checkpoint_policy` | ported | 2 ported | No additional exclusion established in this phase; catalog adaptations remain authoritative. Verification is not exhaustive. |
| `deskagents.session.title` | ported | No source-indexed cases | Fallback titles and user renames; LLM-generated titles external. |
| `deskagents.session.log_deepseek` | ported | 1 ported | No additional exclusion established in this phase; catalog adaptations remain authoritative. Verification is not exhaustive. |
| `deskagents.bundle` | ported | Python-specific; no one-to-one upstream package | Flat id-addressed rows; grouped trees, JavaScript expressions and HMR excluded. |
| `deskagents.testing` | in-progress | Python-specific; no one-to-one upstream package | Reusable Python contracts exist; they do not establish complete upstream parity. |
| `deskagents.llm.openai` | ported | Python-specific; no one-to-one upstream package | New Python adapter with mocked HTTP tests; live endpoint behavior remains unverified. |

The catalog's `adapted`, `not-applicable`, and `deferred` entries carry case-level
reasons; preserve those distinctions. The [literal-case inventory](upstream-case-inventory.json) records 1,425 case
families in 80 spec files across six high-risk packages. Unmapped entries remain
explicitly deferred-review, including applicability assessment; this is not
exhaustive parity or a complete dynamic-test inventory. Runtime/native-tool paths overlap, so package
counts must not be interpreted as independent coverage percentages.

## Release operations

- Production: `.github/workflows/release.yml` runs on pushed `v*` tags. The tag must
  exactly match `pyproject.toml`; version changes require a refreshed lockfile.
- `.github/workflows/testpypi.yml` supplies a manual TestPyPI rehearsal path.
  See [publishing.md](publishing.md) for publisher setup and exact release steps.
- Publisher/account configuration and live endpoint credentials have not been
  inspected. Preparing artifacts does not authorize publication.
- The audit found an access token embedded in local Git remote configuration.
  Remove embedded credentials from the remote and rotate/revoke that token before
  a release push. Removing the URL credential is not token revocation. Do not place
  credential values in reports, tests, commits or logs. The embedded credential was
  removed from the local origin URL during Phase 0 on 2026-11-01; revocation remains
  a maintainer action.

## Current completion boundary

Phases 0–3 implementation is complete. Phases 4–5 local implementation and candidate
checks pass; remote platform evidence and credentialed live tests remain external.
Phase 6 workflow/documentation is prepared, but publisher account setup, TestPyPI
rehearsal and production publication have not occurred. No commit/tag was created.
The alpha does not promise all upstream cases; the catalog and explicit deferred
inventory define the remaining verification work before stable compatibility.

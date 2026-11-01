# Contributing to deskagents

deskagents implements DeepSeek Harness in Python. The pinned upstream TypeScript
source is the specification. Start with [the plan](docs/PLAN.md) and
[how upstream is tracked](docs/upstream.md).

## Set up a checkout

Use Python 3.11, 3.12 or 3.13 and uv. Keep an upstream checkout beside this one
(`../deepseek-harness`) or point `HARNESS_UPSTREAM` at it.

```sh
uv sync --locked --group packaging
uv run --locked python scripts/upstream_changes.py --check
```

The lockfile is committed. Don't update dependency versions as a side effect of
unrelated changes. For an intentional change, edit `pyproject.toml` and regenerate
`uv.lock`.

## Port or fix something

1. Find the upstream package in [docs/upstream.json](docs/upstream.json) and read its
   source and tests at the pinned commit, not at upstream `HEAD`.
2. Write the Python module named in the map. Its docstring names the upstream package
   path and the commit it follows.
3. Port the upstream test cases. Mark each with
   `@pytest.mark.upstream("packages/<group>/<pkg>/tests/<file>.spec.ts::<case title>")`
   and add it to [docs/compatibility-cases.json](docs/compatibility-cases.json) with the
   spec file's SHA-256 at the pin. A case that can't apply in Python is recorded as
   `not-applicable` with a reason, never dropped silently.
4. Set the package's `status` in the map to `in-progress` or `ported`.
5. Update [docs/pepgpt-migration.md](docs/pepgpt-migration.md) for any PepGPT symbol the
   change covers.

Keep event names, tool names, tool schemas and session fields exactly as upstream
has them; sessions and model-visible output must match. Python-specific choices
follow the translation rules in the plan.

A module may import only the standard library, `deskagents`, the core dependencies, or
the dependencies of an extra registered for that module in
`tests/boundary/test_import_boundaries.py`.

## Check before proposing a change

Run the same checks as CI:

```sh
uv run --locked --group packaging pytest -W error
uv run --locked --group packaging ruff check .
uv run --locked --group packaging ruff format --check .
uv run --locked --group packaging mypy
uv build
uv run --locked --group packaging python scripts/check_distribution.py dist/*.whl dist/*.tar.gz
uv run --locked --group packaging twine check --strict dist/*.whl dist/*.tar.gz
```

Remove `dist/` between builds; the audit expects exactly one wheel and one source
archive.

## Releases

Release decisions belong to the maintainer. Preparing artifacts does not authorize
publishing them. In agent-assisted work, don't push, tag or publish without the
maintainer's explicit request.

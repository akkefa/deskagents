# Publishing deskagents

The prepared candidate is `0.1.0a1`. All publication is still a maintainer action;
no tag or package upload is part of local artifact preparation.

## Publisher configuration

Create verified PyPI and TestPyPI accounts with 2FA, then pending publishers for a
new project (or publishers under the existing project's Publishing settings):

| Field | Production | Rehearsal |
|---|---|---|
| Project | deskagents | deskagents |
| GitHub owner | akkefa | akkefa |
| Repository | deskagents | deskagents |
| Workflow filename | release.yml | testpypi.yml |
| GitHub environment | pypi | testpypi |

Create matching GitHub environments. The workflow's publish job uses OIDC
(`id-token: write`); no API token belongs in source. Pending publishers do not
reserve a name. See [PyPI's publisher documentation](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

## Validate a candidate

Use a fresh checkout or empty `dist` directory so artifacts from older versions
are not mixed into the upload. Run:

```sh
uv sync --locked --all-extras --group packaging
uv run --locked --all-extras --group packaging pytest -W error
uv run --locked --all-extras --group packaging ruff check .
uv run --locked --all-extras --group packaging ruff format --check .
uv run --locked --all-extras --group packaging mypy
uv build
uv run --locked --group packaging python scripts/check_distribution.py dist/*.whl dist/*.tar.gz
uv run --locked --group packaging twine check --strict dist/*.whl dist/*.tar.gz
uv run --locked python scripts/check_wheel_environments.py
```

Version changes require `uv lock` followed by a locked sync. CI includes Linux
Python 3.11–3.13, macOS 3.11, and Windows 3.11. A local macOS pass does not establish
that the remote matrix passed; inspect the actual workflow before publishing.

Run the bounded live tests with credentials already in the environment and models
available to your account (these make paid API calls):

```sh
uv run --all-extras python scripts/live_provider_smoke.py --provider deepseek --model YOUR_DEEPSEEK_MODEL
uv run --all-extras python scripts/live_provider_smoke.py --provider openai --model YOUR_OPENAI_MODEL
```

Optional `--base-url` selects a compatible endpoint. Output contains outcome flags
and endpoint/model/date, not replies or credentials. Missing credentials are a
failed live check, not a successful skip. Deterministic CI uses mocked HTTP.

## Rehearse

After committing/pushing the reviewed candidate and configuring its publisher,
run **Rehearse on TestPyPI** manually on that exact ref. This workflow runs CI,
builds/checks the wheel and source archive, and publishes only to TestPyPI.
Use a new candidate version if an artifact with the same version already exists.

Download the exact wheel for `deskagents==0.1.0a1` from its TestPyPI release page,
then install that local wheel in a fresh environment with dependencies from normal
PyPI. Run `scripts/check_installation.py` and the examples outside the checkout.
Do not combine indexes as a workaround for missing TestPyPI dependencies.

## Production

After the release decision, push the exact matching tag `v0.1.0a1`. Every pushed
`v*` tag enters the production workflow; do not use a release tag for rehearsal.
The workflow validates the tag/version, runs CI, checks artifacts, and uploads
through the `pypi` environment. Then install the exact version from public PyPI:

```sh
python -m pip install 'deskagents[deepseek]==0.1.0a1'
```

Verify the installed package, examples, README rendering and project links. Keep
alpha limitations visible. Do not promote to `0.1.0` merely because publishing
works; review the compatibility inventory and outstanding platform/live evidence.

The token previously found in the local Git remote must be revoked/rotated before
release. Removing it from the URL did not revoke it.

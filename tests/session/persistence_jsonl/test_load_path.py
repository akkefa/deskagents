"""Ported from packages/session/session-persistence-jsonl/tests/load-path.spec.ts.

Upstream's Loader ``unwrapExports`` becomes deepseek-cordis ``resolve_plugin`` on the
module reference, and the plugin's ``name``/``Config`` exports are read through
``inspect_plugin``.
"""

from __future__ import annotations

import pytest
from deepseek_cordis import inspect_plugin
from deepseek_cordis.loader import resolve_plugin

from deskagents.session import persistence_jsonl

SPEC = "packages/session/session-persistence-jsonl/tests/load-path.spec.ts"


@pytest.mark.upstream(f"{SPEC}::retains the JSONL namespace and configuration through the Loader")
def test_retains_the_namespace_and_configuration_through_the_loader() -> None:
    assert not hasattr(persistence_jsonl, "default")
    plugin = resolve_plugin("deskagents.session.persistence_jsonl")
    assert plugin is persistence_jsonl
    assert callable(persistence_jsonl.apply)
    meta = inspect_plugin(plugin)
    assert meta.name == "session-persistence-jsonl"
    assert meta.config is not None
    assert meta.config is getattr(persistence_jsonl, "Config", None)

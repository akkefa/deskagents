"""Reusable contract suites for third-party implementations of harness seams.

Each suite is plain ``async`` code with no test-runner dependency: a check raises
``AssertionError`` on failure and :class:`ContractSkip` when the backend lacks an optional
capability. Parametrize a test over the ``*_cases()`` lists, or call the ``run_*``
helpers to collect :class:`ContractResult` outcomes.
"""

from __future__ import annotations

from .live_write_contract import (
    LIVE_WRITE_PATH_CONTRACT_SUITE,
    LiveWriteBackend,
    LiveWriteBackendFactory,
    LiveWriteCheck,
    live_write_path_contract_cases,
    run_live_write_path_contract,
)
from .llm_adapter_contract import (
    LLM_ADAPTER_CONTRACT_SUITE,
    AdapterContractBackend,
    AdapterContractFactory,
    AdapterContractResult,
    ScriptedAdapter,
    llm_adapter_contract_cases,
    run_llm_adapter_contract,
)
from .persistence_contract import (
    PERSISTENCE_CONTRACT_SUITE,
    ContractBackend,
    ContractBackendFactory,
    ContractBackendInstance,
    ContractCheck,
    ContractResult,
    ContractSkip,
    meta,
    one_turn_log,
    persistence_contract_cases,
    run_persistence_contract,
    second_turn,
)
from .plugin_contract import (
    HARNESS_EVENTS,
    PLUGIN_CONTRACT_SUITE,
    PluginContractBackend,
    PluginContractFactory,
    PluginContractResult,
    host_snapshot,
    plugin_contract_cases,
    run_plugin_contract,
)

__all__ = [
    "HARNESS_EVENTS",
    "LIVE_WRITE_PATH_CONTRACT_SUITE",
    "LLM_ADAPTER_CONTRACT_SUITE",
    "PERSISTENCE_CONTRACT_SUITE",
    "PLUGIN_CONTRACT_SUITE",
    "AdapterContractBackend",
    "AdapterContractFactory",
    "AdapterContractResult",
    "ContractBackend",
    "ContractBackendFactory",
    "ContractBackendInstance",
    "ContractCheck",
    "ContractResult",
    "ContractSkip",
    "LiveWriteBackend",
    "LiveWriteBackendFactory",
    "LiveWriteCheck",
    "PluginContractBackend",
    "PluginContractFactory",
    "PluginContractResult",
    "ScriptedAdapter",
    "host_snapshot",
    "live_write_path_contract_cases",
    "llm_adapter_contract_cases",
    "meta",
    "one_turn_log",
    "persistence_contract_cases",
    "plugin_contract_cases",
    "run_live_write_path_contract",
    "run_llm_adapter_contract",
    "run_persistence_contract",
    "run_plugin_contract",
    "second_turn",
]

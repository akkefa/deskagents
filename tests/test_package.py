"""The installed package imports cleanly and declares itself typed."""

from __future__ import annotations

import importlib.resources
import logging

import deskagents


def test_package_import() -> None:
    assert deskagents.__name__ == "deskagents"
    assert all(hasattr(deskagents, name) for name in deskagents.__all__)


def test_package_is_typed() -> None:
    assert importlib.resources.files("deskagents").joinpath("py.typed").is_file()


def test_package_logger_is_quiet_by_default() -> None:
    handlers = logging.getLogger("deskagents").handlers
    assert any(isinstance(handler, logging.NullHandler) for handler in handlers)

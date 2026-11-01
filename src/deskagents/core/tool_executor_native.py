"""Native tool executor plugin.

Upstream: packages/core/tool-executor-native at ce7073d.
"""

from .tools import NativeToolExecutor, install_native_tool_executor

apply = install_native_tool_executor

__all__ = ["NativeToolExecutor", "apply", "install_native_tool_executor"]

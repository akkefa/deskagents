"""Python implementation of the DeepSeek Harness agent runtime.

Modules follow upstream packages one to one; docs/upstream.json maps each upstream
package to its module and records the pinned upstream commit.
"""

from __future__ import annotations

import logging

# Let applications configure logging. Without this, an unconfigured process
# would emit library errors through logging's last-resort stderr handler.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__: list[str] = []

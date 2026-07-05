from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


class PolicyError(ValueError):
    pass


@dataclass(frozen=True)
class ToolPolicy:
    disabled_tools: set[str]
    max_tool_argument_bytes: int

    def is_allowed(self, tool_name: str) -> bool:
        return tool_name not in self.disabled_tools

    def filter_tool_names(self, tool_names: list[str]) -> set[str]:
        return {name for name in tool_names if self.is_allowed(name)}

    def validate_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        allowed_tool_names: set[str],
    ) -> None:
        if tool_name not in allowed_tool_names:
            raise PolicyError(f"Tool is not available: {tool_name}")
        if not self.is_allowed(tool_name):
            raise PolicyError(f"Tool is not allowed: {tool_name}")
        encoded = json.dumps(arguments, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        if len(encoded) > self.max_tool_argument_bytes:
            raise PolicyError(f"Tool arguments exceed {self.max_tool_argument_bytes} bytes")

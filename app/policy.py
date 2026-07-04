from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.config import Settings


class PolicyError(ValueError):
    pass


@dataclass(frozen=True)
class ToolPolicy:
    allowed_tools: set[str]
    allow_all_tools: bool
    max_tool_argument_bytes: int

    @classmethod
    def from_settings(cls, settings: Settings) -> "ToolPolicy":
        return cls(
            allowed_tools=settings.mcp_allowed_tools,
            allow_all_tools=settings.mcp_allow_all_tools,
            max_tool_argument_bytes=settings.max_tool_argument_bytes,
        )

    def is_allowed(self, tool_name: str) -> bool:
        return self.allow_all_tools or tool_name in self.allowed_tools

    def filter_tool_names(self, tool_names: list[str]) -> set[str]:
        if self.allow_all_tools:
            return set(tool_names)
        return {name for name in tool_names if name in self.allowed_tools}

    def validate_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> None:
        if not self.is_allowed(tool_name):
            raise PolicyError(f"Tool is not allowed: {tool_name}")
        encoded = json.dumps(arguments, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        if len(encoded) > self.max_tool_argument_bytes:
            raise PolicyError(f"Tool arguments exceed {self.max_tool_argument_bytes} bytes")


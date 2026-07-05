from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.mcp.schema import CachedTool


@dataclass(frozen=True)
class ToolCacheSnapshot:
    tools: tuple[CachedTool, ...]

    @property
    def is_valid(self) -> bool:
        return bool(self.tools)

    def public_tools(self, include_schema: bool = False) -> list[dict[str, Any]]:
        result = []
        for tool in self.tools:
            item: dict[str, Any] = {"name": tool.name, "description": tool.description}
            if include_schema:
                item["input_schema"] = tool.input_schema
            result.append(item)
        return result


class ToolCache:
    def __init__(self) -> None:
        self._snapshot = ToolCacheSnapshot(())
        self.last_error: str | None = None

    def snapshot(self) -> ToolCacheSnapshot:
        return self._snapshot

    @property
    def is_valid(self) -> bool:
        return self._snapshot.is_valid

    def replace(self, snapshot: ToolCacheSnapshot) -> None:
        self._snapshot = snapshot
        self.last_error = None

    def mark_error(self, error: str) -> None:
        self.last_error = error

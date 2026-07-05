from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CachedTool:
    name: str
    description: str
    input_schema: dict[str, Any]


def normalize_mcp_tool(tool: Any) -> CachedTool | None:
    name = getattr(tool, "name", None)
    if not isinstance(name, str) or not name:
        return None

    description = getattr(tool, "description", None) or ""
    input_schema = getattr(tool, "inputSchema", None)
    if input_schema is not None and not isinstance(input_schema, dict):
        return None
    if not input_schema:
        input_schema = {"type": "object", "properties": {}, "additionalProperties": False}
    if input_schema.get("type") != "object":
        return None
    if not isinstance(input_schema.get("properties", {}), dict):
        return None

    return CachedTool(name=name, description=str(description), input_schema=input_schema)

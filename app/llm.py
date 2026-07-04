from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from app.tool_schema import CachedTool


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    status: str
    result: str
    truncated: bool


@dataclass(frozen=True)
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    provider_state: Any = None


class LLMClient(Protocol):
    @property
    def provider(self) -> str:
        ...

    @property
    def model(self) -> str:
        ...

    def prepare_tools(self, tools: tuple[CachedTool, ...]) -> list[Any]:
        ...

    def start_conversation(self, user_text: str) -> Any:
        ...

    async def create_response(self, conversation: Any, tools: list[Any]) -> LLMResponse:
        ...

    def append_response(self, conversation: Any, response: LLMResponse) -> None:
        ...

    def append_tool_result(self, conversation: Any, tool_call: ToolCall, result: ToolResult) -> None:
        ...

    async def close(self) -> None:
        ...

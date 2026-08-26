from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.messages import ModelMessage


class AgentOutput(BaseModel):
    text: str


ToolSignature = tuple[str, str]


@dataclass(frozen=True)
class ToolCallMetadata:
    server: str
    tool_name: str
    signature: ToolSignature


@dataclass
class RunState:
    attempted_tool_calls: int = 0
    dispatched_tool_calls: int = 0
    completed_tool_calls: int = 0
    failed_signatures: set[ToolSignature] = field(default_factory=set)
    in_flight_calls: dict[str, ToolCallMetadata] = field(default_factory=dict)
    unknown_outcomes: dict[str, ToolCallMetadata] = field(default_factory=dict)
    last_server: str | None = None
    last_tool_name: str | None = None
    approval_resume_only: bool = False
    approved_tool_call_id: str | None = None


@dataclass
class AgentDependencies:
    generation: Any
    user_id: int
    chat_id: int
    update_id: int
    request_id: str
    run_state: RunState = field(default_factory=RunState)


class PendingApproval(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    approval_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    generation_id: str
    user_id: int
    chat_id: int
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    arguments_hash: str
    message_history: list[ModelMessage]
    expires_at: float
    used: bool = False

    @classmethod
    def create(cls, *, generation_id: str, user_id: int, chat_id: int, tool_call_id: str, tool_name: str, arguments: dict[str, Any], message_history: list[ModelMessage], ttl_seconds: int = 300) -> "PendingApproval":
        encoded = json.dumps(arguments, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return cls(
            generation_id=generation_id,
            user_id=user_id,
            chat_id=chat_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=json.loads(encoded),
            arguments_hash=hashlib.sha256(encoded.encode()).hexdigest(),
            message_history=message_history,
            expires_at=time.monotonic() + ttl_seconds,
        )


class ApprovalStore:
    def __init__(self) -> None:
        self._items: dict[str, PendingApproval] = {}
        self._lock = asyncio.Lock()

    async def add(self, approval: PendingApproval) -> None:
        async with self._lock:
            self._items[approval.approval_id] = approval

    async def consume(self, approval_id: str, user_id: int, chat_id: int) -> PendingApproval | None:
        async with self._lock:
            approval = self._items.get(approval_id)
            if approval is None or approval.expires_at <= time.monotonic():
                return None
            if approval.used or approval.user_id != user_id or approval.chat_id != chat_id:
                return None
            self._items.pop(approval_id, None)
            return approval.model_copy(update={"used": True})

    async def purge(self) -> list[PendingApproval]:
        async with self._lock:
            return self._purge_locked()

    def _purge_locked(self) -> list[PendingApproval]:
        now = time.monotonic()
        expired = [item for item in self._items.values() if item.expires_at <= now]
        for item in expired:
            self._items.pop(item.approval_id, None)
        return expired

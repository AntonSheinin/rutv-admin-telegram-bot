import asyncio
import time

import pytest

from app.agent.models import ApprovalStore, PendingApproval


def make_approval() -> PendingApproval:
    return PendingApproval.create(
        generation_id="generation",
        user_id=1,
        chat_id=2,
        tool_call_id="call",
        tool_name="server_tool",
        arguments={},
        message_history=[],
        ttl_seconds=1,
    )


@pytest.mark.asyncio
async def test_expired_approval_is_left_for_maintenance_purge():
    store = ApprovalStore()
    approval = make_approval().model_copy(update={"expires_at": time.monotonic() - 1})
    await store.add(approval)
    assert await store.consume(approval.approval_id, 1, 2) is None
    assert await store.purge() == [approval]


@pytest.mark.asyncio
async def test_approval_can_only_be_consumed_once():
    store = ApprovalStore()
    approval = make_approval()
    await store.add(approval)
    assert (await store.consume(approval.approval_id, 1, 2)) is not None
    assert await store.consume(approval.approval_id, 1, 2) is None

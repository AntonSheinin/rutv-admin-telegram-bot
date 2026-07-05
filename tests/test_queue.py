import pytest
import asyncio

from app.core.structured_log import StructuredLogger
from app.core.config import Settings
from app.telegram.queue import TTLUpdateDedupe, WebhookQueue

from tests.test_config import base_env


@pytest.mark.asyncio
async def test_dedupe_marks_duplicate():
    dedupe = TTLUpdateDedupe(60)
    assert await dedupe.mark_new(10) is True
    assert await dedupe.mark_new(10) is False


@pytest.mark.asyncio
async def test_queue_full_returns_full():
    env = base_env()
    env["UPDATE_QUEUE_MAX_SIZE"] = "1"
    env["UPDATE_WORKER_COUNT"] = "1"
    env["SHUTDOWN_DRAIN_TIMEOUT_SECONDS"] = "1"
    settings = Settings.from_env(env)
    handler_started = asyncio.Event()
    release_handler = asyncio.Event()

    async def handler(update):
        handler_started.set()
        await release_handler.wait()

    queue = WebhookQueue(settings, StructuredLogger(), handler)
    await queue.start()
    try:
        assert await queue.enqueue({"update_id": 1}) == "enqueued"
        await asyncio.wait_for(handler_started.wait(), timeout=1)
        assert await queue.enqueue({"update_id": 2}) == "enqueued"
        assert await queue.enqueue({"update_id": 3}) == "full"
        assert await queue.enqueue({"update_id": 3}) == "full"
    finally:
        release_handler.set()
        await queue.shutdown()


@pytest.mark.asyncio
async def test_user_locks_are_pruned_after_handled_update():
    settings = Settings.from_env(base_env())

    async def handler(update):
        return None

    queue = WebhookQueue(settings, StructuredLogger(), handler)
    await queue._handle_update({"update_id": 1, "message": {"from": {"id": 42}}})

    assert queue._user_locks == {}

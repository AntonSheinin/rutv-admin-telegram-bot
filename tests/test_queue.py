import pytest

from app.audit import AuditLogger
from app.config import Settings
from app.webhook_queue import TTLUpdateDedupe, WebhookQueue

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
    settings = Settings.from_env(env)

    async def handler(update):
        return None

    queue = WebhookQueue(settings, AuditLogger(), handler)
    queue._accepting = True
    status1 = await queue.enqueue({"update_id": 1})
    status2 = await queue.enqueue({"update_id": 2})
    assert status1 == "enqueued"
    assert status2 == "full"
    assert await queue.enqueue({"update_id": 2}) == "full"

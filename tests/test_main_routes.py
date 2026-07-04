from fastapi.testclient import TestClient
import pytest

from app import main
from app.audit import AuditLogger
from app.config import Settings
from app.mcp_client import ToolCacheSnapshot
from app.tool_schema import CachedTool

from tests.test_config import base_env


def configure_minimal_ready_state():
    service = main.ServiceState()
    settings = Settings.from_env(base_env())
    service.settings = settings
    service.audit = AuditLogger()
    service.config_valid = True
    service.webhook_registered = True
    service.telegram = object()
    service.llm = object()
    service.mcp = object()
    service.policy = object()
    service.agent = object()

    class Queue:
        workers_running = True

        async def enqueue(self, update):
            return "enqueued"

    service.queue = Queue()
    tool = CachedTool(
        name="refresh_playlist",
        description="Refresh",
        input_schema={"type": "object", "properties": {}},
    )
    return service, settings, ToolCacheSnapshot((tool,))


@pytest.fixture()
def client():
    service, _, _ = configure_minimal_ready_state()
    main.app.state.service = service
    return TestClient(main.app)


def test_ready_reports_degraded_when_cache_missing(client):
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


def test_tools_requires_admin_bearer(client):
    response = client.get("/tools", headers={"Authorization": "Bearer wrong"})
    assert response.status_code == 401


def test_tools_returns_public_tool_list(client):
    service, settings, snapshot = configure_minimal_ready_state()
    main.app.state.service = service

    async def replace():
        await service.tool_cache.replace(snapshot)

    import anyio

    anyio.run(replace)
    response = client.get("/tools", headers={"Authorization": f"Bearer {settings.admin_api_token}"})

    assert response.status_code == 200
    assert response.json()["tools"] == [{"name": "refresh_playlist", "description": "Refresh"}]


def test_webhook_validates_secret(client):
    response = client.post("/telegram/webhook", json={"update_id": 1}, headers={"X-Telegram-Bot-Api-Secret-Token": "bad"})
    assert response.status_code == 401


def test_webhook_enqueues_with_valid_secret(client):
    service = main.app.state.service
    response = client.post(
        "/telegram/webhook",
        json={"update_id": 1},
        headers={"X-Telegram-Bot-Api-Secret-Token": service.settings.telegram_webhook_secret},
    )
    assert response.status_code == 200
    assert response.json() == {"status": "enqueued"}

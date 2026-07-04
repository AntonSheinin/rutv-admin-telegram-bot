from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.agent_factory import create_agent_runner
from app.audit import AuditLogger
from app.config import ConfigError, Settings
from app.llm_factory import create_llm_client
from app.main_state import ServiceState, ToolReloadError, reload_tools
from app.mcp_client import McpClient
from app.policy import ToolPolicy
from app.telegram_bot import TelegramBotService, TelegramServiceError
from app.telegram_handlers import process_update
from app.webhook_queue import WebhookQueue


admin_bearer = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = ServiceState()
    app.state.service = service
    await startup(service)
    try:
        yield
    finally:
        await shutdown(service)


app = FastAPI(title="RuTV Admin Bot", lifespan=lifespan)


def get_state(request: Request) -> ServiceState:
    return request.app.state.service


async def require_admin_api(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(admin_bearer),
) -> None:
    service = get_state(request)
    if service.settings is None or credentials is None:
        raise HTTPException(status_code=401, detail={"status": "error", "reason": "unauthorized"})
    if credentials.scheme.lower() != "bearer" or credentials.credentials != service.settings.admin_api_token:
        raise HTTPException(status_code=401, detail={"status": "error", "reason": "unauthorized"})


async def startup(service: ServiceState) -> None:
    try:
        settings = Settings.from_env()
        audit = AuditLogger(settings.audit_log_path, settings.log_level)
        policy = ToolPolicy.from_settings(settings)
        telegram = TelegramBotService(settings, audit)
        llm = create_llm_client(settings)
        mcp = McpClient(settings, audit)
        agent = create_agent_runner(settings, audit, llm, mcp, policy)
        service.settings = settings
        service.audit = audit
        service.policy = policy
        service.telegram = telegram
        service.llm = llm
        service.mcp = mcp
        service.agent = agent
        service.config_valid = True
        service.queue = WebhookQueue(settings, audit, lambda update: process_update(service, update))
        await service.queue.start()
    except ConfigError as exc:
        service.startup_error = str(exc)
        service.audit = AuditLogger()
        service.audit.error("startup_degraded", reason="config_invalid", error=str(exc))
        return
    except ValueError as exc:
        service.startup_error = str(exc)
        if service.audit is None:
            service.audit = AuditLogger()
        service.audit.error("startup_degraded", reason="initialization_failed", error=str(exc))
        return

    try:
        await reload_tools(service)
    except ToolReloadError:
        pass
    try:
        await service.telegram.register_webhook()
        service.webhook_registered = True
        service.audit.info("telegram_webhook_registered")
    except TelegramServiceError as exc:
        service.webhook_registered = False
        service.audit.warning("startup_degraded", reason="webhook_registration_failed", error=str(exc))


async def shutdown(service: ServiceState) -> None:
    if service.queue is not None:
        await service.queue.shutdown()
    if service.telegram is not None:
        await service.telegram.close()
    if service.llm is not None:
        await service.llm.close()
    if service.audit is not None:
        service.audit.flush()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready(request: Request, response: Response) -> dict[str, str]:
    reason = get_state(request).ready_reason()
    if reason is not None:
        response.status_code = 503
        return {"status": "degraded", "reason": reason}
    return {"status": "ok"}


@app.post("/telegram/webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
) -> dict[str, str]:
    service = get_state(request)
    if service.settings is None or service.queue is None:
        raise HTTPException(status_code=503, detail={"status": "degraded", "reason": "service_not_initialized"})
    if x_telegram_bot_api_secret_token != service.settings.telegram_webhook_secret:
        raise HTTPException(status_code=401, detail={"status": "error", "reason": "invalid_webhook_secret"})
    update = await request.json()
    if service.audit:
        service.audit.debug("webhook_received", update_id=update.get("update_id"))
    status = await service.queue.enqueue(update)
    if status == "full":
        raise HTTPException(status_code=503, detail={"status": "degraded", "reason": "queue_full"})
    if status == "stopped":
        raise HTTPException(status_code=503, detail={"status": "degraded", "reason": "queue_stopped"})
    return {"status": status}


@app.get("/tools")
async def tools(
    request: Request,
    include_schema: bool = False,
    _: None = Depends(require_admin_api),
) -> dict[str, Any]:
    snapshot = await get_state(request).tool_cache.snapshot()
    return {"status": "ok", "tools": snapshot.public_tools(include_schema=include_schema)}


@app.post("/tools/reload")
async def tools_reload(request: Request, _: None = Depends(require_admin_api)) -> dict[str, Any]:
    try:
        return await reload_tools(get_state(request))
    except ToolReloadError as exc:
        raise HTTPException(status_code=503, detail={"status": "degraded", "reason": str(exc)}) from exc


@app.post("/telegram/webhook/register")
async def webhook_register(request: Request, _: None = Depends(require_admin_api)) -> dict[str, str]:
    service = get_state(request)
    if service.telegram is None:
        raise HTTPException(status_code=503, detail={"status": "degraded", "reason": "telegram_unavailable"})
    try:
        await service.telegram.register_webhook()
        service.webhook_registered = True
        if service.audit:
            service.audit.info("telegram_webhook_registered", manual=True)
        return {"status": "ok"}
    except TelegramServiceError as exc:
        service.webhook_registered = False
        if service.audit:
            service.audit.warning("telegram_webhook_register_failed", error=str(exc), manual=True)
        raise HTTPException(status_code=503, detail={"status": "degraded", "reason": "webhook_registration_failed"})

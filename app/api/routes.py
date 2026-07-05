from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.state import ServiceState, ToolReloadError, reload_tools
from app.telegram.bot import TelegramServiceError


router = APIRouter()
admin_bearer = HTTPBearer(auto_error=False)


def get_state(request: Request) -> ServiceState:
    return request.app.state.service


async def require_admin_api(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(admin_bearer),
) -> None:
    service = get_state(request)
    if service.settings is None or credentials is None:
        raise HTTPException(status_code=401, detail={"status": "error", "reason": "unauthorized"})
    if credentials.scheme.lower() != "bearer" or credentials.credentials != service.settings.mcp_auth_token:
        raise HTTPException(status_code=401, detail={"status": "error", "reason": "unauthorized"})


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request, response: Response) -> dict[str, str]:
    reason = get_state(request).ready_reason()
    if reason is not None:
        response.status_code = 503
        return {"status": "degraded", "reason": reason}
    return {"status": "ok"}


@router.post("/telegram/webhook")
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
    if service.logger:
        service.logger.debug("webhook_received", update_id=update.get("update_id"))
    status = await service.queue.enqueue(update)
    if status == "full":
        raise HTTPException(status_code=503, detail={"status": "degraded", "reason": "queue_full"})
    if status == "stopped":
        raise HTTPException(status_code=503, detail={"status": "degraded", "reason": "queue_stopped"})
    return {"status": status}


@router.get("/tools")
async def tools(
    request: Request,
    include_schema: bool = False,
    _: None = Depends(require_admin_api),
) -> dict[str, Any]:
    snapshot = get_state(request).tool_cache.snapshot()
    return {"status": "ok", "tools": snapshot.public_tools(include_schema=include_schema)}


@router.post("/tools/reload")
async def tools_reload(request: Request, _: None = Depends(require_admin_api)) -> dict[str, Any]:
    try:
        return await reload_tools(get_state(request))
    except ToolReloadError as exc:
        raise HTTPException(status_code=503, detail={"status": "degraded", "reason": str(exc)}) from exc


@router.post("/telegram/webhook/register")
async def webhook_register(request: Request, _: None = Depends(require_admin_api)) -> dict[str, str]:
    service = get_state(request)
    if service.telegram is None:
        raise HTTPException(status_code=503, detail={"status": "degraded", "reason": "telegram_unavailable"})
    try:
        await service.telegram.register_webhook()
        service.webhook_registered = True
        if service.logger:
            service.logger.info("telegram_webhook_registered", manual=True)
        return {"status": "ok"}
    except TelegramServiceError as exc:
        service.webhook_registered = False
        if service.logger:
            service.logger.warning("telegram_webhook_register_failed", error=str(exc), manual=True)
        raise HTTPException(status_code=503, detail={"status": "degraded", "reason": "webhook_registration_failed"})

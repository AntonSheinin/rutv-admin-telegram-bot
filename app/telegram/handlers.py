from __future__ import annotations

from app.core.state import ServiceState, ToolReloadError, reload_tools
from app.telegram.bot import extract_message


async def process_update(service: ServiceState, update: dict) -> None:
    if not service.telegram or not service.agent_service or not service.logger:
        return
    extracted = extract_message(update)
    if extracted is None:
        return
    update_id, user_id, chat_id, text = extracted
    if not service.telegram.is_admin(user_id):
        await service.telegram.send_text(chat_id, "Unauthorized.")
        return
    if text.startswith("/"):
        await handle_command(service, user_id, chat_id, text)
        return
    reason = service.ready_reason()
    if reason is not None or service.generation is None:
        await service.telegram.send_text(chat_id, f"No action was performed. Service is degraded: {reason or 'mcp_tools_unavailable'}.")
        return

    async def run_agent() -> str:
        return await service.agent_service.handle_message(
            service.generation, text, update_id=update_id, user_id=user_id, chat_id=chat_id
        )

    try:
        response = await service.telegram.with_typing(chat_id, run_agent)
    except Exception as exc:
        service.logger.error("agent_run_failed", update_id=update_id, error=str(exc))
        response = "No action was performed because the agent request failed."
    await service.telegram.send_text(chat_id, response)


async def handle_command(service: ServiceState, user_id: int, chat_id: int, text: str) -> None:
    if service.telegram is None:
        return
    command = text.split(maxsplit=1)[0].split("@", 1)[0].lower()
    if command == "/start":
        await service.telegram.send_text(chat_id, "RuTV admin bot is available.")
    elif command == "/help":
        await service.telegram.send_text(chat_id, "Commands: /health, /tools, /reload_tools, /confirm <id>.")
    elif command == "/health":
        reason = service.ready_reason()
        await service.telegram.send_text(chat_id, "Status: ok" if reason is None else f"Status: degraded ({reason})")
    elif command == "/tools":
        generation = service.generation
        if generation is None:
            await service.telegram.send_text(chat_id, "No MCP tools are currently available.")
        else:
            lines = [f"{name}: {status['status']}" for name, status in generation.statuses.items()]
            await service.telegram.send_text(chat_id, "\n".join(lines) or "No MCP servers are configured.")
    elif command == "/reload_tools":
        try:
            result = await reload_tools(service)
            await service.telegram.send_text(chat_id, f"Reloaded MCP tools: {result['tool_count']}")
        except ToolReloadError as exc:
            await service.telegram.send_text(chat_id, f"Tool reload failed: {exc}")
    elif command == "/confirm":
        parts = text.split(maxsplit=1)
        if len(parts) != 2 or service.agent_service is None:
            await service.telegram.send_text(chat_id, "Usage: /confirm <confirmation_id>")
        else:
            response = await service.agent_service.confirm(
                service.generations, parts[1].strip(), user_id=user_id, chat_id=chat_id
            )
            await service.telegram.send_text(chat_id, response)
    else:
        await service.telegram.send_text(chat_id, "Unknown command. Use /help.")

from __future__ import annotations

from app.core.state import ServiceState, ToolReloadError, reload_tools
from app.telegram.bot import extract_message


async def process_update(service: ServiceState, update: dict) -> None:
    if not service.telegram or not service.agent or not service.logger:
        return
    extracted = extract_message(update)
    if extracted is None:
        service.logger.debug("telegram_update_ignored", update_id=update.get("update_id"))
        return
    update_id, user_id, chat_id, text = extracted
    if not service.telegram.is_admin(user_id):
        await service.telegram.send_text(chat_id, "Unauthorized.")
        service.logger.warning("telegram_unauthorized", update_id=update_id, user_id=user_id, chat_id=chat_id)
        return
    if text.startswith("/"):
        await handle_command(service, chat_id, text)
        return

    reason = service.ready_reason()
    if reason is not None:
        await service.telegram.send_text(chat_id, f"No action was performed. Service is degraded: {reason}.")
        return

    snapshot = service.tool_cache.snapshot()

    async def run_agent() -> str:
        result = await service.agent.handle_message(text, snapshot, update_id=update_id, user_id=user_id, chat_id=chat_id)
        return result.text

    response = await service.telegram.with_typing(chat_id, run_agent)
    await service.telegram.send_text(chat_id, response)


async def handle_command(service: ServiceState, chat_id: int, text: str) -> None:
    if service.telegram is None:
        return

    command = text.split(maxsplit=1)[0].split("@", 1)[0].lower()
    if command == "/start":
        await service.telegram.send_text(chat_id, "RuTV admin bot is available.")
    elif command == "/help":
        await service.telegram.send_text(chat_id, "Commands: /health, /tools, /reload_tools. Send natural language requests for playlist actions.")
    elif command == "/health":
        reason = service.ready_reason()
        if reason is None:
            await service.telegram.send_text(chat_id, "Status: ok")
        else:
            await service.telegram.send_text(chat_id, f"Status: degraded ({reason})")
    elif command == "/tools":
        snapshot = service.tool_cache.snapshot()
        if not snapshot.tools:
            await service.telegram.send_text(chat_id, "No MCP tools are currently available.")
        else:
            lines = [f"- {tool.name}: {tool.description}" for tool in snapshot.tools]
            await service.telegram.send_text(chat_id, "\n".join(lines))
    elif command == "/reload_tools":
        try:
            result = await reload_tools(service)
            await service.telegram.send_text(chat_id, f"Reloaded MCP tools: {result['tool_count']}")
        except ToolReloadError as exc:
            await service.telegram.send_text(chat_id, f"Tool reload failed: {exc}")
    else:
        await service.telegram.send_text(chat_id, "Unknown command. Use /help.")

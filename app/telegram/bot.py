from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest, TelegramForbiddenError, TelegramRetryAfter, TelegramServerError

from app.core.config import Settings

SAFE_MESSAGE_LIMIT = 3900


class TelegramBotService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.bot = Bot(token=settings.telegram_bot_token, default=DefaultBotProperties(parse_mode=None))

    async def register_webhook(self) -> None:
        try:
            await self.bot.set_webhook(
                url=self.settings.telegram_webhook_url,
                secret_token=self.settings.telegram_webhook_secret,
                drop_pending_updates=False,
            )
        except TelegramAPIError as exc:
            raise TelegramServiceError(f"Failed to register Telegram webhook: {exc}") from exc

    async def close(self) -> None:
        await self.bot.session.close()

    def is_admin(self, user_id: int | None) -> bool:
        return user_id in self.settings.telegram_admin_user_ids if user_id is not None else False

    async def send_text(self, chat_id: int, text: str) -> None:
        chunks = split_telegram_text(text)
        for chunk in chunks:
            await self._send_with_retry(chat_id, chunk)

    async def _send_with_retry(self, chat_id: int, text: str) -> None:
        delay = 1.0
        for attempt in range(3):
            try:
                await self.bot.send_message(chat_id=chat_id, text=text)
                return
            except TelegramRetryAfter as exc:
                await asyncio.sleep(float(exc.retry_after))
            except TelegramServerError:
                if attempt == 2:
                    raise
                await asyncio.sleep(delay)
                delay *= 2
            except (TelegramForbiddenError, TelegramBadRequest):
                raise

    async def with_typing(self, chat_id: int, work: Callable[[], Awaitable[str]]) -> str:
        stop = asyncio.Event()
        task = asyncio.create_task(self._typing_loop(chat_id, stop))
        try:
            return await work()
        finally:
            stop.set()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _typing_loop(self, chat_id: int, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await self.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            except TelegramRetryAfter as exc:
                await asyncio.sleep(float(exc.retry_after))
                continue
            except TelegramServerError:
                await asyncio.sleep(1)
                continue
            except (TelegramForbiddenError, TelegramBadRequest):
                return
            try:
                await asyncio.wait_for(stop.wait(), timeout=4)
            except TimeoutError:
                continue


def extract_message(update: dict[str, Any]) -> tuple[int, int, int, str] | None:
    message = update.get("message") or update.get("edited_message")
    if not isinstance(message, dict):
        return None
    text = message.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    chat = message.get("chat") or {}
    user = message.get("from") or {}
    update_id = update.get("update_id")
    chat_id = chat.get("id")
    user_id = user.get("id")
    if not isinstance(update_id, int) or not isinstance(chat_id, int) or not isinstance(user_id, int):
        return None
    return update_id, user_id, chat_id, text.strip()


def split_telegram_text(text: str) -> list[str]:
    if not text:
        return ["No response."]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > SAFE_MESSAGE_LIMIT:
        split_at = remaining.rfind("\n", 0, SAFE_MESSAGE_LIMIT)
        if split_at <= 0:
            split_at = SAFE_MESSAGE_LIMIT
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip()
    chunks.append(remaining)
    return chunks


class TelegramServiceError(RuntimeError):
    pass

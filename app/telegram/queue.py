from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from app.core.structured_log import StructuredLogger
from app.core.config import Settings

UpdateHandler = Callable[[dict[str, Any]], Awaitable[None]]


class TTLUpdateDedupe:
    def __init__(self, ttl_seconds: int) -> None:
        self._ttl_seconds = ttl_seconds
        self._items: dict[int, float] = {}
        self._lock = asyncio.Lock()

    async def mark_new(self, update_id: int) -> bool:
        async with self._lock:
            now = time.monotonic()
            self._prune(now)
            if update_id in self._items:
                return False
            self._items[update_id] = now + self._ttl_seconds
            return True

    async def forget(self, update_id: int) -> None:
        async with self._lock:
            self._items.pop(update_id, None)

    def _prune(self, now: float) -> None:
        expired = [update_id for update_id, expires_at in self._items.items() if expires_at <= now]
        for update_id in expired:
            self._items.pop(update_id, None)


class WebhookQueue:
    def __init__(self, settings: Settings, logger: StructuredLogger, handler: UpdateHandler) -> None:
        self._settings = settings
        self._logger = logger
        self._handler = handler
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=settings.update_queue_max_size)
        self._dedupe = TTLUpdateDedupe(settings.update_dedupe_ttl_seconds)
        self._workers: list[asyncio.Task[None]] = []
        self._accepting = False
        self._user_locks: dict[int, asyncio.Lock] = {}

    @property
    def workers_running(self) -> bool:
        return bool(self._workers) and all(not task.done() for task in self._workers)

    async def start(self) -> None:
        self._accepting = True
        for idx in range(self._settings.update_worker_count):
            self._workers.append(asyncio.create_task(self._worker(idx)))

    async def enqueue(self, update: dict[str, Any]) -> str:
        update_id = update.get("update_id")
        if not isinstance(update_id, int):
            return "ignored"
        if not await self._dedupe.mark_new(update_id):
            return "duplicate"
        if not self._accepting:
            await self._dedupe.forget(update_id)
            return "stopped"
        try:
            self._queue.put_nowait(update)
        except asyncio.QueueFull:
            await self._dedupe.forget(update_id)
            self._logger.warning("queue_full", update_id=update_id)
            return "full"
        self._logger.debug("queue_enqueued", update_id=update_id)
        return "enqueued"

    async def shutdown(self) -> None:
        self._accepting = False
        try:
            await asyncio.wait_for(self._queue.join(), timeout=self._settings.shutdown_drain_timeout_seconds)
        except TimeoutError:
            pass
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def _worker(self, idx: int) -> None:
        self._logger.debug("worker_start", worker=idx)
        while True:
            update = await self._queue.get()
            try:
                await self._handle_update(update)
            except Exception as exc:
                self._logger.error("worker_error", worker=idx, error=str(exc), update_id=update.get("update_id"))
            finally:
                self._queue.task_done()

    async def _handle_update(self, update: dict[str, Any]) -> None:
        user_id = _extract_user_id(update)
        if user_id is None:
            await self._handler(update)
            return

        lock = self._user_locks.setdefault(user_id, asyncio.Lock())
        async with lock:
            await self._handler(update)
        if not lock.locked() and not _lock_has_waiters(lock):
            self._user_locks.pop(user_id, None)


def _extract_user_id(update: dict[str, Any]) -> int | None:
    message = update.get("message") or update.get("edited_message") or {}
    user = message.get("from") if isinstance(message, dict) else None
    user_id = user.get("id") if isinstance(user, dict) else None
    return user_id if isinstance(user_id, int) else None


def _lock_has_waiters(lock: asyncio.Lock) -> bool:
    waiters = getattr(lock, "_waiters", None)
    return bool(waiters)

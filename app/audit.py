from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

SENSITIVE_KEY_PARTS = ("token", "secret", "password", "key", "auth", "authorization", "bearer")


def redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if any(part in key_str.lower() for part in SENSITIVE_KEY_PARTS):
                redacted[key_str] = "[REDACTED]"
            else:
                redacted[key_str] = redact(item)
        return redacted
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


class AuditLogger:
    def __init__(self, path: str = "", level: str = "INFO") -> None:
        self._logger = logging.getLogger("rutv_admin_bot")
        self._logger.setLevel(_parse_level(level))
        self._logger.handlers.clear()
        if path:
            handler: logging.Handler = logging.FileHandler(path, encoding="utf-8")
        else:
            handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)

    def event(self, event: str, level: str = "INFO", **fields: Any) -> None:
        payload = {
            "ts": datetime.now(UTC).isoformat(),
            "level": level.upper(),
            "event": event,
            **redact(fields),
        }
        self._logger.log(_parse_level(level), json.dumps(payload, ensure_ascii=True, default=str))

    def debug(self, event: str, **fields: Any) -> None:
        self.event(event, level="DEBUG", **fields)

    def info(self, event: str, **fields: Any) -> None:
        self.event(event, level="INFO", **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self.event(event, level="WARNING", **fields)

    def error(self, event: str, **fields: Any) -> None:
        self.event(event, level="ERROR", **fields)

    def flush(self) -> None:
        for handler in self._logger.handlers:
            handler.flush()


def _parse_level(level: str) -> int:
    return getattr(logging, level.upper(), logging.INFO)

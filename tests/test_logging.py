import json

from app.core.structured_log import StructuredLogger


def test_structured_logger_redacts_sensitive_keys(capsys):
    logger = StructuredLogger("DEBUG")

    logger.info("test_event", token="secret", nested={"authorization": "bearer value", "safe": "ok"})

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["token"] == "[REDACTED]"
    assert payload["nested"]["authorization"] == "[REDACTED]"
    assert payload["nested"]["safe"] == "ok"


def test_structured_logger_log_level_filters_debug(capsys):
    logger = StructuredLogger("INFO")

    logger.debug("debug_event")
    logger.info("info_event")

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["event"] == "info_event"


def test_structured_logger_redacts_tuples(capsys):
    logger = StructuredLogger("INFO")

    logger.info("test_event", values=({"api_key": "secret"},))

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["values"][0]["api_key"] == "[REDACTED]"

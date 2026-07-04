import json

from app.audit import AuditLogger


def test_audit_redacts_sensitive_keys(tmp_path):
    path = tmp_path / "audit.jsonl"
    audit = AuditLogger(str(path), "DEBUG")

    audit.info("test_event", token="secret", nested={"authorization": "bearer value", "safe": "ok"})
    audit.flush()

    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["token"] == "[REDACTED]"
    assert payload["nested"]["authorization"] == "[REDACTED]"
    assert payload["nested"]["safe"] == "ok"


def test_audit_log_level_filters_debug(tmp_path):
    path = tmp_path / "audit.jsonl"
    audit = AuditLogger(str(path), "INFO")

    audit.debug("debug_event")
    audit.info("info_event")
    audit.flush()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["event"] == "info_event"


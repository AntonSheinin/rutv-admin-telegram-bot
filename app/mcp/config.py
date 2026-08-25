from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, field_validator, model_validator
from dotenv import dotenv_values


class McpConfigError(ValueError):
    pass


class McpServerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    display_name: str | None = None
    enabled: bool = True
    url: AnyHttpUrl
    auth_type: Literal["access_token", "bearer_token"] = "bearer_token"
    disabled_tools: frozenset[str] = frozenset()
    confirmation_required_tools: frozenset[str] = frozenset()
    connect_timeout_seconds: int = Field(default=10, ge=1)
    tool_timeout_seconds: int = Field(default=30, ge=1)
    max_concurrent_calls: int = Field(default=2, ge=1)
    max_tools: int = Field(default=30, ge=1)
    max_tool_description_bytes: int = Field(default=4096, ge=1)
    max_tool_schema_bytes: int = Field(default=16384, ge=1)
    expected_tenant: str | None = None
    tenant_identity_tool: str | None = None
    tenant_identity_field: str | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", value):
            raise ValueError("must use lowercase letters, numbers, and underscores")
        return value

    @model_validator(mode="after")
    def validate_tenant(self) -> "McpServerConfig":
        values = (self.expected_tenant, self.tenant_identity_tool, self.tenant_identity_field)
        if any(values) and not all(values):
            raise ValueError("expected_tenant, tenant_identity_tool, and tenant_identity_field are required together")
        return self

    @property
    def token_env_name(self) -> str:
        return f"MCP_{self.name.upper()}_AUTH_TOKEN"

    @property
    def label(self) -> str:
        return self.display_name or self.name


class McpConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    servers: tuple[McpServerConfig, ...]
    max_total_tools: int = Field(default=100, ge=1)

    @model_validator(mode="after")
    def unique_names(self) -> "McpConfig":
        names = [server.name for server in self.servers]
        if len(names) != len(set(names)):
            raise ValueError("server names must be unique")
        return self


def default_mcp_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "mcp_config.toml"


def load_mcp_config(env: dict[str, str] | None = None, path: Path | None = None) -> tuple[McpConfig, dict[str, str]]:
    if env is None:
        source = {
            key: value
            for key, value in dotenv_values(Path(__file__).resolve().parents[2] / ".env").items()
            if value is not None
        }
        source.update(os.environ)
    else:
        source = dict(env)
    target = path or default_mcp_config_path()
    try:
        raw = tomllib.loads(target.read_text(encoding="utf-8"))
        config = McpConfig.model_validate(raw)
    except (OSError, tomllib.TOMLDecodeError, ValueError) as exc:
        raise McpConfigError(f"invalid MCP config: {exc}") from exc
    tokens: dict[str, str] = {}
    for server in config.servers:
        token = source.get(server.token_env_name, "").strip()
        if server.enabled and not token:
            raise McpConfigError(f"{server.token_env_name} is required for enabled server {server.name}")
        if token:
            tokens[server.name] = token
    return config, tokens

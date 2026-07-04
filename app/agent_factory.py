from __future__ import annotations

from app.agent import AgentRunner, Agent
from app.audit import AuditLogger
from app.config import Settings
from app.llm import LLMClient
from app.mcp_client import McpClient
from app.policy import ToolPolicy


def create_agent_runner(
    settings: Settings,
    audit: AuditLogger,
    llm: LLMClient,
    mcp: McpClient,
    policy: ToolPolicy,
) -> AgentRunner:
    return Agent(settings, audit, llm, mcp, policy)


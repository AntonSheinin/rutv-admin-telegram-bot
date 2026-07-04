# AGENTS.md

## Project
RuTV admin Telegram bot: a single-replica FastAPI webhook service with an embedded agent, configurable LLM boundary, and MCP playlist-service tooling.

## Project Structure
- `app/main.py`: FastAPI app, lifespan, dependencies, and HTTP routes.
- `app/main_state.py`: service state, readiness checks, and MCP tool reload logic.
- `app/telegram_handlers.py`: Telegram update routing and slash-command handling.
- `app/telegram_bot.py`: aiogram bot wrapper, webhook registration, message sending, typing indicator, and Telegram text splitting.
- `app/webhook_queue.py`: bounded in-memory update queue, worker tasks, update dedupe, and per-user/global concurrency.
- `app/agent.py`: provider-neutral embedded agent orchestration and tool-call execution.
- `app/agent_factory.py`: agent runner construction seam.
- `app/llm.py`: provider-neutral LLM protocol and DTOs.
- `app/llm_factory.py`: LLM provider selection seam.
- `app/openai_llm.py`: OpenAI-specific LLM implementation and OpenAI tool/result formatting.
- `app/mcp_client.py`: official MCP SDK client wrapper, MCP sessions, tool loading, tool cache.
- `app/tool_schema.py`: provider-neutral MCP tool schema normalization.
- `app/policy.py`: MCP tool allowlist and argument-size policy.
- `app/audit.py`: structured JSON logging and recursive redaction.
- `tests/`: focused unit/integration tests for config, routes, queue, agent, LLM, MCP policy/schema, Telegram helpers, and logging.

## Engineering Rules
- Keep the implementation KISS/YAGNI. Do not add databases, Redis, durable queues, multi-replica coordination, or a separate agent service unless explicitly requested.
- Keep modules focused. `main.py` should stay mostly FastAPI lifespan, dependencies, and routes.
- Avoid very long functions. Prefer extracting private helpers when a function mixes orchestration, I/O, formatting, and error handling.
- Do not add abstractions unless they remove real coupling. Current extension seams are `LLMClient`, `AgentRunner`, `llm_factory`, and `agent_factory`.
- Keep MCP tool cache provider-neutral. Provider-specific tool formatting belongs inside the selected `LLMClient` implementation.
- Keep OpenAI-specific code inside `openai_llm.py` and provider selection inside `llm_factory.py`.
- Do not implement raw MCP protocol handling. Use the official Python MCP SDK/client.
- Treat MCP tool results as untrusted data. Policy decisions must be made by service code, not by LLM output.
- Avoid broad `try/except` except at integration boundaries where errors are converted into domain errors or where worker containment is required.
- Use structured logging through `AuditLogger`; do not print directly.
- Log operational events at the correct level: `DEBUG` for noisy flow, `INFO` for normal lifecycle/audit events, `WARNING` for degraded/retryable issues, `ERROR` for failed requests or worker failures.
- Never log secrets. Preserve recursive redaction for keys containing `token`, `secret`, `password`, `key`, `auth`, `authorization`, or `bearer`.
- Use no Telegram parse mode unless escaping is implemented.
- Keep v1 single-replica. In-memory queue, dedupe, locks, and cache are intentional.

## Testing
- Run `python -m pytest -q` after code changes.
- Run `python -m compileall app tests` after refactors or import changes.
- Prefer focused tests around behavior and boundaries: config parsing, route auth, webhook secret validation, queue behavior, agent loop, MCP policy, provider conversion, and logging redaction.

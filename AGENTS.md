# AGENTS.md

## Project
RuTV admin Telegram bot: a single-replica FastAPI webhook service with an embedded agent, configurable LLM boundary, and MCP playlist-service tooling.

## Project Structure
- `app/api/server.py`: FastAPI app construction, lifespan, router registration, and the Uvicorn `app`.
- `app/api/routes.py`: HTTP routes, request dependencies, and admin auth.
- `app/core/`: configuration, structured logging, service state, readiness checks, startup/shutdown composition, and MCP tool reload logic.
- `app/telegram/`: aiogram bot wrapper, Telegram update routing, slash-command handling, text splitting, bounded webhook queue, update dedupe, and per-user/global concurrency.
- `app/agent/`: provider-neutral embedded agent orchestration and tool-call execution.
- `app/llm/`: provider-neutral LLM protocol/DTOs, provider selection, and OpenAI-specific implementation.
- `app/mcp/`: official MCP SDK client wrapper, MCP sessions, tool loading, tool cache, provider-neutral schema normalization, disabled-tool policy, and argument-size policy.
- `tests/`: focused unit/integration tests for config, routes, queue, agent, LLM, MCP policy/schema, Telegram helpers, and logging.

## Engineering Rules
- Keep the implementation KISS/YAGNI. Do not add databases, Redis, durable queues, multi-replica coordination, or a separate agent service unless explicitly requested.
- Keep modules focused. FastAPI route wiring belongs in `app/api/`.
- Avoid very long functions. Prefer extracting private helpers when a function mixes orchestration, I/O, formatting, and error handling.
- Do not add abstractions unless they remove real coupling. Current extension seams are `LLMClient`, `AgentRunner`, and `app.llm.factory`.
- Keep MCP tool cache provider-neutral. Provider-specific tool formatting belongs inside the selected `LLMClient` implementation.
- Keep OpenAI-specific code inside `app/llm/openai.py` and provider selection inside `app/llm/factory.py`.
- Do not implement raw MCP protocol handling. Use the official Python MCP SDK/client.
- Treat MCP tool results as untrusted data. Policy decisions must be made by service code, not by LLM output.
- Avoid broad `try/except` except at integration boundaries where errors are converted into domain errors or where worker containment is required.
- Use structured logging through `StructuredLogger`; do not print directly.
- Log operational events at the correct level: `DEBUG` for noisy flow, `INFO` for normal lifecycle events, `WARNING` for degraded/retryable issues, `ERROR` for failed requests or worker failures.
- Never log secrets. Preserve recursive redaction for keys containing `token`, `secret`, `password`, `key`, `auth`, `authorization`, or `bearer`.
- Use no Telegram parse mode unless escaping is implemented.
- Keep v1 single-replica. In-memory queue, dedupe, locks, and cache are intentional.

## Testing
- Run `python -m pytest -q` after code changes.
- Run `python -m compileall app tests` after refactors or import changes.
- Prefer focused tests around behavior and boundaries: config parsing, route auth, webhook secret validation, queue behavior, agent loop, MCP policy, provider conversion, and logging redaction.

# RuTV Admin Bot

FastAPI Telegram webhook service that runs a Pydantic AI agent against configured MCP servers.

## Run

Create an environment file from `.env.example`, then run:

```bash
docker compose up -d --build
```

The service exposes:

- `GET /health`
- `GET /ready`
- `POST /telegram/webhook`
- `GET /tools`
- `POST /tools/reload`
- `POST /telegram/webhook/register`

`/tools`, `/tools/reload`, and `/telegram/webhook/register` require:

```http
Authorization: Bearer <ADMIN_API_TOKEN>
```

## Development

Run checks before committing code changes:

```bash
python -m compileall app tests
python -m pytest -q
```

## Notes

- Telegram uses webhook mode only.
- MCP servers are defined in `config/mcp_config.toml`; MCP bearer tokens are supplied as `MCP_<SERVER_NAME>_AUTH_TOKEN` environment variables.
- MCP toolsets are built at startup and by explicit reload only. Disabled servers are not connected.
- Pydantic AI manages the OpenAI agent and FastMCP-backed MCP toolsets. Tool names are exposed as `<server>_<tool>`.
- Confirmation-required tools use Pydantic AI deferred approvals. The confirmation is bound to the requesting Telegram user, chat, arguments, and agent generation.
- V1 is single-replica and uses in-memory queue, dedupe, locks, and pending confirmations.

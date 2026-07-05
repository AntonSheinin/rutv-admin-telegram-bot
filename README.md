# RuTV Admin Bot

FastAPI Telegram webhook service that runs an embedded LLM-backed agent against cached MCP tools from the RuTV playlist service.

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
Authorization: Bearer <MCP_AUTH_TOKEN>
```

## Notes

- Telegram uses webhook mode only.
- MCP tools are fetched at startup and by explicit reload only.
- V1 is single-replica and uses in-memory queue, dedupe, locks, and tool cache.
- The service uses the official Python MCP SDK with the playlist-service streamable HTTP MCP endpoint.
- LLM integration goes through `app.llm.LLMClient`; OpenAI lives in `app.llm.openai` and is selected with `LLM_PROVIDER=openai`.

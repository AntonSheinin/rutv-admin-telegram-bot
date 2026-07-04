# RuTV Admin Bot

FastAPI Telegram webhook service that runs an embedded LLM-backed agent against cached MCP tools from the RuTV playlist service.

## Run

Create an environment file from `.env.example`, then run:

```bash
docker build -t rutv-admin-bot .
docker run --env-file .env -p 8000:8000 rutv-admin-bot
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

## Notes

- Telegram uses webhook mode only.
- MCP tools are fetched at startup and by explicit reload only.
- V1 is single-replica and uses in-memory queue, dedupe, locks, and tool cache.
- The service uses the official Python MCP SDK with an SSE endpoint. If the playlist-service endpoint is incompatible, update the design before adding any custom MCP protocol code.
- LLM integration goes through `app.llm.LLMClient`; OpenAI is the first implementation and is selected with `LLM_PROVIDER=openai`.

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.lifecycle import shutdown, startup
from app.core.state import ServiceState


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = ServiceState()
    app.state.service = service
    await startup(service)
    try:
        yield
    finally:
        await shutdown(service)


app = FastAPI(title="RuTV Admin Bot", lifespan=lifespan)
app.include_router(router)

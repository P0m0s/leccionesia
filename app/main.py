import asyncio
import contextlib
from collections.abc import AsyncIterator

import structlog
from fastapi import FastAPI

from app.config import settings
from app.routers import embeddings, estimations, sessions
from app.sessions import session_store
from vector_store import vector_store

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(colors=False),
    ],
)
logger = structlog.get_logger(__name__)


SESSION_GC_INTERVAL_SECONDS = 60


async def _session_gc_loop() -> None:
    while True:
        try:
            await asyncio.sleep(SESSION_GC_INTERVAL_SECONDS)
            removed = session_store.evict_inactive(
                ttl_seconds=settings.session_ttl_seconds,
            )
            if removed:
                logger.info(
                    "sessions_gc_evicted",
                    removed=removed,
                    remaining=len(session_store),
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("sessions_gc_error", error=str(exc))


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    if settings.vector_store_path:
        vector_store.load(settings.vector_store_path)
    task = asyncio.create_task(_session_gc_loop())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app = FastAPI(
    title="Estimador CAG",
    description=(
        "API de estimación de software con CAG versionado (Jinja2), "
        "wrapper multi-proveedor y caché exact-match."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(estimations.router)
app.include_router(sessions.router)
app.include_router(embeddings.router)


@app.get("/health")
def health() -> dict[str, str | int]:
    return {"status": "ok", "active_sessions": len(session_store)}

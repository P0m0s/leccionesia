import structlog
from fastapi import FastAPI

from app.routers import estimations

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(colors=False),
    ],
)

app = FastAPI(
    title="Estimador CAG",
    description=(
        "API de estimación de software con CAG versionado (Jinja2), "
        "wrapper multi-proveedor y caché exact-match."
    ),
    version="0.1.0",
)

app.include_router(estimations.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

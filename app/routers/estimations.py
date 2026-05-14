from typing import Literal

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from app.schemas import EstimationRequest, EstimationResponse
from app.services.estimate_service import iter_estimate_stream, run_estimate_sync

router = APIRouter(tags=["estimations"])

PromptVersionQuery = Literal["v1", "v2"]


@router.post("/estimate", response_model=EstimationResponse)
def estimate(
    body: EstimationRequest,
    prompt_version: PromptVersionQuery = Query("v1", description="Versión de plantillas Jinja"),
) -> EstimationResponse:
    return run_estimate_sync(body, prompt_version)


@router.post("/estimate/stream")
def estimate_stream(
    body: EstimationRequest,
    prompt_version: PromptVersionQuery = Query("v1", description="Versión de plantillas Jinja"),
) -> StreamingResponse:
    return StreamingResponse(
        iter_estimate_stream(body, prompt_version),
        media_type="text/plain; charset=utf-8",
    )

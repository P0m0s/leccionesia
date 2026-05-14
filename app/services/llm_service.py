import time
from collections.abc import Iterator
from typing import Any

from fastapi import HTTPException

from app.config import settings


def generate_estimation_messages(
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, str, str]:
    """
    Llama al LLM configurado con mensajes system + user y devuelve
    (texto, id_modelo, proveedor).
    """
    provider = settings.llm_provider

    if provider == "openai":
        if not settings.openai_api_key:
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY no configurada",
            )
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        completion = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
        )
        text = completion.choices[0].message.content or ""
        return text.strip(), settings.openai_model, "openai"

    if not settings.anthropic_api_key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY no configurada",
        )
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    message = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    parts: list[str] = []
    for block in message.content:
        if block.type == "text":
            parts.append(block.text)
    text = "\n".join(parts).strip()
    return text, settings.anthropic_model, "anthropic"


def stream_estimation_messages(
    system_prompt: str,
    user_prompt: str,
    *,
    metrics_out: dict[str, Any] | None = None,
) -> Iterator[str]:
    """
    Igual que generate_estimation_messages, pero en streaming.
    Al terminar, rellena metrics_out con model, provider, input_tokens, output_tokens, elapsed_ms.
    """
    provider = settings.llm_provider
    t0 = time.perf_counter()

    def _finish(
        model: str,
        prov: str,
        in_tok: int | None,
        out_tok: int | None,
    ) -> None:
        if metrics_out is not None:
            metrics_out["model"] = model
            metrics_out["provider"] = prov
            metrics_out["input_tokens"] = in_tok
            metrics_out["output_tokens"] = out_tok
            metrics_out["elapsed_ms"] = (time.perf_counter() - t0) * 1000

    if provider == "openai":
        if not settings.openai_api_key:
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY no configurada",
            )
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        stream = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            stream=True,
            stream_options={"include_usage": True},
        )
        in_tok: int | None = None
        out_tok: int | None = None
        try:
            for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content
                if chunk.usage is not None:
                    in_tok = chunk.usage.prompt_tokens
                    out_tok = chunk.usage.completion_tokens
        finally:
            _finish(settings.openai_model, "openai", in_tok, out_tok)
        return

    if not settings.anthropic_api_key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY no configurada",
        )
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    in_tok_a: int | None = None
    out_tok_a: int | None = None
    try:
        with client.messages.stream(
            model=settings.anthropic_model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text
            final = stream.get_final_message()
            if final.usage is not None:
                in_tok_a = final.usage.input_tokens
                out_tok_a = final.usage.output_tokens
    finally:
        _finish(settings.anthropic_model, "anthropic", in_tok_a, out_tok_a)

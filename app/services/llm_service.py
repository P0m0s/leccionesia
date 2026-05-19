import time
from collections.abc import Iterator
from copy import deepcopy
from typing import Any

from fastapi import HTTPException

from app.config import settings

ImageInput = tuple[str, str]
"""Imagen como ``(mime_type, base64_data)`` para inyectar en el último user message."""


def _split_system_and_chat(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Separa el primer mensaje ``system`` del resto (formato Anthropic)."""
    system_prompt = ""
    rest: list[dict[str, Any]] = []
    for msg in messages:
        if msg["role"] == "system" and not system_prompt:
            content = msg["content"]
            system_prompt = content if isinstance(content, str) else str(content)
        else:
            rest.append(msg)
    return system_prompt, rest


def _inject_images_openai(
    messages: list[dict[str, Any]],
    images: list[ImageInput],
) -> list[dict[str, Any]]:
    """Convierte el último user message en multimodal con ``image_url`` data URIs."""
    if not images:
        return messages
    out = deepcopy(messages)
    for i in range(len(out) - 1, -1, -1):
        if out[i]["role"] == "user":
            text = out[i]["content"] if isinstance(out[i]["content"], str) else ""
            content_parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
            for mime, b64 in images:
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                )
            out[i] = {"role": "user", "content": content_parts}
            break
    return out


def _inject_images_anthropic(
    messages: list[dict[str, Any]],
    images: list[ImageInput],
) -> list[dict[str, Any]]:
    """Convierte el último user message a content blocks compatibles con Anthropic."""
    if not images:
        return messages
    out = deepcopy(messages)
    for i in range(len(out) - 1, -1, -1):
        if out[i]["role"] == "user":
            text = out[i]["content"] if isinstance(out[i]["content"], str) else ""
            blocks: list[dict[str, Any]] = []
            for mime, b64 in images:
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime,
                            "data": b64,
                        },
                    },
                )
            blocks.append({"type": "text", "text": text})
            out[i] = {"role": "user", "content": blocks}
            break
    return out


def generate_chat_messages(
    messages: list[dict[str, Any]],
    *,
    images: list[ImageInput] | None = None,
    metrics_out: dict[str, Any] | None = None,
) -> tuple[str, str, str]:
    """
    Llama al LLM con una lista completa de mensajes (incluido el ``system`` inicial)
    y devuelve ``(texto, id_modelo, proveedor)``.

    Si se pasa ``images`` (lista de ``(mime_type, base64_data)``), se inyectan
    en el último user message como contenido multimodal en el formato propio de
    cada proveedor.

    Si se pasa ``metrics_out``, se rellena con ``model``, ``provider``,
    ``input_tokens``, ``output_tokens`` y ``elapsed_ms`` para que el llamante
    pueda agregar costes y métricas.
    """
    provider = settings.llm_provider
    imgs = images or []
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
        msgs = _inject_images_openai(messages, imgs)
        completion = client.chat.completions.create(
            model=settings.openai_model,
            messages=msgs,
            temperature=0.4,
        )
        text = completion.choices[0].message.content or ""
        in_tok = getattr(getattr(completion, "usage", None), "prompt_tokens", None)
        out_tok = getattr(getattr(completion, "usage", None), "completion_tokens", None)
        _finish(settings.openai_model, "openai", in_tok, out_tok)
        return text.strip(), settings.openai_model, "openai"

    if not settings.anthropic_api_key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY no configurada",
        )
    import anthropic

    msgs = _inject_images_anthropic(messages, imgs)
    system_prompt, chat_msgs = _split_system_and_chat(msgs)
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    message = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=4096,
        system=system_prompt,
        messages=chat_msgs,
    )
    parts: list[str] = []
    for block in message.content:
        if block.type == "text":
            parts.append(block.text)
    text = "\n".join(parts).strip()
    in_tok_a = getattr(getattr(message, "usage", None), "input_tokens", None)
    out_tok_a = getattr(getattr(message, "usage", None), "output_tokens", None)
    _finish(settings.anthropic_model, "anthropic", in_tok_a, out_tok_a)
    return text, settings.anthropic_model, "anthropic"


def generate_estimation_messages(
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, str, str]:
    """Atajo legacy: estimación one-shot con ``system`` + ``user``."""
    return generate_chat_messages(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )


def stream_chat_messages(
    messages: list[dict[str, Any]],
    *,
    metrics_out: dict[str, Any] | None = None,
    images: list[ImageInput] | None = None,
) -> Iterator[str]:
    """
    Igual que ``generate_chat_messages``, pero en streaming.

    Al terminar, rellena ``metrics_out`` con ``model``, ``provider``,
    ``input_tokens``, ``output_tokens`` y ``elapsed_ms``.
    """
    provider = settings.llm_provider
    imgs = images or []
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
        msgs = _inject_images_openai(messages, imgs)
        stream = client.chat.completions.create(
            model=settings.openai_model,
            messages=msgs,
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

    msgs = _inject_images_anthropic(messages, imgs)
    system_prompt, chat_msgs = _split_system_and_chat(msgs)
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    in_tok_a: int | None = None
    out_tok_a: int | None = None
    try:
        with client.messages.stream(
            model=settings.anthropic_model,
            max_tokens=4096,
            system=system_prompt,
            messages=chat_msgs,
        ) as stream:
            for text in stream.text_stream:
                yield text
            final = stream.get_final_message()
            if final.usage is not None:
                in_tok_a = final.usage.input_tokens
                out_tok_a = final.usage.output_tokens
    finally:
        _finish(settings.anthropic_model, "anthropic", in_tok_a, out_tok_a)


def stream_estimation_messages(
    system_prompt: str,
    user_prompt: str,
    *,
    metrics_out: dict[str, Any] | None = None,
) -> Iterator[str]:
    """Atajo legacy para el endpoint one-shot ``/estimate/stream``."""
    yield from stream_chat_messages(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        metrics_out=metrics_out,
    )

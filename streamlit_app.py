"""
Interfaz Streamlit: chat conversacional, formulario tipado y atajo transcripción.

Ejecutar desde la raíz del proyecto:

    .\\.venv\\Scripts\\python -m streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

import httpx
import pandas as pd
import streamlit as st

from app.frontend_utils import (
    conversation_to_markdown,
    conversation_to_pdf,
    metadata_card_specs,
    parse_markdown_tables,
)
from app.prompts.loader import (
    render_chat_system_prompt,
    render_estimation_prompt,
)
from app.schemas import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
)
from app.services.attachments import extract_attachment_text, is_image_attachment
from app.sessions import MAX_TURNS, ProjectMetadata


def _apply_streamlit_secrets() -> None:
    try:
        sec = st.secrets
    except (RuntimeError, FileNotFoundError, OSError):
        return
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "LLM_PROVIDER",
        "OPENAI_MODEL",
        "ANTHROPIC_MODEL",
        "ESTIMATOR_API_BASE",
    ):
        try:
            if key in sec:
                os.environ[key] = str(sec[key])
        except Exception:
            continue


_apply_streamlit_secrets()

_DEFAULT_SAMPLE = EstimationRequest(
    description="Portal interno de aprobación de gastos con flujo multi-nivel y exportación contable.",
    project_type=ProjectType.INTERNAL_TOOL,
    detail_level=DetailLevel.MEDIUM,
    output_format=OutputFormat.LINE_ITEMS,
)

st.set_page_config(
    page_title="Estimador CAG",
    page_icon="📋",
    layout="wide",
)

_DEFAULTS = {
    "response_history": [],
    "last_metrics": None,
    "chat_session_id": None,
    "chat_history": [],
    "chat_project_metadata": {},
    "chat_session_metrics": {},
    "chat_prompt_version": "v1",
    "chat_use_stream": True,
    "chat_use_refine": False,
    "chat_last_structured": None,
    "chat_last_attachments_processed": [],
    "chat_tier": "conversational",
    "chat_user_id": "demo-user",
}

TIER_OPTIONS: list[tuple[str, str]] = [
    ("conversational", "💬 Conversacional (sin tier)"),
    ("developer", "🛠️ Developer (técnico)"),
    ("pm", "🧭 PM (gestión)"),
    ("executive", "💼 Executive (1 página)"),
    ("research", "🔬 Research (informe profundo)"),
]
TIER_LABELS = dict(TIER_OPTIONS)
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

API_BASE = os.environ.get("ESTIMATOR_API_BASE", "http://127.0.0.1:8000").rstrip("/")


def _create_chat_session() -> str | None:
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(f"{API_BASE}/sessions")
            r.raise_for_status()
            sid = r.json().get("session_id")
    except (httpx.HTTPError, httpx.RequestError) as e:
        st.error(f"No se pudo crear la sesión: {e}")
        return None
    st.session_state["chat_session_id"] = sid
    st.session_state["chat_history"] = []
    st.session_state["chat_project_metadata"] = {}
    st.session_state["chat_session_metrics"] = {}
    st.session_state["chat_last_structured"] = None
    return sid


def _ensure_chat_session() -> str | None:
    """Devuelve un ``session_id`` válido; valida contra el backend y lo recrea si caducó."""
    sid = st.session_state.get("chat_session_id")
    if not sid:
        return _create_chat_session()
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(f"{API_BASE}/sessions/{sid}")
        if r.status_code == 404:
            st.info(
                "La sesión previa ya no existe en el servidor (probablemente "
                "se reinició). He creado una nueva."
            )
            return _create_chat_session()
        r.raise_for_status()
    except httpx.RequestError as e:
        st.error(f"No se pudo contactar la API: {e}")
        return None
    except httpx.HTTPStatusError:
        return _create_chat_session()
    return sid


def _reset_chat_session() -> None:
    st.session_state["chat_session_id"] = None
    st.session_state["chat_history"] = []
    st.session_state["chat_project_metadata"] = {}
    st.session_state["chat_session_metrics"] = {}
    st.session_state["chat_last_structured"] = None


def _render_metadata_cards(metadata: dict[str, Any]) -> None:
    cards = metadata_card_specs(metadata)
    if not cards:
        st.caption("Aún sin contexto enriquecido. Habla con el modelo o adjunta documentos.")
        return
    for card in cards:
        st.markdown(
            f"**{card['icon']} {card['label']}**  \n{card['value']}",
        )


def _render_window_indicator(history: list[dict[str, str]]) -> None:
    in_window = min(len(history), MAX_TURNS)
    total = len(history)
    pct = in_window / MAX_TURNS if MAX_TURNS else 0
    st.markdown("**🪟 Ventana deslizante**")
    st.progress(min(pct, 1.0))
    if total > MAX_TURNS:
        st.caption(
            f"En ventana: {in_window}/{MAX_TURNS} · Total turnos: {total // 2}. "
            "Los antiguos se han resumido automáticamente."
        )
    else:
        st.caption(f"En ventana: {in_window}/{MAX_TURNS}")


def _render_session_metrics(metrics: dict[str, Any]) -> None:
    if not metrics:
        st.caption("Métricas en cuanto envíes el primer turno.")
        return
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Turnos", metrics.get("turns_count", 0))
        st.metric("Tokens entrada", metrics.get("input_tokens_total", 0))
    with c2:
        st.metric("Llamadas LLM", metrics.get("llm_calls", 0))
        st.metric("Tokens salida", metrics.get("output_tokens_total", 0))
    cost = metrics.get("estimated_cost_usd") or 0
    st.caption(f"Coste estimado: **${cost:.4f}**")
    if metrics.get("last_model"):
        st.caption(f"Último modelo: `{metrics['last_model']}` ({metrics.get('last_provider')})")


def _clean_summary_markdown(raw: str) -> str:
    """Si el LLM metió JSON crudo dentro de ``summary_markdown``, extrae solo el texto."""
    if not raw:
        return ""
    text = raw.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            obj = json.loads(text)
            inner = obj.get("summary_markdown")
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
        except (json.JSONDecodeError, AttributeError):
            pass
    return text


_CONFIDENCE_BUCKETS = (
    (3, "🔴", "baja", "Estimación muy preliminar"),
    (6, "🟡", "media", "Estimación tentativa"),
    (10, "🟢", "alta", "Estimación con buena base"),
)


def _confidence_meta(conf: int | None) -> tuple[str, str, str] | None:
    if conf is None:
        return None
    for limit, icon, label, caption in _CONFIDENCE_BUCKETS:
        if conf <= limit:
            return icon, label, caption
    return "🟢", "alta", "Estimación con buena base"


_SEVERITY_ICONS = {"low": "🟢", "medium": "🟡", "high": "🔴"}


def _render_structured(structured: dict[str, Any] | None, summary_markdown: str) -> None:
    """Render rico de la estimación estructurada."""
    if not structured:
        st.markdown(_clean_summary_markdown(summary_markdown))
        return

    summary = _clean_summary_markdown(
        structured.get("summary_markdown") or summary_markdown,
    )
    tmin = structured.get("total_hours_min")
    tmax = structured.get("total_hours_max")
    line_items = structured.get("line_items") or []
    phases = structured.get("phases") or []
    assumptions = structured.get("assumptions") or []
    risks = structured.get("risks") or []
    conf = structured.get("confidence")
    next_step = structured.get("next_step")

    has_numbers = (tmin is not None and tmax is not None) or line_items or phases

    if has_numbers and tmin is not None and tmax is not None:
        m1, m2, m3 = st.columns(3)
        m1.metric("⏱️ Horas (mín)", f"{tmin:.0f} h")
        m2.metric("⏱️ Horas (máx)", f"{tmax:.0f} h")
        mid = (tmin + tmax) / 2.0
        m3.metric("📌 Punto medio", f"{mid:.0f} h")

    if summary:
        st.markdown(summary)

    embedded_tables = parse_markdown_tables(summary)
    for t in embedded_tables:
        try:
            df = pd.DataFrame(t[1:], columns=t[0])
            st.dataframe(df, use_container_width=True, hide_index=True)
        except Exception:
            continue

    if line_items:
        with st.expander(f"🧱 Line items ({len(line_items)})", expanded=True):
            df = pd.DataFrame(line_items)
            preferred = [c for c in ("name", "area", "t_shirt", "hours_min", "hours_max", "description") if c in df.columns]
            other = [c for c in df.columns if c not in preferred]
            df = df[preferred + other]
            st.dataframe(df, use_container_width=True, hide_index=True)

    if phases:
        with st.expander(f"🚦 Fases ({len(phases)})", expanded=True):
            df = pd.DataFrame(phases)
            st.dataframe(df, use_container_width=True, hide_index=True)

    if assumptions or risks:
        col_a, col_r = st.columns(2)
        with col_a:
            if assumptions:
                st.markdown(f"**📝 Supuestos** ({len(assumptions)})")
                for a in assumptions:
                    st.markdown(f"- {a.get('text', '')}")
        with col_r:
            if risks:
                st.markdown(f"**⚠️ Riesgos** ({len(risks)})")
                for r in risks:
                    sev = r.get("severity")
                    icon = _SEVERITY_ICONS.get(sev or "", "•")
                    sev_str = f" _({sev})_" if sev else ""
                    st.markdown(f"- {icon} {r.get('text', '')}{sev_str}")

    if next_step:
        st.info(f"👉 **Próximo paso:** {next_step}")

    meta = _confidence_meta(conf)
    if meta is not None and conf is not None:
        icon, label, caption = meta
        st.caption(f"{icon} Confianza **{label}** · {conf}/10 — {caption}")


def _attachment_preview(upload) -> tuple[str, str]:
    """Devuelve ``(tipo, preview)`` para mostrar al usuario antes de enviar."""
    if is_image_attachment(upload.name, upload.type):
        return "imagen", f"Imagen `{upload.name}` ({upload.size or 0} bytes)"
    data = upload.getvalue()
    res = extract_attachment_text(filename=upload.name, content_type=upload.type, data=data)
    if res.error:
        return "error", f"`{upload.name}`: {res.error}"
    text = res.text.strip()
    preview = text[:400] + ("…" if len(text) > 400 else "")
    return "texto", preview


def _apply_turn_to_history(
    *,
    user_label: str,
    assistant_text: str,
    structured: dict[str, Any] | None,
    command: str,
    tier: str | None = None,
) -> None:
    """Aplica el resultado de un turno al historial del frontend.

    Trata los slash commands especialmente para que el frontend refleje el
    estado real del backend tras ``/reset``, ``/clear`` o ``/regenerate``.
    """
    cmd = command.strip().lower().split(maxsplit=1)[0] if command else ""
    assistant_turn = {
        "role": "assistant",
        "content": assistant_text,
        "structured": structured,
        "tier": tier,
    }
    history = st.session_state["chat_history"]

    if cmd in {"/reset", "/clear"}:
        history.clear()
        st.session_state["chat_last_structured"] = None
        history.append({"role": "user", "content": user_label})
        history.append(assistant_turn)
        return

    if cmd == "/regenerate":
        while history and history[-1]["role"] == "assistant":
            history.pop()
        history.append(assistant_turn)
        return

    history.append({"role": "user", "content": user_label})
    history.append(assistant_turn)


_CONFIDENCE_LEVEL_ICON = {"low": "🔴", "medium": "🟡", "high": "🟢"}
_GO_NO_GO_BADGE = {
    "go": ("🟢", "GO"),
    "conditional_go": ("🟡", "GO con condiciones"),
    "no_go": ("🔴", "NO GO"),
}


def _render_tier_developer(s: dict[str, Any]) -> None:
    rng = s.get("total_hours_range") or {}
    if rng:
        c1, c2, c3 = st.columns(3)
        c1.metric("⏱️ Horas (mín)", f"{rng.get('min', 0):.0f} h")
        c2.metric("⏱️ Horas (máx)", f"{rng.get('max', 0):.0f} h")
        mid = ((rng.get("min", 0) or 0) + (rng.get("max", 0) or 0)) / 2
        c3.metric("📌 Punto medio", f"{mid:.0f} h")

    comps = s.get("components") or []
    if comps:
        with st.expander(f"🧱 Componentes técnicos ({len(comps)})", expanded=True):
            rows = []
            for c in comps:
                r = c.get("hours_range") or {}
                rows.append(
                    {
                        "name": c.get("name"),
                        "complexity": c.get("complexity"),
                        "hours_min": r.get("min"),
                        "hours_max": r.get("max"),
                        "description": c.get("description"),
                    },
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        risks = s.get("technical_risks") or []
        if risks:
            st.markdown(f"**⚠️ Riesgos técnicos** ({len(risks)})")
            for r in risks:
                mit = r.get("mitigation")
                mit_str = f" — _Mitigación: {mit}_" if mit else ""
                st.markdown(f"- {r.get('description', '')}{mit_str}")
    with col_b:
        unc = s.get("uncertainty_drivers") or []
        if unc:
            st.markdown(f"**🎲 Drivers de incertidumbre** ({len(unc)})")
            for u in unc:
                st.markdown(f"- {u}")

    stack = s.get("stack_assumptions") or []
    if stack:
        st.markdown(f"**🧰 Supuestos de stack** ({len(stack)})")
        st.markdown("\n".join(f"- {x}" for x in stack))


def _render_tier_pm(s: dict[str, Any]) -> None:
    rng = s.get("duration_weeks_range") or {}
    phases = s.get("phases") or []
    milestones = s.get("milestones") or []
    team = s.get("team_composition") or []
    if rng:
        c1, c2, c3 = st.columns(3)
        c1.metric("📆 Semanas (mín)", f"{rng.get('min', 0):.0f}")
        c2.metric("📆 Semanas (máx)", f"{rng.get('max', 0):.0f}")
        c3.metric("👥 Roles", len(team))

    if phases:
        with st.expander(f"🚦 Fases ({len(phases)})", expanded=True):
            rows = []
            for p in phases:
                d = p.get("duration_weeks") or {}
                rows.append(
                    {
                        "name": p.get("name"),
                        "weeks_min": d.get("min"),
                        "weeks_max": d.get("max"),
                        "deliverables": ", ".join(p.get("deliverables") or []),
                        "dependencies": ", ".join(p.get("dependencies") or []),
                    },
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if milestones:
        with st.expander(f"🏁 Hitos ({len(milestones)})", expanded=True):
            rows = [
                {
                    "name": m.get("name"),
                    "week": m.get("week"),
                    "deliverables": ", ".join(m.get("deliverables") or []),
                }
                for m in milestones
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if team:
        with st.expander(f"👥 Composición del equipo ({len(team)})"):
            st.dataframe(pd.DataFrame(team), use_container_width=True, hide_index=True)

    blockers = s.get("blockers") or []
    if blockers:
        st.markdown(f"**🚧 Blockers** ({len(blockers)})")
        for b in blockers:
            sev = b.get("impact")
            icon = _SEVERITY_ICONS.get(sev or "", "•")
            st.markdown(f"- {icon} {b.get('description', '')} _({sev})_")


def _render_tier_executive(s: dict[str, Any]) -> None:
    cost = s.get("headline_cost_range") or {}
    dur = s.get("headline_duration_range") or {}
    rec = s.get("go_no_go_recommendation")
    badge_icon, badge_label = _GO_NO_GO_BADGE.get(rec or "", ("•", rec or "—"))

    c1, c2, c3 = st.columns([2, 2, 3])
    c1.metric("💶 Coste", f"{cost.get('min', 0):,} – {cost.get('max', 0):,} €")
    c2.metric("📆 Duración", f"{dur.get('min', 0):.0f} – {dur.get('max', 0):.0f} sem.")
    c3.metric("📊 Recomendación", f"{badge_icon} {badge_label}")

    conf = s.get("confidence_level")
    if conf:
        icon = _CONFIDENCE_LEVEL_ICON.get(conf, "•")
        st.caption(f"{icon} Confianza **{conf}**")

    rationale = s.get("rationale")
    if rationale:
        st.markdown("**📝 Rationale**")
        st.info(rationale)

    risks = s.get("top_three_risks") or []
    if risks:
        st.markdown(f"**⚠️ Riesgos clave** ({len(risks)})")
        for r in risks:
            st.markdown(f"- **{r.get('headline', '')}** — {r.get('impact', '')}")


def _render_tier_research(s: dict[str, Any]) -> None:
    title = s.get("title")
    if title:
        st.markdown(f"### 📄 {title}")

    rng_h = s.get("total_hours_range") or {}
    rng_c = s.get("total_cost_range") or {}
    conf = s.get("confidence_level")
    c1, c2, c3 = st.columns(3)
    if rng_h:
        c1.metric("⏱️ Horas", f"{rng_h.get('min', 0):.0f} – {rng_h.get('max', 0):.0f}")
    if rng_c:
        c2.metric("💶 Coste", f"{rng_c.get('min', 0):,} – {rng_c.get('max', 0):,} €")
    if conf:
        icon = _CONFIDENCE_LEVEL_ICON.get(conf, "•")
        c3.metric("📊 Confianza", f"{icon} {conf}")

    summary = s.get("executive_summary")
    if summary:
        st.info(summary)

    toc = s.get("table_of_contents") or []
    if toc:
        with st.expander("📑 Índice", expanded=False):
            for i, t in enumerate(toc, 1):
                st.markdown(f"{i}. {t}")

    for section in s.get("sections") or []:
        with st.expander(f"**{section.get('title', 'Sección')}**", expanded=True):
            st.markdown(section.get("content_markdown") or "")
            citations = section.get("citations") or []
            if citations:
                st.caption("**Citas:**")
                for c in citations:
                    url = c.get("url")
                    src = c.get("source", "")
                    quote = c.get("quote")
                    line = f"- {src}" + (f" — _{quote}_" if quote else "")
                    if url:
                        line += f" · [{url}]({url})"
                    st.markdown(line)

    method = s.get("methodology")
    if method:
        st.markdown("**🧪 Metodología**")
        st.markdown(method)

    steps = s.get("next_steps") or []
    if steps:
        st.markdown("**👉 Siguientes pasos**")
        for s_ in steps:
            st.markdown(f"- {s_}")


_TIER_RENDERERS: dict[str, Any] = {
    "developer": _render_tier_developer,
    "pm": _render_tier_pm,
    "executive": _render_tier_executive,
    "research": _render_tier_research,
}


def _render_assistant_turn(turn: dict[str, Any]) -> None:
    """Despacha el render del turno del asistente según el tier."""
    tier = turn.get("tier")
    structured = turn.get("structured")
    content = turn.get("content", "")
    if tier in _TIER_RENDERERS and isinstance(structured, dict):
        st.markdown(_clean_summary_markdown(content))
        _TIER_RENDERERS[tier](structured)
        return
    if structured:
        _render_structured(structured, content)
        return
    st.markdown(_clean_summary_markdown(content))


def _http_error_message(err: httpx.HTTPStatusError) -> str:
    """Extrae el mensaje de error de una respuesta HTTP de forma segura.

    Funciona también con respuestas en streaming, donde ``response.text`` lanza
    ``ResponseNotRead`` si el cuerpo aún no se ha consumido.
    """
    resp = err.response
    try:
        resp.read()
    except Exception:
        pass
    try:
        payload = resp.json()
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if detail:
            return f"HTTP {resp.status_code}: {detail}"
    except Exception:
        pass
    try:
        body = resp.text
    except Exception:
        body = ""
    return f"HTTP {resp.status_code}: {body[:500]}"


with st.sidebar:
    st.header("Sesión conversacional")
    sid = st.session_state.get("chat_session_id")
    st.caption(f"session_id: `{sid or 'sin iniciar'}`")

    if st.button("🆕 Nueva conversación", key="sidebar_reset_chat", use_container_width=True):
        _reset_chat_session()
        st.rerun()

    st.subheader("📊 Métricas sesión")
    _render_session_metrics(st.session_state.get("chat_session_metrics") or {})

    st.subheader("🪟 Ventana deslizante")
    _render_window_indicator(st.session_state.get("chat_history") or [])

    st.subheader("🗂️ project_metadata")
    _render_metadata_cards(st.session_state.get("chat_project_metadata") or {})

    st.subheader("⬇️ Exportar")
    chat_md = conversation_to_markdown(
        sid,
        st.session_state.get("chat_history") or [],
        st.session_state.get("chat_project_metadata") or {},
        st.session_state.get("chat_session_metrics") or {},
    )
    st.download_button(
        "📄 Markdown",
        data=chat_md.encode("utf-8"),
        file_name=f"conversacion-{sid or 'sin-sesion'}.md",
        mime="text/markdown",
        use_container_width=True,
    )
    st.download_button(
        "🖨️ PDF",
        data=conversation_to_pdf(chat_md),
        file_name=f"conversacion-{sid or 'sin-sesion'}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    st.divider()
    st.header("🔍 System prompt efectivo")
    show_chat_prompt = st.checkbox("Mostrar prompt actual del chat", value=False)
    if show_chat_prompt:
        try:
            current_pm = ProjectMetadata.model_validate(
                st.session_state.get("chat_project_metadata") or {},
            )
        except Exception:
            current_pm = ProjectMetadata()
        try:
            chat_sys = render_chat_system_prompt(
                current_pm,
                version=st.session_state.get("chat_prompt_version", "v1"),
            )
        except ValueError as exc:
            chat_sys = f"(error: {exc})"
        st.text_area(
            "chat/{version}/system.j2",
            value=chat_sys,
            height=320,
            disabled=True,
        )

    st.divider()
    st.header("Contexto CAG (one-shot)")
    pv_preview = st.selectbox("Versión one-shot", ("v1", "v2"), index=0, key="sidebar_pv")
    try:
        sys_prev, user_prev = render_estimation_prompt(_DEFAULT_SAMPLE, version=pv_preview)
    except ValueError as e:
        st.error(str(e))
        sys_prev, user_prev = "", ""
    with st.expander("system.j2 renderizado", expanded=False):
        st.text_area("system", value=sys_prev, height=200, disabled=True, label_visibility="collapsed")
    with st.expander("user.j2 renderizado", expanded=False):
        st.text_area("user", value=user_prev, height=120, disabled=True, label_visibility="collapsed")


st.title("📋 Estimador de proyectos")
st.caption(
    "Conversación con memoria, adjuntos (PDF/DOCX/TXT/imágenes) y salida "
    f"estructurada. API base: `{API_BASE}`"
)

tab_chat, tab_form, tab_transcript = st.tabs(
    ["💬 Conversación (sesión)", "📝 Formulario de producto", "🎙️ Atajo transcripción"],
)

with tab_chat:
    tier_keys = [k for k, _ in TIER_OPTIONS]
    tier_labels_list = [lbl for _, lbl in TIER_OPTIONS]
    current_tier = st.session_state["chat_tier"]
    tier_index = tier_keys.index(current_tier) if current_tier in tier_keys else 0
    selected_tier_label = st.selectbox(
        "Perfil del cliente (tier)",
        options=tier_labels_list,
        index=tier_index,
        help=(
            "Cambia template, schema y pipeline según el perfil del receptor. "
            "En producción este tier vendría de tu sistema de auth, no de un dropdown."
        ),
    )
    st.session_state["chat_tier"] = tier_keys[tier_labels_list.index(selected_tier_label)]
    active_tier = st.session_state["chat_tier"]
    tier_is_active = active_tier != "conversational"

    cfg_cols = st.columns([2, 2, 2, 2])
    with cfg_cols[0]:
        st.session_state["chat_prompt_version"] = st.selectbox(
            "Prompt",
            options=("v1", "v2"),
            index=0 if st.session_state["chat_prompt_version"] == "v1" else 1,
            help="v1 estándar, v2 adversarial. Solo aplica al modo conversacional.",
            disabled=tier_is_active,
        )
    with cfg_cols[1]:
        st.session_state["chat_use_stream"] = st.checkbox(
            "Streaming",
            value=st.session_state["chat_use_stream"] and not tier_is_active,
            help="No disponible en modo tier (la salida es JSON estricto).",
            disabled=tier_is_active,
        )
    with cfg_cols[2]:
        st.session_state["chat_use_refine"] = st.checkbox(
            "Auto-crítica",
            value=st.session_state["chat_use_refine"],
            help="Doble pasada de revisión. Duplica latencia y tokens.",
            disabled=st.session_state["chat_use_stream"] or tier_is_active,
        )
    with cfg_cols[3]:
        st.caption("Comandos: `/help`, `/reset`, `/metadata`, `/regenerate`.")
    if tier_is_active:
        st.info(
            f"🎯 Modo tier activo: **{TIER_LABELS[active_tier]}**. "
            "Cada turno usa template + schema + pipeline específicos."
        )

    for turn in st.session_state["chat_history"]:
        with st.chat_message(turn["role"]):
            if turn["role"] == "assistant":
                _render_assistant_turn(turn)
            else:
                st.markdown(turn["content"])

    with st.form("chat_form", clear_on_submit=True):
        chat_text = st.text_area(
            "Mensaje al estimador",
            height=120,
            help="Describe el proyecto, continúa, o usa un comando `/`.",
        )
        chat_files = st.file_uploader(
            "Adjuntos (PDF, DOCX, TXT, imágenes)",
            type=["pdf", "docx", "txt", "md", "png", "jpg", "jpeg", "gif", "webp"],
            accept_multiple_files=True,
        )
        if chat_files:
            with st.expander(f"🔍 Preview de {len(chat_files)} adjunto(s)"):
                for f in chat_files:
                    kind, preview = _attachment_preview(f)
                    if kind == "imagen":
                        st.image(f, caption=preview, width=160)
                    elif kind == "error":
                        st.warning(preview)
                    else:
                        st.markdown(f"**`{f.name}`** — {len(preview)} chars (texto extraído):")
                        st.code(preview, language="markdown")
        chat_submitted = st.form_submit_button("Enviar turno", use_container_width=True)

    if chat_submitted:
        cleaned = (chat_text or "").strip()
        if not cleaned and not chat_files:
            st.warning("Escribe un mensaje o adjunta un documento.")
        else:
            current_sid = _ensure_chat_session()
            if current_sid:
                files_payload = [
                    (
                        "attachments",
                        (
                            upload.name,
                            upload.getvalue(),
                            upload.type or "application/octet-stream",
                        ),
                    )
                    for upload in chat_files or []
                ]
                data_payload = {"transcript": cleaned}
                params = {"prompt_version": st.session_state["chat_prompt_version"]}
                if st.session_state["chat_use_refine"] and not st.session_state["chat_use_stream"]:
                    params["refine"] = "true"

                tier_headers: dict[str, str] = {}
                if tier_is_active:
                    tier_headers["X-Estimator-Tier"] = active_tier
                    tier_headers["X-Estimator-User"] = st.session_state["chat_user_id"]

                use_stream = st.session_state["chat_use_stream"] and not tier_is_active
                if use_stream:
                    stream_url = f"{API_BASE}/sessions/{current_sid}/estimate/stream"
                    user_label = cleaned or "(adjuntos sin transcript)"

                    with st.chat_message("user"):
                        st.markdown(user_label)

                    buf = io.StringIO()
                    final_payload: dict[str, Any] | None = None
                    start_payload: dict[str, Any] | None = None
                    stream_error: str | None = None

                    with st.chat_message("assistant"):
                        placeholder = st.empty()
                        try:
                            with httpx.Client(timeout=180.0) as client:
                                with client.stream(
                                    "POST",
                                    stream_url,
                                    params=params,
                                    data=data_payload,
                                    files=files_payload or None,
                                ) as resp:
                                    resp.raise_for_status()
                                    for line in resp.iter_lines():
                                        if not line:
                                            continue
                                        try:
                                            event = json.loads(line)
                                        except json.JSONDecodeError:
                                            continue
                                        etype = event.get("type")
                                        if etype == "start":
                                            start_payload = event
                                        elif etype == "token":
                                            buf.write(event.get("delta", ""))
                                            placeholder.markdown(buf.getvalue() + " ▌")
                                        elif etype == "final":
                                            final_payload = event
                                            placeholder.markdown(event.get("text", buf.getvalue()))
                                        elif etype == "error":
                                            stream_error = event.get("error")
                        except httpx.HTTPStatusError as e:
                            stream_error = _http_error_message(e)
                        except httpx.RequestError as e:
                            stream_error = f"No se pudo contactar la API: {e}"

                    if start_payload and start_payload.get("attachments_failed"):
                        for f in start_payload["attachments_failed"]:
                            st.warning(
                                f"No se pudo procesar `{f.get('filename')}`: {f.get('error')}",
                            )

                    if stream_error:
                        st.error(stream_error)

                    if final_payload:
                        assistant_text = final_payload.get("text", buf.getvalue())
                        structured = final_payload.get("structured")
                        _apply_turn_to_history(
                            user_label=user_label,
                            assistant_text=assistant_text,
                            structured=structured,
                            command=cleaned,
                        )
                        st.session_state["chat_project_metadata"] = (
                            final_payload.get("project_metadata", {})
                        )
                        st.session_state["chat_session_metrics"] = (
                            final_payload.get("metrics", {})
                        )
                        st.session_state["chat_last_structured"] = structured
                        st.rerun()
                    elif not stream_error:
                        st.warning(
                            "El stream terminó sin un evento `final`. "
                            "El turno no se ha persistido en el historial."
                        )
                else:
                    try:
                        timeout_s = 600.0 if active_tier == "research" else 300.0
                        with httpx.Client(timeout=timeout_s) as client:
                            r = client.post(
                                f"{API_BASE}/sessions/{current_sid}/estimate",
                                params=params,
                                data=data_payload,
                                files=files_payload or None,
                                headers=tier_headers or None,
                            )
                            r.raise_for_status()
                            data = r.json()
                        user_label = cleaned or "(adjuntos sin transcript)"
                        _apply_turn_to_history(
                            user_label=user_label,
                            assistant_text=data.get("text", ""),
                            structured=data.get("structured"),
                            command=cleaned,
                            tier=data.get("tier"),
                        )
                        st.session_state["chat_project_metadata"] = data.get(
                            "project_metadata", {},
                        )
                        st.session_state["chat_session_metrics"] = data.get(
                            "metrics", {},
                        )
                        st.session_state["chat_last_structured"] = data.get("structured")
                        failed = data.get("attachments_failed") or []
                        for f in failed:
                            st.warning(
                                f"No se pudo procesar `{f.get('filename')}`: {f.get('error')}",
                            )
                        st.rerun()
                    except httpx.HTTPStatusError as e:
                        st.error(_http_error_message(e))
                    except httpx.RequestError as e:
                        st.error(f"No se pudo contactar la API: {e}")


with tab_form:
    use_stream = st.checkbox("Streaming de respuesta", value=True, key="form_stream")
    with st.form("estimation_form", clear_on_submit=False):
        description = st.text_area(
            "Descripción del proyecto",
            height=160,
            help="Entre 20 y 2000 caracteres.",
        )
        c1, c2 = st.columns(2)
        with c1:
            project_type = st.selectbox(
                "Tipo de proyecto",
                options=list(ProjectType),
                format_func=lambda x: x.value,
            )
            detail_level = st.selectbox(
                "Nivel de detalle",
                options=list(DetailLevel),
                format_func=lambda x: x.value,
            )
        with c2:
            output_format = st.selectbox(
                "Formato de salida",
                options=list(OutputFormat),
                format_func=lambda x: x.value,
            )
            prompt_version = st.selectbox("Versión de prompt", ("v1", "v2"), index=0)

        ref_raw = st.text_area(
            "Proyectos de referencia (opcional, uno por línea; v2 los usa en system)",
            height=72,
        )
        submitted = st.form_submit_button("Estimar")

    if submitted:
        refs = [line.strip() for line in ref_raw.splitlines() if line.strip()]
        try:
            req = EstimationRequest(
                description=description.strip(),
                project_type=project_type,
                detail_level=detail_level,
                output_format=output_format,
                reference_projects=refs or None,
            )
        except Exception as e:
            st.error(f"Datos no válidos: {e}")
        else:
            payload = req.model_dump(mode="json")
            params = {"prompt_version": prompt_version}
            try:
                if use_stream:

                    def token_stream():
                        with httpx.Client(timeout=120.0) as client:
                            with client.stream(
                                "POST",
                                f"{API_BASE}/estimate/stream",
                                params=params,
                                json=payload,
                            ) as resp:
                                resp.raise_for_status()
                                for chunk in resp.iter_text():
                                    if chunk:
                                        yield chunk

                    st.subheader("Resultado (streaming)")
                    full = st.write_stream(token_stream())
                    st.session_state["response_history"].append(
                        {
                            "request": payload,
                            "prompt_version": prompt_version,
                            "text": full if isinstance(full, str) else str(full or ""),
                            "streamed": True,
                        },
                    )
                else:
                    with httpx.Client(timeout=120.0) as client:
                        r = client.post(
                            f"{API_BASE}/estimate",
                            params=params,
                            json=payload,
                        )
                        r.raise_for_status()
                        data = r.json()
                    text = data.get("text", "")
                    st.subheader("Resultado")
                    st.markdown(text)
                    st.session_state["response_history"].append(
                        {
                            "request": payload,
                            "prompt_version": data.get("prompt_version", prompt_version),
                            "text": text,
                            "streamed": False,
                        },
                    )
            except httpx.HTTPStatusError as e:
                st.error(_http_error_message(e))
            except httpx.RequestError as e:
                st.error(f"No se pudo contactar la API: {e}")


with tab_transcript:
    st.caption(
        "Misma ruta de backend: convierte la transcripción en `description` "
        "(tipo web SaaS, detalle medio, narrativa)."
    )
    tr_stream = st.checkbox("Streaming", value=True, key="tr_stream")
    tr_text = st.text_area("Transcripción de reunión", height=200, key="tr_area")
    if st.button("Estimar desde transcripción", key="tr_btn"):
        desc = tr_text.strip()
        if len(desc) < 20:
            st.error("La transcripción debe tener al menos 20 caracteres.")
        else:
            req = EstimationRequest(
                description=desc,
                project_type=ProjectType.WEB_SAAS,
                detail_level=DetailLevel.MEDIUM,
                output_format=OutputFormat.NARRATIVE,
            )
            payload = req.model_dump(mode="json")
            params = {"prompt_version": "v1"}
            try:
                if tr_stream:

                    def tr_token_stream():
                        with httpx.Client(timeout=120.0) as client:
                            with client.stream(
                                "POST",
                                f"{API_BASE}/estimate/stream",
                                params=params,
                                json=payload,
                            ) as resp:
                                resp.raise_for_status()
                                for chunk in resp.iter_text():
                                    if chunk:
                                        yield chunk

                    st.markdown("**Resultado**")
                    full_tr = st.write_stream(tr_token_stream())
                    st.session_state["response_history"].append(
                        {
                            "request": payload,
                            "prompt_version": "v1",
                            "text": full_tr if isinstance(full_tr, str) else str(full_tr or ""),
                            "streamed": True,
                            "source": "transcription",
                        },
                    )
                else:
                    with httpx.Client(timeout=120.0) as client:
                        r = client.post(
                            f"{API_BASE}/estimate",
                            params=params,
                            json=payload,
                        )
                        r.raise_for_status()
                        data = r.json()
                    st.markdown(data.get("text", ""))
                    st.session_state["response_history"].append(
                        {
                            "request": payload,
                            "prompt_version": data.get("prompt_version", "v1"),
                            "text": data.get("text", ""),
                            "streamed": False,
                            "source": "transcription",
                        },
                    )
            except httpx.HTTPStatusError as e:
                st.error(_http_error_message(e))
            except httpx.RequestError as e:
                st.error(f"No se pudo contactar la API: {e}")

st.divider()
st.subheader("📜 Historial de estimaciones one-shot")
if not st.session_state["response_history"]:
    st.info("Aún no hay estimaciones one-shot en esta sesión.")
else:
    for i, item in enumerate(reversed(st.session_state["response_history"]), start=1):
        with st.expander(f"#{len(st.session_state['response_history']) - i + 1} — {item.get('prompt_version', '?')}"):
            st.caption("Petición JSON")
            st.code(json.dumps(item.get("request", {}), ensure_ascii=False, indent=2), language="json")
            st.markdown(item.get("text", ""))

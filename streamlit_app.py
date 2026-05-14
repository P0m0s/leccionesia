"""
Interfaz Streamlit: formulario tipado → POST /estimate, historial y streaming HTTP.
Ejecutar desde la raíz del proyecto:

    .\\.venv\\Scripts\\python -m streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

import httpx
import streamlit as st

from app.prompts.loader import render_estimation_prompt
from app.schemas import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
)


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

if "response_history" not in st.session_state:
    st.session_state["response_history"] = []
if "last_metrics" not in st.session_state:
    st.session_state["last_metrics"] = None

API_BASE = os.environ.get("ESTIMATOR_API_BASE", "http://127.0.0.1:8000").rstrip("/")

with st.sidebar:
    st.header("Contexto CAG (plantillas)")
    pv_preview = st.selectbox("Vista previa versión", ("v1", "v2"), index=0, key="sidebar_pv")
    try:
        sys_prev, user_prev = render_estimation_prompt(_DEFAULT_SAMPLE, version=pv_preview)
    except ValueError as e:
        st.error(str(e))
        sys_prev, user_prev = "", ""
    st.subheader("System (render)")
    st.text_area("system.j2 renderizado", value=sys_prev, height=260, disabled=True)
    st.subheader("User (render)")
    st.text_area("user.j2 renderizado", value=user_prev, height=120, disabled=True)

    st.subheader("Última llamada (métricas)")
    lm = st.session_state["last_metrics"]
    if lm:
        st.metric("Modelo", lm.get("model", "—"))
        st.caption(f"Proveedor: **{lm.get('provider', '—')}**")
        c1, c2 = st.columns(2)
        with c1:
            st.metric(
                "Tokens entrada",
                lm.get("input_tokens") if lm.get("input_tokens") is not None else "—",
            )
        with c2:
            st.metric(
                "Tokens salida",
                lm.get("output_tokens") if lm.get("output_tokens") is not None else "—",
            )
        elapsed = lm.get("elapsed_ms")
        if elapsed is not None:
            st.caption(f"Tiempo de respuesta: **{elapsed:.0f} ms**")
    else:
        st.caption(
            "Las métricas detalladas solo están disponibles en modo streaming local al LLM; "
            "vía API HTTP no se propagan en esta versión."
        )

st.title("Estimador de proyectos (CAG + producto)")
st.caption(
    "Formulario tipado contra `POST /estimate` y opcionalmente streaming por `POST /estimate/stream`. "
    f"API base: `{API_BASE}`"
)

tab_form, tab_transcript = st.tabs(["Formulario de producto", "Atajo transcripción"])

with tab_form:
    use_stream = st.checkbox("Streaming de respuesta", value=True)
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
                        }
                    )
                    st.session_state["last_metrics"] = None
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
                        }
                    )
                    st.session_state["last_metrics"] = None
            except httpx.HTTPStatusError as e:
                st.error(f"HTTP {e.response.status_code}: {e.response.text}")
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
                        }
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
                        }
                    )
            except httpx.HTTPStatusError as e:
                st.error(f"HTTP {e.response.status_code}: {e.response.text}")
            except httpx.RequestError as e:
                st.error(f"No se pudo contactar la API: {e}")

st.divider()
st.subheader("Historial de respuestas")
if not st.session_state["response_history"]:
    st.info("Aún no hay estimaciones en esta sesión.")
else:
    for i, item in enumerate(reversed(st.session_state["response_history"]), start=1):
        with st.expander(f"#{len(st.session_state['response_history']) - i + 1} — {item.get('prompt_version', '?')}"):
            st.caption("Petición JSON")
            st.code(json.dumps(item.get("request", {}), ensure_ascii=False, indent=2), language="json")
            st.markdown(item.get("text", ""))

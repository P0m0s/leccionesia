"""
Runner de stress test multi-turno para el CAG.

Uso::

    uv run python -m evals.stress.run \\
      --http http://localhost:8000 \\
      --scenarios growing,pivot,contradiction \\
      --attachment-sizes 0,5,20,50,100 \\
      --repeats 3 \\
      --output evals/stress/results.csv

Estrategia de captura: usa la respuesta HTTP de ``POST /sessions/{id}/estimate``
(campos ``metrics`` y ``project_metadata``) en lugar de parsear logs.
Justificación en REPORT.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time

import httpx

from evals.stress.metrics import CostBudgetMetric, LatencyBudgetMetric, MemoryDriftMetric
from evals.stress.scenarios import ALL_SCENARIOS, StressScenario

_DEFAULT_LATENCY_BUDGET_MS = 10_000.0
_DEFAULT_COST_BUDGET_USD = 0.10

_FIXTURES_DIR = pathlib.Path(__file__).resolve().parent / "fixtures"

CSV_COLUMNS = [
    "run_id",
    "scenario",
    "attachment_kb",
    "repeat",
    "turn_index",
    "session_id",
    "transcript_chars",
    "enriched_transcript_chars",
    "attachments_total_chars",
    "messages_in_window",
    "anchors_count",
    "summary_chars",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "latency_ms",
    "cache_hit_kind",
    "last_resolved_tier",
    "fact_to_remember",
    "memory_drift_score",
    "memory_drift_passed",
    "latency_budget_score",
    "cost_budget_score",
    "structured_ok",
]


def _create_session(base_url: str, client: httpx.Client) -> str:
    resp = client.post(f"{base_url}/sessions", timeout=30.0)
    resp.raise_for_status()
    return resp.json()["session_id"]


def _send_turn(
    base_url: str,
    client: httpx.Client,
    session_id: str,
    transcript: str,
    attachment_path: pathlib.Path | None = None,
) -> dict:
    url = f"{base_url}/sessions/{session_id}/estimate"
    files = None
    if attachment_path and attachment_path.exists():
        files = [("attachments", (attachment_path.name, attachment_path.read_bytes(), "application/pdf"))]

    t0 = time.perf_counter()
    resp = client.post(url, data={"transcript": transcript}, files=files, timeout=120.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    data = resp.json()
    data["_client_latency_ms"] = elapsed_ms
    return data


def _get_session_state(base_url: str, client: httpx.Client, session_id: str) -> dict:
    resp = client.get(f"{base_url}/sessions/{session_id}", timeout=30.0)
    resp.raise_for_status()
    return resp.json()


def _anchors_text(metadata: dict) -> str:
    parts = []
    for key in ("project_name", "project_type", "agreed_scope"):
        v = metadata.get(key)
        if v:
            parts.append(str(v))
    techs = metadata.get("mentioned_technologies") or []
    parts.extend(techs)
    return " ".join(parts)


def _count_anchors(metadata: dict) -> int:
    count = 0
    if metadata.get("project_name"):
        count += 1
    if metadata.get("project_type"):
        count += 1
    if metadata.get("assumed_team_size") is not None:
        count += 1
    if metadata.get("mentioned_technologies"):
        count += 1
    if metadata.get("agreed_scope"):
        count += 1
    if metadata.get("conversation_summary"):
        count += 1
    return count


def run_scenario(
    *,
    base_url: str,
    client: httpx.Client,
    scenario: StressScenario,
    attachment_kb: int,
    repeat: int,
    latency_budget_ms: float = _DEFAULT_LATENCY_BUDGET_MS,
    cost_budget_usd: float = _DEFAULT_COST_BUDGET_USD,
) -> list[dict]:
    """Ejecuta un escenario completo y devuelve una fila por turno."""
    session_id = _create_session(base_url, client)
    run_id = f"{scenario.scenario_name}_att{attachment_kb}_r{repeat}"

    attachment_path: pathlib.Path | None = None
    if attachment_kb > 0:
        attachment_path = _FIXTURES_DIR / f"synthetic_{attachment_kb}kb.pdf"
        if not attachment_path.exists():
            print(f"  WARN PDF {attachment_path.name} no encontrado. Ejecuta build_pdfs.py primero.", file=sys.stderr)
            attachment_path = None

    latency_metric = LatencyBudgetMetric(budget_ms=latency_budget_ms)
    cost_metric = CostBudgetMetric(budget_usd=cost_budget_usd)
    drift_metric = MemoryDriftMetric()

    rows: list[dict] = []
    cumulative_cost = 0.0

    for turn_index, transcript, fact in scenario.turns:
        try:
            resp = _send_turn(base_url, client, session_id, transcript, attachment_path)
        except Exception as exc:
            print(f"  FAIL {run_id} turno {turn_index}: {exc}", file=sys.stderr)
            continue

        metrics = resp.get("metrics", {})
        metadata = resp.get("project_metadata", {})
        client_latency = resp.get("_client_latency_ms", 0.0)

        last_metrics = metrics.get("last_metrics", {})
        turn_tokens_in = int(last_metrics.get("input_tokens") or 0)
        turn_tokens_out = int(last_metrics.get("output_tokens") or 0)
        turn_latency = float(last_metrics.get("elapsed_ms") or client_latency)
        turn_cost = float(metrics.get("estimated_cost_usd") or 0.0) - cumulative_cost
        if turn_cost < 0:
            turn_cost = 0.0
        cumulative_cost = float(metrics.get("estimated_cost_usd") or 0.0)

        summary = metadata.get("conversation_summary") or ""
        anchors = _anchors_text(metadata)
        metadata_raw = json.dumps(metadata, ensure_ascii=False)

        lat_result = latency_metric.evaluate(turn_latency)
        cost_result = cost_metric.evaluate(cumulative_cost)
        drift_result = drift_metric.evaluate(
            fact,
            summary=summary,
            anchors=anchors,
            metadata_raw=metadata_raw,
        )

        att_chars = 0
        if attachment_path and attachment_path.exists():
            att_chars = attachment_kb * 1024

        row = {
            "run_id": run_id,
            "scenario": scenario.scenario_name,
            "attachment_kb": attachment_kb,
            "repeat": repeat,
            "turn_index": turn_index,
            "session_id": session_id,
            "transcript_chars": len(transcript),
            "enriched_transcript_chars": len(transcript) + att_chars,
            "attachments_total_chars": att_chars,
            "messages_in_window": metrics.get("turns_count", turn_index) * 2,
            "anchors_count": _count_anchors(metadata),
            "summary_chars": len(summary),
            "tokens_in": turn_tokens_in,
            "tokens_out": turn_tokens_out,
            "cost_usd": round(turn_cost, 8),
            "latency_ms": round(turn_latency, 2),
            "cache_hit_kind": "none",
            "last_resolved_tier": None,
            "fact_to_remember": fact,
            "memory_drift_score": drift_result.score,
            "memory_drift_passed": drift_result.passed,
            "latency_budget_score": lat_result.score,
            "cost_budget_score": cost_result.score,
            "structured_ok": resp.get("structured_ok", False),
        }
        rows.append(row)
        status = "OK" if drift_result.passed else "FAIL"
        print(f"  {status} {run_id} t{turn_index:02d} lat={turn_latency:.0f}ms drift={drift_result.score}")

    return rows


def write_csv(rows: list[dict], output_path: str) -> None:
    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV escrito: {path} ({len(rows)} filas)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress test runner para el CAG.")
    parser.add_argument("--http", default="http://localhost:8000", help="Base URL de la API.")
    parser.add_argument("--scenarios", default="growing,pivot,contradiction", help="Escenarios separados por coma.")
    parser.add_argument("--attachment-sizes", default="0,5,20,50,100", help="Tamaños de adjunto en KB.")
    parser.add_argument("--repeats", type=int, default=3, help="Repeticiones por combinación.")
    parser.add_argument("--output", default="evals/stress/results.csv", help="Ruta de salida CSV.")
    parser.add_argument("--latency-budget", type=float, default=_DEFAULT_LATENCY_BUDGET_MS, help="Budget de latencia en ms.")
    parser.add_argument("--cost-budget", type=float, default=_DEFAULT_COST_BUDGET_USD, help="Budget de coste en USD.")
    args = parser.parse_args()

    scenario_names = [s.strip() for s in args.scenarios.split(",")]
    attachment_sizes = [int(s.strip()) for s in args.attachment_sizes.split(",")]

    all_rows: list[dict] = []
    client = httpx.Client(timeout=httpx.Timeout(120.0, connect=30.0))

    try:
        for name in scenario_names:
            scenario = ALL_SCENARIOS.get(name)
            if scenario is None:
                print(f"WARN Escenario '{name}' no encontrado. Disponibles: {list(ALL_SCENARIOS.keys())}", file=sys.stderr)
                continue

            for att_kb in attachment_sizes:
                for repeat in range(1, args.repeats + 1):
                    print(f"\n-- {name} | att={att_kb}KB | repeat={repeat}/{args.repeats} --")
                    rows = run_scenario(
                        base_url=args.http,
                        client=client,
                        scenario=scenario,
                        attachment_kb=att_kb,
                        repeat=repeat,
                        latency_budget_ms=args.latency_budget,
                        cost_budget_usd=args.cost_budget,
                    )
                    all_rows.extend(rows)
    finally:
        client.close()

    write_csv(all_rows, args.output)
    print(f"Total filas: {len(all_rows)}")


if __name__ == "__main__":
    main()

"""Trace export/import utilities for sharing routing data."""

from __future__ import annotations

import json
import orjson
from pathlib import Path

import httpx
import pandas as pd

from .schema import Trace


def _db_row_to_trace_payload(row: dict, activations: list[dict]) -> dict:
    """Convert a database row into a Trace-compatible API payload."""
    tokens_raw = row.get("tokens", "[]")
    token_strs = json.loads(tokens_raw) if isinstance(tokens_raw, str) else tokens_raw
    token_ids_raw = row.get("token_ids", "[]")
    token_ids = json.loads(token_ids_raw) if isinstance(token_ids_raw, str) else token_ids_raw

    model_id = row.get("model_id") or row.get("model_name") or "unknown"

    payload = {
        "model_id": model_id,
        "model_name": row.get("model_name") or model_id,
        "model_class": row.get("model_class", ""),
        "num_layers": row["num_layers"],
        "num_experts": row["num_experts"],
        "top_k": row["top_k"],
        "text": row.get("text", ""),
        "token_ids": token_ids,
        "token_strs": token_strs,
        "activations": activations,
        "timestamp": row.get("timestamp"),
    }
    return payload


def _import_payload_to_trace(data: dict) -> dict:
    """Normalize imported JSON into a Trace-compatible payload."""
    payload = dict(data)

    if "token_strs" not in payload and "tokens" in payload:
        tokens = payload["tokens"]
        payload["token_strs"] = json.loads(tokens) if isinstance(tokens, str) else tokens

    if "model_id" not in payload:
        payload["model_id"] = payload.get("model_name") or "unknown"

    if "token_ids" not in payload:
        payload["token_ids"] = []

    for key in ("trace_id", "id", "activation_id"):
        payload.pop(key, None)

    Trace.model_validate(payload)
    return payload


def export_traces(db_path: Path, output_dir: Path, fmt: str = "json") -> Path:
    """Export all traces from SQLite database to a file."""
    import sqlite3
    from datetime import datetime
    from itertools import groupby

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        trace_rows = conn.execute("SELECT * FROM traces ORDER BY trace_id").fetchall()
        # Stream activations already ordered by trace_id and group them in a
        # single pass (avoids a SQLite -> DataFrame -> dict round-trip). A tuple
        # cursor (row_factory=None) is faster than sqlite3.Row key/positional
        # access when materializing 100k+ activation dicts.
        activation_cursor = conn.cursor()
        activation_cursor.row_factory = None
        activation_cursor.execute(
            "SELECT trace_id, layer, token_idx, expert_idx, gate_weight "
            "FROM activations ORDER BY trace_id, layer, token_idx"
        )
        activations_by_trace: dict = {}
        for trace_id, group in groupby(activation_cursor, key=lambda r: r[0]):
            activations_by_trace[trace_id] = [
                {
                    "layer": row[1],
                    "token_idx": row[2],
                    "expert_idx": row[3],
                    "gate_weight": row[4],
                }
                for row in group
            ]
    finally:
        conn.close()

    traces = [
        _db_row_to_trace_payload(dict(row), activations_by_trace.get(row["trace_id"], []))
        for row in trace_rows
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if fmt == "json":
        output_path = output_dir / f"moe_atlas_export_{timestamp}.json"
        output_path.write_bytes(orjson.dumps({"traces": traces}, option=orjson.OPT_INDENT_2))
    elif fmt == "jsonl":
        output_path = output_dir / f"moe_atlas_export_{timestamp}.jsonl"
        with open(output_path, "wb") as handle:
            for trace in traces:
                # orjson (Rust) serializes ~8x faster than stdlib json and emits
                # compact bytes -> smaller, quicker-to-share records.
                handle.write(orjson.dumps(trace))
                handle.write(b"\n")
    elif fmt == "parquet":
        output_path = output_dir / f"moe_atlas_export_{timestamp}.parquet"
        flat = []
        for trace in traces:
            base = {key: value for key, value in trace.items() if key != "activations"}
            for activation in trace["activations"]:
                flat.append({**base, **activation})
        pd.DataFrame(flat).to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported export format: {fmt}")

    return output_path


def import_traces(input_path: str, backend_url: str) -> int:
    """Import traces from a file into a backend."""
    path = Path(input_path)

    if path.suffix == ".jsonl":
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        traces = [orjson.loads(line) for line in lines]
    else:
        data = orjson.loads(path.read_text(encoding="utf-8"))
        traces = data.get("traces", [data])

    count = 0
    with httpx.Client() as client:
        for trace in traces:
            try:
                payload = _import_payload_to_trace(trace)
                response = client.post(
                    f"{backend_url}/traces",
                    json=payload,
                    timeout=30.0,
                )
                response.raise_for_status()
                count += 1
            except Exception as exc:
                trace_ref = trace.get("trace_id") or trace.get("id") or "?"
                print(f"Failed to import trace {trace_ref}: {exc}")

    return count


def load_trace_from_file(path: str) -> dict:
    """Load a single trace from a JSON file."""
    return orjson.loads(Path(path).read_bytes())


def save_trace_to_file(trace: dict, path: str) -> None:
    """Save a trace to a JSON file."""
    Path(path).write_bytes(orjson.dumps(trace, option=orjson.OPT_INDENT_2))
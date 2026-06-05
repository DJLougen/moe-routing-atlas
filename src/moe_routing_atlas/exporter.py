"""Trace export/import utilities for sharing routing data."""

import json
from pathlib import Path
from typing import List

import httpx
import pandas as pd

from .config import get_config


def export_traces(db_path: Path, output_dir: Path, fmt: str = "json") -> Path:
    """Export all traces from SQLite database to a file.

    Args:
        db_path: Path to SQLite database
        output_dir: Directory to write export to
        fmt: Export format (json, jsonl, parquet)

    Returns:
        Path to the exported file
    """
    import sqlite3

    conn = sqlite3.connect(db_path)

    # Get all traces with activations
    traces_df = pd.read_sql_query("SELECT * FROM traces ORDER BY trace_id", conn)
    activations_df = pd.read_sql_query(
        "SELECT * FROM activations ORDER BY trace_id, layer, token_idx",
        conn,
    )

    conn.close()

    # Build full trace objects
    traces = []
    for _, row in traces_df.iterrows():
        trace = dict(row)
        trace["tokens"] = json.loads(trace["tokens"])
        trace["activations"] = (
            activations_df[activations_df["trace_id"] == trace["trace_id"]]
            .drop(columns=["trace_id"])
            .to_dict("records")
        )
        traces.append(trace)

    # Write in chosen format
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    if fmt == "json":
        output_path = output_dir / f"moe_atlas_export_{timestamp}.json"
        output_path.write_text(
            json.dumps({"traces": traces}, indent=2),
            encoding="utf-8",
        )
    elif fmt == "jsonl":
        output_path = output_dir / f"moe_atlas_export_{timestamp}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for trace in traces:
                f.write(json.dumps(trace) + "\n")
    elif fmt == "parquet":
        output_path = output_dir / f"moe_atlas_export_{timestamp}.parquet"
        # Flatten for parquet
        flat = []
        for trace in traces:
            base = {k: v for k, v in trace.items() if k != "activations"}
            for act in trace["activations"]:
                flat.append({**base, **act})
        pd.DataFrame(flat).to_parquet(output_path, index=False)

    return output_path


def import_traces(input_path: str, backend_url: str) -> int:
    """Import traces from a file into a backend.

    Args:
        input_path: Path to JSON/JSONL file
        backend_url: Backend API URL

    Returns:
        Number of traces imported
    """
    path = Path(input_path)

    if path.suffix == ".jsonl":
        traces = [json.loads(line) for line in path.read_text(encoding="utf-8").strip().split("\n")]
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        traces = data.get("traces", [data])

    count = 0
    with httpx.Client() as client:
        for trace in traces:
            try:
                response = client.post(
                    f"{backend_url}/traces",
                    json=trace,
                    timeout=30.0,
                )
                response.raise_for_status()
                count += 1
            except Exception as e:
                print(f"Failed to import trace {trace.get('trace_id', '?')}: {e}")

    return count


def load_trace_from_file(path: str) -> dict:
    """Load a single trace from a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_trace_to_file(trace: dict, path: str) -> None:
    """Save a trace to a JSON file."""
    Path(path).write_text(json.dumps(trace, indent=2), encoding="utf-8")

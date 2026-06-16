"""Trace export/import utilities for sharing routing data."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import httpx
import orjson
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


def collect_traces(db_path: Path) -> list[dict]:
    """Read every trace from the SQLite database as a shareable JSON payload."""
    import sqlite3
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

    return [
        _db_row_to_trace_payload(dict(row), activations_by_trace.get(row["trace_id"], []))
        for row in trace_rows
    ]


def export_traces(db_path: Path, output_dir: Path, fmt: str = "json") -> Path:
    """Export all traces from SQLite database to a file."""
    from datetime import datetime

    traces = collect_traces(db_path)

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


def trace_fingerprint(trace: dict) -> str:
    """Stable content hash identifying a trace by its routing data.

    The fingerprint depends only on the model identity, input tokens, and the
    set of expert activations (layer, token, expert, gate weight) -- not on
    activation ordering, display name, timestamp, or backend-assigned id. Two
    contributors who trace the same input through the same model therefore get
    the same fingerprint, so identical traces deduplicate cleanly when merged
    into a shared community file.
    """
    activations = sorted(
        (
            int(a["layer"]),
            int(a["token_idx"]),
            int(a["expert_idx"]),
            float(a["gate_weight"]),
        )
        for a in trace.get("activations", [])
    )
    canonical = {
        "model_id": trace.get("model_id") or trace.get("model_name") or "unknown",
        "top_k": trace.get("top_k"),
        "num_layers": trace.get("num_layers"),
        "num_experts": trace.get("num_experts"),
        "token_ids": list(trace.get("token_ids", [])),
        "activations": activations,
    }
    return hashlib.sha256(orjson.dumps(canonical, option=orjson.OPT_SORT_KEYS)).hexdigest()


@dataclass(slots=True)
class MergeResult:
    """Outcome of merging trace files into one shared file."""

    output: Path
    inputs: int
    written: int
    duplicates: int
    invalid: int


@dataclass(slots=True)
class AppendResult:
    """Outcome of appending traces to a shared file."""

    path: Path
    added: int
    duplicates: int
    invalid: int


@dataclass(slots=True)
class ValidationResult:
    """Outcome of validating a trace file before sharing."""

    path: Path
    valid: int
    duplicates: int
    invalid: int
    errors: list[str]


def _read_trace_records(path: Path) -> tuple[list[dict], int]:
    """Read raw trace dicts from a ``.json`` or ``.jsonl`` file.

    Returns ``(records, malformed)`` where ``malformed`` counts JSONL lines that
    are not valid JSON (or ``1`` for a corrupt ``.json`` file). Content errors
    are never raised, so one bad line cannot abort a community merge.
    """
    path = Path(path)
    malformed = 0
    records: list[dict] = []
    if path.suffix == ".jsonl":
        with open(path, "rb") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                try:
                    records.append(orjson.loads(line))
                except orjson.JSONDecodeError:
                    malformed += 1
    else:
        try:
            data = orjson.loads(path.read_bytes())
        except orjson.JSONDecodeError:
            return [], 1
        if isinstance(data, dict) and "traces" in data:
            records = list(data["traces"])
        elif isinstance(data, list):
            records = data
        else:
            records = [data]
    return records, malformed


def append_traces(traces: Iterable[dict], path: str | Path, dedup: bool = True) -> AppendResult:
    """Append traces to a JSONL file, creating it (and parents) if needed.

    Each trace is normalized and validated against the :class:`Trace` schema;
    invalid traces are skipped and counted rather than written. With ``dedup``
    (default), a trace whose routing data already appears in the file is skipped,
    so re-running or re-contributing the same trace does not bloat the file.
    """
    path = Path(path)
    seen: set[str] = set()
    if dedup and path.exists():
        existing, _ = _read_trace_records(path)
        for record in existing:
            try:
                seen.add(trace_fingerprint(_import_payload_to_trace(record)))
            except Exception:
                continue

    added = duplicates = invalid = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab") as handle:
        for trace in traces:
            try:
                payload = _import_payload_to_trace(trace)
            except Exception:
                invalid += 1
                continue
            if dedup:
                fingerprint = trace_fingerprint(payload)
                if fingerprint in seen:
                    duplicates += 1
                    continue
                seen.add(fingerprint)
            handle.write(orjson.dumps(payload))
            handle.write(b"\n")
            added += 1
    return AppendResult(path=path, added=added, duplicates=duplicates, invalid=invalid)


def merge_trace_files(
    inputs: Iterable[str | Path], output: str | Path, dedup: bool = True
) -> MergeResult:
    """Merge several trace files into a single JSONL file.

    Reads each input (``.json`` or ``.jsonl``), normalizes and validates every
    record, and writes the result as JSONL. Invalid records are skipped and
    counted so a single bad contribution cannot corrupt the shared file. With
    ``dedup`` (default), traces with identical routing data are written once.

    The output is written to a temporary file and atomically moved into place,
    so ``output`` may also be one of the inputs -- letting you grow a canonical
    community file in place (``merge community.jsonl new.jsonl -o community.jsonl``).
    """
    input_paths = [Path(p) for p in inputs]
    output = Path(output)

    seen: set[str] = set()
    written = duplicates = invalid = 0

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as handle:
            for source in input_paths:
                records, malformed = _read_trace_records(source)
                invalid += malformed
                for record in records:
                    try:
                        payload = _import_payload_to_trace(record)
                    except Exception:
                        invalid += 1
                        continue
                    if dedup:
                        fingerprint = trace_fingerprint(payload)
                        if fingerprint in seen:
                            duplicates += 1
                            continue
                        seen.add(fingerprint)
                    handle.write(orjson.dumps(payload))
                    handle.write(b"\n")
                    written += 1
        os.replace(tmp, output)
    finally:
        if tmp.exists():
            tmp.unlink()
    return MergeResult(
        output=output,
        inputs=len(input_paths),
        written=written,
        duplicates=duplicates,
        invalid=invalid,
    )


def validate_trace_file(path: str | Path) -> ValidationResult:
    """Validate a trace file for sharing without modifying it.

    Reports how many records are valid (unique), duplicated by routing data, and
    invalid (malformed JSON or failing the :class:`Trace` schema), with a few
    sample error messages. Useful before opening a community pull request.
    """
    path = Path(path)
    records, malformed = _read_trace_records(path)
    valid = duplicates = 0
    invalid = malformed
    errors: list[str] = []
    if malformed:
        errors.append(f"{malformed} line(s) were not valid JSON")
    seen: set[str] = set()
    for index, record in enumerate(records):
        try:
            payload = _import_payload_to_trace(record)
        except Exception as exc:
            invalid += 1
            if len(errors) < 10:
                errors.append(f"record {index}: {exc}")
            continue
        fingerprint = trace_fingerprint(payload)
        if fingerprint in seen:
            duplicates += 1
        else:
            seen.add(fingerprint)
            valid += 1
    return ValidationResult(
        path=path, valid=valid, duplicates=duplicates, invalid=invalid, errors=errors
    )

"""Tests for trace export/import payload normalization."""

from __future__ import annotations

import json
from pathlib import Path

import orjson

from moe_routing_atlas.exporter import (
    _db_row_to_trace_payload,
    _import_payload_to_trace,
    append_traces,
    merge_trace_files,
    trace_fingerprint,
    validate_trace_file,
)


def test_db_row_to_trace_payload_maps_model_id():
    payload = _db_row_to_trace_payload(
        {
            "text": "Hello",
            "tokens": json.dumps(["Hello"]),
            "token_ids": json.dumps([1]),
            "num_layers": 2,
            "num_experts": 8,
            "top_k": 2,
            "model_name": "test-model",
        },
        [{"layer": 0, "token_idx": 0, "expert_idx": 1, "gate_weight": 0.5}],
    )
    assert payload["model_id"] == "test-model"
    assert payload["token_strs"] == ["Hello"]
    assert payload["token_ids"] == [1]


def test_import_payload_normalizes_legacy_fields():
    payload = _import_payload_to_trace(
        {
            "tokens": ["Hi"],
            "model_name": "legacy-model",
            "num_layers": 1,
            "num_experts": 4,
            "top_k": 1,
            "activations": [],
        }
    )
    assert payload["model_id"] == "legacy-model"
    assert payload["token_strs"] == ["Hi"]


def _trace(token_ids=(1, 2), acts=((0, 0, 1, 0.5), (1, 0, 3, 0.25)), model_id="m", text="hi"):
    """Build a minimal valid trace payload for sharing tests."""
    return {
        "model_id": model_id,
        "num_layers": 2,
        "num_experts": 8,
        "top_k": 2,
        "text": text,
        "token_ids": list(token_ids),
        "token_strs": [str(t) for t in token_ids],
        "activations": [
            {"layer": layer, "token_idx": tok, "expert_idx": exp, "gate_weight": gate}
            for (layer, tok, exp, gate) in acts
        ],
    }


def _read_jsonl(path):
    return [orjson.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def test_fingerprint_ignores_order_and_volatile_metadata():
    base = _trace()
    shuffled = _trace()
    shuffled["activations"] = list(reversed(shuffled["activations"]))
    shuffled["model_name"] = "Pretty Display Name"
    shuffled["trace_id"] = 999
    shuffled["timestamp"] = "2099-01-01T00:00:00"
    assert trace_fingerprint(base) == trace_fingerprint(shuffled)


def test_fingerprint_changes_with_routing_data():
    base = _trace()
    other_expert = _trace(acts=((0, 0, 2, 0.5), (1, 0, 3, 0.25)))
    other_weight = _trace(acts=((0, 0, 1, 0.9), (1, 0, 3, 0.25)))
    assert trace_fingerprint(base) != trace_fingerprint(other_expert)
    assert trace_fingerprint(base) != trace_fingerprint(other_weight)


def test_append_traces_creates_file_and_dedups(tmp_path):
    target = tmp_path / "community" / "traces.jsonl"
    t1, t2, t3 = _trace(token_ids=(1,)), _trace(token_ids=(2,)), _trace(token_ids=(3,))

    first = append_traces([t1, t2], target)
    assert first.added == 2
    assert first.duplicates == 0
    assert target.exists()

    second = append_traces([t2, t3], target)
    assert second.added == 1
    assert second.duplicates == 1
    assert len(_read_jsonl(target)) == 3


def test_append_traces_skips_invalid(tmp_path):
    target = tmp_path / "traces.jsonl"
    good = _trace()
    bad = {"model_id": "m"}  # missing required num_layers/num_experts/top_k
    result = append_traces([good, bad], target)
    assert result.added == 1
    assert result.invalid == 1
    assert len(_read_jsonl(target)) == 1


def test_merge_dedups_across_files_and_in_place(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    append_traces([_trace(token_ids=(1,)), _trace(token_ids=(2,))], a)
    append_traces([_trace(token_ids=(2,)), _trace(token_ids=(3,))], b)

    out = tmp_path / "community.jsonl"
    result = merge_trace_files([a, b], out)
    assert result.written == 3
    assert result.duplicates == 1
    assert len(_read_jsonl(out)) == 3

    # output may also be an input: grow the canonical file in place
    grow = merge_trace_files([out, b], out)
    assert grow.written == 3
    assert len(_read_jsonl(out)) == 3


def test_merge_skips_malformed_jsonl(tmp_path):
    src = tmp_path / "mixed.jsonl"
    good = orjson.dumps(_import_payload_to_trace(_trace()))
    src.write_bytes(good + b"\n" + b"{not json}\n" + good + b"\n")
    out = tmp_path / "out.jsonl"
    result = merge_trace_files([src], out)
    assert result.invalid == 1
    assert result.written == 1
    assert result.duplicates == 1


def test_validate_trace_file_reports_counts(tmp_path):
    src = tmp_path / "check.jsonl"
    good = orjson.dumps(_import_payload_to_trace(_trace(token_ids=(1,))))
    other = orjson.dumps(_import_payload_to_trace(_trace(token_ids=(9,))))
    src.write_bytes(good + b"\n" + good + b"\n" + other + b"\n" + b"oops\n")
    result = validate_trace_file(src)
    assert result.valid == 2
    assert result.duplicates == 1
    assert result.invalid == 1
    assert result.errors


def test_cli_merge_and_validate(tmp_path):
    from click.testing import CliRunner

    from moe_routing_atlas.cli import cli

    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    append_traces([_trace(token_ids=(1,))], a)
    append_traces([_trace(token_ids=(2,))], b)
    out = tmp_path / "community.jsonl"

    runner = CliRunner()
    merged = runner.invoke(cli, ["merge", str(a), str(b), "-o", str(out)])
    assert merged.exit_code == 0, merged.output
    assert len(_read_jsonl(out)) == 2

    ok = runner.invoke(cli, ["validate", str(out)])
    assert ok.exit_code == 0, ok.output

    bad = tmp_path / "bad.jsonl"
    bad.write_bytes(b"{nope}\n")
    failed = runner.invoke(cli, ["validate", str(bad)])
    assert failed.exit_code == 1

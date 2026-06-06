"""Tests for trace export/import payload normalization."""

from __future__ import annotations

import json

from moe_routing_atlas.exporter import _db_row_to_trace_payload, _import_payload_to_trace


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
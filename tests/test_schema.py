"""Tests for core schema validation."""

import pytest

from moe_routing_atlas.schema import Trace, TraceActivation


def test_activation_creation():
    act = TraceActivation(layer=0, token_idx=0, expert_idx=5, gate_weight=0.42)
    assert act.layer == 0
    assert act.gate_weight == 0.42


def test_trace_creation():
    trace = Trace(
        model_id="test/model",
        text="Hello world",
        token_ids=[1, 2],
        token_strs=["Hello", " world"],
        num_layers=24,
        num_experts=64,
        top_k=4,
        activations=[
            TraceActivation(layer=0, token_idx=0, expert_idx=5, gate_weight=0.42),
        ],
    )
    assert trace.num_tokens == 2
    assert len(trace.activations) == 1


def test_trace_serialization():
    trace = Trace(
        model_id="test/model",
        text="Test",
        token_ids=[100],
        token_strs=["Test"],
        num_layers=4,
        num_experts=8,
        top_k=2,
        activations=[],
    )
    json_str = trace.model_dump_json()
    assert "Test" in json_str
    assert '"num_layers":4' in json_str

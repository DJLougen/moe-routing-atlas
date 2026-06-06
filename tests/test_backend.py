"""Integration tests for the FastAPI backend."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from moe_routing_atlas.backend import create_app
from moe_routing_atlas.schema import Trace, TraceActivation


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture
def client(test_db: Path) -> TestClient:
    app = create_app(db_path=str(test_db))
    return TestClient(app)


def _sample_trace() -> dict:
    return Trace(
        model_id="test/model",
        model_class="TestMoE",
        text="Hello world",
        token_ids=[1, 2, 3],
        token_strs=["Hello", " world", "!"],
        num_layers=2,
        num_experts=8,
        top_k=2,
        activations=[
            TraceActivation(layer=0, token_idx=0, expert_idx=1, gate_weight=0.5),
            TraceActivation(layer=1, token_idx=1, expert_idx=3, gate_weight=0.7),
        ],
        model_name="model",
    ).model_dump()


def test_create_and_get_trace_round_trip(client: TestClient):
    payload = _sample_trace()
    create_response = client.post("/traces", json=payload)
    assert create_response.status_code == 200
    trace_id = create_response.json()["trace_id"]

    get_response = client.get(f"/trace/{trace_id}")
    assert get_response.status_code == 200
    data = get_response.json()

    assert data["id"] == trace_id
    assert data["trace_id"] == trace_id
    assert data["token_strs"] == payload["token_strs"]
    assert data["token_ids"] == payload["token_ids"]
    assert len(data["activations"]) == 2


def test_missing_trace_returns_404(client: TestClient):
    response = client.get("/trace/9999")
    assert response.status_code == 404


def test_list_traces_returns_normalized_ids(client: TestClient):
    client.post("/traces", json=_sample_trace())
    response = client.get("/traces?limit=10")
    assert response.status_code == 200
    traces = response.json()
    assert len(traces) == 1
    assert traces[0]["id"] == traces[0]["trace_id"]
    assert "text" not in traces[0]
    assert "token_strs" not in traces[0]
    assert traces[0]["text_preview"] == "Hello world"


def test_list_limit_is_clamped(client: TestClient):
    response = client.get("/traces?limit=99999")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_oversized_activation_list_rejected(client: TestClient):
    payload = _sample_trace()
    payload["activations"] = [
        {
            "layer": 0,
            "token_idx": 0,
            "expert_idx": 0,
            "gate_weight": 0.1,
        }
    ] * 500_001
    response = client.post("/traces", json=payload)
    assert response.status_code == 422


def test_token_ids_persisted_in_database(client: TestClient, test_db: Path):
    payload = _sample_trace()
    client.post("/traces", json=payload)

    conn = sqlite3.connect(test_db)
    row = conn.execute("SELECT token_ids FROM traces").fetchone()
    conn.close()

    assert json.loads(row[0]) == payload["token_ids"]
"""Backend server factory."""

from __future__ import annotations

import json
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .config import get_config
from .schema import Trace

MAX_REQUEST_BYTES = 50 * 1024 * 1024


def create_app(db_path: str | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    config = get_config()
    db_path = db_path or str(config.db_path)
    _init_db(db_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _init_db(db_path)
        yield

    app = FastAPI(
        title="MoE Routing Atlas",
        description="API for Mixture-of-Experts routing visualization",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origin_list(),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestSizeLimitMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)

    _setup_routes(app, db_path)

    viz_dir = Path(__file__).parent / "visualizer"
    if viz_dir.exists():
        app.mount("/visualizer", StaticFiles(directory=str(viz_dir), html=True), name="viz")

    return app


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject oversized request bodies."""

    async def dispatch(self, request: Request, call_next) -> Response:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BYTES:
            raise HTTPException(status_code=413, detail="Request body too large")
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers for API and visualizer responses."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        if request.url.path.startswith("/visualizer"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline'; "
                "connect-src 'self'; "
                "img-src 'self' data:;"
            )
        return response


def _connect_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _init_db(db_path: str) -> None:
    """Initialize SQLite database with tables."""
    conn = _connect_db(db_path)
    cursor = conn.cursor()

    cursor.execute("PRAGMA journal_mode = WAL")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS traces (
            trace_id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            tokens TEXT NOT NULL,
            token_ids TEXT NOT NULL DEFAULT '[]',
            num_tokens INTEGER NOT NULL,
            num_layers INTEGER NOT NULL,
            num_experts INTEGER NOT NULL,
            top_k INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            model_name TEXT,
            model_id TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS activations (
            activation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id INTEGER NOT NULL,
            layer INTEGER NOT NULL,
            token_idx INTEGER NOT NULL,
            expert_idx INTEGER NOT NULL,
            gate_weight REAL NOT NULL,
            FOREIGN KEY (trace_id) REFERENCES traces(trace_id) ON DELETE CASCADE
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_activations_trace_layer_token
        ON activations (trace_id, layer, token_idx)
    """)

    _migrate_schema(cursor)
    conn.commit()
    conn.close()


def _migrate_schema(cursor: sqlite3.Cursor) -> None:
    """Apply lightweight schema migrations for existing databases."""
    cursor.execute("PRAGMA table_info(traces)")
    columns = {row[1] for row in cursor.fetchall()}
    if "token_ids" not in columns:
        cursor.execute("ALTER TABLE traces ADD COLUMN token_ids TEXT NOT NULL DEFAULT '[]'")
    if "model_id" not in columns:
        cursor.execute("ALTER TABLE traces ADD COLUMN model_id TEXT")


def _normalize_trace_row(trace: dict[str, Any]) -> dict[str, Any]:
    """Normalize DB row fields for API consumers and the visualizer."""
    tokens_raw = trace.get("tokens", "[]")
    token_strs = json.loads(tokens_raw) if isinstance(tokens_raw, str) else tokens_raw
    token_ids_raw = trace.get("token_ids", "[]")
    token_ids = json.loads(token_ids_raw) if isinstance(token_ids_raw, str) else token_ids_raw

    normalized = {
        "id": trace["trace_id"],
        "trace_id": trace["trace_id"],
        "text": trace.get("text", ""),
        "token_strs": token_strs,
        "tokens": token_strs,
        "token_ids": token_ids,
        "num_tokens": trace.get("num_tokens", len(token_ids)),
        "num_layers": trace.get("num_layers"),
        "num_experts": trace.get("num_experts"),
        "top_k": trace.get("top_k"),
        "timestamp": trace.get("timestamp"),
        "model_name": trace.get("model_name"),
        "model_id": trace.get("model_id") or trace.get("model_name"),
    }
    if "activations" in trace:
        normalized["activations"] = trace["activations"]
    return normalized


def _setup_routes(app: FastAPI, db_path: str) -> None:
    """Set up API routes."""
    config = get_config()

    @app.get("/")
    def root():
        return {"message": "MoE Routing Atlas API", "version": "0.1.0"}

    @app.get("/traces")
    def list_traces(limit: int = 100):
        limit = max(1, min(limit, config.max_list_limit))
        conn = _connect_db(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT trace_id, text, num_tokens, num_layers, num_experts, timestamp, model_name "
                "FROM traces ORDER BY trace_id DESC LIMIT ?",
                (limit,),
            )
            traces = [_normalize_trace_row(dict(row)) for row in cursor.fetchall()]
            return traces
        finally:
            conn.close()

    @app.get("/trace/{trace_id}")
    def get_trace(trace_id: int):
        conn = _connect_db(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM traces WHERE trace_id = ?", (trace_id,))
            trace_row = cursor.fetchone()
            if not trace_row:
                raise HTTPException(status_code=404, detail="Trace not found")

            trace = dict(trace_row)
            try:
                trace["activations"] = [
                    dict(row)
                    for row in cursor.execute(
                        "SELECT layer, token_idx, expert_idx, gate_weight "
                        "FROM activations WHERE trace_id = ?",
                        (trace_id,),
                    ).fetchall()
                ]
            except sqlite3.Error as exc:
                raise HTTPException(status_code=500, detail=f"Database error: {exc}") from exc

            return _normalize_trace_row(trace)
        finally:
            conn.close()

    @app.post("/traces")
    def create_trace(trace: Trace):
        conn = _connect_db(db_path)
        cursor = conn.cursor()
        try:
            with conn:
                cursor.execute(
                    "INSERT INTO traces "
                    "(text, tokens, token_ids, num_tokens, num_layers, num_experts, top_k, model_name, model_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        trace.text,
                        json.dumps(trace.token_strs),
                        json.dumps(trace.token_ids),
                        trace.num_tokens,
                        trace.num_layers,
                        trace.num_experts,
                        trace.top_k,
                        trace.model_name,
                        trace.model_id,
                    ),
                )
                new_trace_id = cursor.lastrowid

                for activation in trace.activations:
                    cursor.execute(
                        "INSERT INTO activations "
                        "(trace_id, layer, token_idx, expert_idx, gate_weight) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            new_trace_id,
                            activation.layer,
                            activation.token_idx,
                            activation.expert_idx,
                            activation.gate_weight,
                        ),
                    )
        except sqlite3.Error as exc:
            raise HTTPException(status_code=500, detail=f"Failed to store trace: {exc}") from exc
        finally:
            conn.close()

        return {"trace_id": new_trace_id, "id": new_trace_id, "status": "created"}

    @app.get("/stats")
    def get_stats():
        conn = _connect_db(db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM traces")
            total_traces = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM activations")
            total_activations = cursor.fetchone()[0]

            cursor.execute("""
                SELECT expert_idx, COUNT(*) as count
                FROM activations
                GROUP BY expert_idx
                ORDER BY count DESC
                LIMIT 20
            """)
            expert_usage = [
                {"expert": row[0], "activations": row[1]}
                for row in cursor.fetchall()
            ]

            return {
                "total_traces": total_traces,
                "total_activations": total_activations,
                "expert_usage": expert_usage,
            }
        finally:
            conn.close()
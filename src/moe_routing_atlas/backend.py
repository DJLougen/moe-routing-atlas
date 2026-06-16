"""Backend server factory."""

from __future__ import annotations

import json
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import orjson
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.routing import APIRoute
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .__version__ import __version__
from .config import get_config
from .schema import Trace

MAX_REQUEST_BYTES = 50 * 1024 * 1024


class ORJSONRequest(Request):
    """Request that parses the JSON body with orjson (Rust, ~3x faster than stdlib)."""

    async def json(self) -> Any:
        if not hasattr(self, "_json"):
            self._json = orjson.loads(await self.body())
        return self._json


class ORJSONRoute(APIRoute):
    """Route class wiring ORJSONRequest so request-body parsing uses orjson.

    Keeps FastAPI's normal parameter validation, OpenAPI schema, and 422 error
    handling intact -- only the JSON decode step is swapped.
    """

    def get_route_handler(self):
        original_handler = super().get_route_handler()

        async def handler(request: Request) -> Response:
            return await original_handler(ORJSONRequest(request.scope, request.receive))

        return handler


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
        version=__version__,
        lifespan=lifespan,
    )
    app.router.route_class = ORJSONRoute

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

    # activation_id is a plain rowid alias (no AUTOINCREMENT): it is a surrogate
    # key never read by any query, so the monotonic-no-reuse guarantee is pure
    # per-insert overhead (sqlite_sequence maintenance) on the hot 100k+-row path.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS activations (
            activation_id INTEGER PRIMARY KEY,
            trace_id INTEGER NOT NULL,
            layer INTEGER NOT NULL,
            token_idx INTEGER NOT NULL,
            expert_idx INTEGER NOT NULL,
            gate_weight REAL NOT NULL,
            FOREIGN KEY (trace_id) REFERENCES traces(trace_id) ON DELETE CASCADE
        )
    """)

    # get_trace / DELETE cascade filter activations by trace_id only, so a
    # single-column index matches the access pattern. It is smaller than the
    # former (trace_id, layer, token_idx) composite, giving faster inserts AND
    # faster trace_id lookups (export's ORDER BY is unaffected at scale). Drop
    # the old composite first so existing databases don't keep both.
    cursor.execute("DROP INDEX IF EXISTS idx_activations_trace_layer_token")
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_activations_trace
        ON activations (trace_id)
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


def _text_preview(text: str, max_len: int = 40) -> str:
    """Return a short, non-sensitive preview of trace input text."""
    cleaned = (text or "").replace("\n", " ").strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1] + "…"


def _normalize_trace_row(trace: dict[str, Any], *, include_text: bool = True) -> dict[str, Any]:
    """Normalize DB row fields for API consumers and the visualizer."""
    tokens_raw = trace.get("tokens", "[]")
    token_strs = json.loads(tokens_raw) if isinstance(tokens_raw, str) else tokens_raw
    token_ids_raw = trace.get("token_ids", "[]")
    token_ids = json.loads(token_ids_raw) if isinstance(token_ids_raw, str) else token_ids_raw
    full_text = trace.get("text", "")

    normalized = {
        "id": trace["trace_id"],
        "trace_id": trace["trace_id"],
        "text_preview": _text_preview(full_text),
        "num_tokens": trace.get("num_tokens", len(token_ids)),
        "num_layers": trace.get("num_layers"),
        "num_experts": trace.get("num_experts"),
        "top_k": trace.get("top_k"),
        "timestamp": trace.get("timestamp"),
        "model_name": trace.get("model_name"),
        "model_id": trace.get("model_id") or trace.get("model_name"),
    }
    if include_text:
        normalized["text"] = full_text
        normalized["token_strs"] = token_strs
        normalized["tokens"] = token_strs
        normalized["token_ids"] = token_ids
    if "activations" in trace:
        normalized["activations"] = trace["activations"]
    return normalized


def _setup_routes(app: FastAPI, db_path: str) -> None:
    """Set up API routes."""
    config = get_config()

    @app.get("/")
    def root():
        return {"message": "MoE Routing Atlas API", "version": __version__}

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
            traces = [
                _normalize_trace_row(dict(row), include_text=False)
                for row in cursor.fetchall()
            ]
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
                act_cursor = conn.cursor()
                act_cursor.row_factory = None  # tuples beat sqlite3.Row for 100k+ rows
                trace["activations"] = [
                    {
                        "layer": row[0],
                        "token_idx": row[1],
                        "expert_idx": row[2],
                        "gate_weight": row[3],
                    }
                    for row in act_cursor.execute(
                        "SELECT layer, token_idx, expert_idx, gate_weight "
                        "FROM activations WHERE trace_id = ?",
                        (trace_id,),
                    ).fetchall()
                ]
            except sqlite3.Error as exc:
                raise HTTPException(status_code=500, detail=f"Database error: {exc}") from exc

            # Return a pre-rendered Response so FastAPI skips jsonable_encoder's
            # recursive walk over the (potentially huge) activation list; orjson
            # (Rust) serializes the row dicts ~8x faster than stdlib json. Plain
            # Response (vs ORJSONResponse) stays version-agnostic and avoids
            # FastAPI's response-class deprecation path.
            return Response(
                orjson.dumps(_normalize_trace_row(trace)),
                media_type="application/json",
            )
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

                cursor.executemany(
                    "INSERT INTO activations "
                    "(trace_id, layer, token_idx, expert_idx, gate_weight) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [
                        (
                            new_trace_id,
                            activation.layer,
                            activation.token_idx,
                            activation.expert_idx,
                            activation.gate_weight,
                        )
                        for activation in trace.activations
                    ],
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
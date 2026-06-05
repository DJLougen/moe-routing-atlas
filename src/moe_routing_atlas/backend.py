"""Backend server factory."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import get_config


def create_app(db_path: str = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        db_path: Path to SQLite database (uses config default if None)
    """
    config = get_config()
    db_path = db_path or str(config.db_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager."""
        # Startup: ensure tables exist
        _init_db(db_path)
        yield
        # Shutdown: cleanup

    _init_db(db_path)
    app = FastAPI(
        title="MoE Routing Atlas",
        description="API for Mixture-of-Experts routing visualization",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes
    _setup_routes(app, db_path)

    # Static files (visualizer)
    viz_dir = Path(__file__).parent / "visualizer"
    if viz_dir.exists():
        app.mount("/visualizer", StaticFiles(directory=str(viz_dir), html=True), name="viz")

    return app


def _init_db(db_path: str) -> None:
    """Initialize SQLite database with tables."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS traces (
            trace_id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            tokens TEXT NOT NULL,
            num_tokens INTEGER NOT NULL,
            num_layers INTEGER NOT NULL,
            num_experts INTEGER NOT NULL,
            top_k INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            model_name TEXT
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
            FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_activations_trace_layer_token
        ON activations (trace_id, layer, token_idx)
    """)

    conn.commit()
    conn.close()


def _setup_routes(app: FastAPI, db_path: str) -> None:
    """Set up API routes."""
    import json
    import sqlite3

    import httpx
    from pydantic import BaseModel

    from .schema import Trace

    @app.get("/")
    def root():
        return {"message": "MoE Routing Atlas API", "version": "0.1.0"}

    @app.get("/traces")
    def list_traces(limit: int = 100):
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT trace_id, text, num_tokens, num_layers, num_experts, timestamp, model_name "
            "FROM traces ORDER BY trace_id DESC LIMIT ?",
            (limit,),
        )
        traces = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return traces

    @app.get("/trace/{trace_id}")
    def get_trace(trace_id: int):
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM traces WHERE trace_id = ?", (trace_id,))
        trace_row = cursor.fetchone()
        if not trace_row:
            conn.close()
            return {"error": "Trace not found"}

        trace = dict(trace_row)
        trace["tokens"] = json.loads(trace["tokens"])

        cursor.execute(
            "SELECT layer, token_idx, expert_idx, gate_weight FROM activations WHERE trace_id = ?",
            (trace_id,),
        )
        trace["activations"] = [dict(row) for row in cursor.fetchall()]

        conn.close()
        return trace

    @app.post("/traces")
    def create_trace(trace: Trace):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO traces (text, tokens, num_tokens, num_layers, num_experts, top_k, model_name) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                trace.text,
                json.dumps(trace.token_strs),
                trace.num_tokens,
                trace.num_layers,
                trace.num_experts,
                trace.top_k,
                trace.model_name,
            ),
        )
        trace_id = cursor.lastrowid

        for activation in trace.activations:
            cursor.execute(
                "INSERT INTO activations (trace_id, layer, token_idx, expert_idx, gate_weight) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    trace_id,
                    activation.layer,
                    activation.token_idx,
                    activation.expert_idx,
                    activation.gate_weight,
                ),
            )

        conn.commit()
        conn.close()
        return {"trace_id": trace_id, "status": "created"}

    @app.get("/stats")
    def get_stats():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

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

        conn.close()
        return {
            "total_traces": total_traces,
            "total_activations": total_activations,
            "expert_usage": expert_usage,
        }

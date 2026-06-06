"""Single-text tracer for MoE models."""

from __future__ import annotations

import time
from pathlib import Path

import httpx

from .capture import load_moe_model, run_trace_forward
from .config import get_config
from .schema import Trace


def trace_model(
    text: str,
    model_id: str | None = None,
    quant: str = "nf4",
    device: str = "cuda",
    backend_url: str | None = None,
    trust_remote_code: bool | None = None,
) -> Trace:
    """Trace expert routing for a single text."""
    config = get_config()
    model_id = model_id or config.default_model
    backend_url = backend_url or config.backend_url
    trust_remote_code = (
        config.trust_remote_code if trust_remote_code is None else trust_remote_code
    )

    print(f"[INFO] Loading {model_id} with {quant} quantization...")
    start = time.time()

    model, tokenizer = load_moe_model(
        model_id=model_id,
        quant=quant,
        device=device,
        trust_remote_code=trust_remote_code,
    )

    print(f"[INFO] Model loaded in {time.time() - start:.1f}s")

    trace = run_trace_forward(
        model=model,
        tokenizer=tokenizer,
        text=text,
        model_id=model_id,
        device=device,
    )

    print(
        f"[INFO] Tracing {trace.num_tokens} tokens through {trace.num_layers} layers..."
    )
    print(f"[INFO] Captured {len(trace.activations)} activations")

    if backend_url:
        try:
            response = httpx.post(
                f"{backend_url}/traces",
                json=trace.model_dump(),
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json()
            trace.trace_id = result.get("trace_id")
            print(f"[INFO] Trace sent to backend — ID {trace.trace_id}")
        except Exception as exc:
            print(f"[WARN] Failed to send to backend: {exc}")

    return trace


def trace_model_to_file(
    text: str,
    output_path: str,
    model_id: str | None = None,
    quant: str = "nf4",
    device: str = "cuda",
    trust_remote_code: bool | None = None,
) -> Trace:
    """Trace and save to file."""
    trace = trace_model(
        text,
        model_id=model_id,
        quant=quant,
        device=device,
        backend_url=None,
        trust_remote_code=trust_remote_code,
    )
    Path(output_path).write_text(trace.model_dump_json(indent=2), encoding="utf-8")
    print(f"[INFO] Trace saved to {output_path}")
    return trace
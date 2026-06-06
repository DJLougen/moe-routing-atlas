"""Batch tracer for multiple texts."""

from __future__ import annotations

from typing import List

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .capture import load_moe_model, run_trace_forward
from .config import get_config

console = Console()


def batch_trace(
    texts: List[str],
    model_id: str | None = None,
    quant: str = "nf4",
    device: str = "cuda",
    backend_url: str | None = None,
    trust_remote_code: bool | None = None,
) -> List[dict]:
    """Trace multiple texts with a single model load."""
    config = get_config()
    model_id = model_id or config.default_model
    backend_url = backend_url or config.backend_url
    trust_remote_code = (
        config.trust_remote_code if trust_remote_code is None else trust_remote_code
    )

    console.print(f"[cyan]Loading {model_id}...[/cyan]")
    model, tokenizer = load_moe_model(
        model_id=model_id,
        quant=quant,
        device=device,
        trust_remote_code=trust_remote_code,
    )
    console.print("[green]Model loaded[/green]\n")

    results: List[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for index, text in enumerate(texts):
            task = progress.add_task(f"Tracing {index + 1}/{len(texts)}...", total=None)

            try:
                trace = run_trace_forward(
                    model=model,
                    tokenizer=tokenizer,
                    text=text,
                    model_id=model_id,
                    device=device,
                )

                trace_id = None
                if backend_url:
                    try:
                        response = httpx.post(
                            f"{backend_url}/traces",
                            json=trace.model_dump(),
                            timeout=30.0,
                        )
                        response.raise_for_status()
                        trace_id = response.json().get("trace_id")
                        trace.trace_id = trace_id
                    except Exception as exc:
                        console.print(f"[yellow]Backend error: {exc}[/yellow]")

                results.append(
                    {
                        "text": text[:50],
                        "num_tokens": trace.num_tokens,
                        "num_activations": len(trace.activations),
                        "trace_id": trace_id,
                    }
                )

            except Exception as exc:
                console.print(f"[red]Failed to trace: {exc}[/red]")
                results.append(
                    {
                        "text": text[:50],
                        "error": str(exc),
                        "trace_id": None,
                    }
                )

            progress.remove_task(task)

    return results
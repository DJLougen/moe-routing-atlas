"""Batch tracer for multiple texts."""

from pathlib import Path
from typing import List

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import get_config
from .schema import Trace
from .tracer import trace_model

console = Console()


def batch_trace(
    texts: List[str],
    model_id: str = None,
    quant: str = "nf4",
    device: str = "cuda",
    backend_url: str = None,
) -> List[dict]:
    """Trace multiple texts with a single model load.

    Args:
        texts: List of texts to trace
        model_id: HuggingFace model ID
        quant: Quantization type
        device: Compute device
        backend_url: Backend to send traces to

    Returns:
        List of result dicts with trace_id and metadata
    """
    config = get_config()
    model_id = model_id or config.default_model
    backend_url = backend_url or config.backend_url

    # Load model once
    console.print(f"[cyan]Loading {model_id}...[/cyan]")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": device if device != "cpu" else None,
        "trust_remote_code": True,
    }

    if quant in ("nf4", "nb4"):
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4" if quant == "nf4" else "fp4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif quant == "int8":
        load_kwargs["load_in_8bit"] = True

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    console.print("[green]Model loaded[/green]\n")

    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for i, text in enumerate(texts):
            task = progress.add_task(f"Tracing {i+1}/{len(texts)}...", total=None)

            try:
                trace = _trace_with_loaded_model(
                    text=text,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                )

                # Send to backend
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
                    except Exception as e:
                        console.print(f"[yellow]Backend error: {e}[/yellow]")

                results.append({
                    "text": text[:50],
                    "num_tokens": trace.num_tokens,
                    "num_activations": len(trace.activations),
                    "trace_id": trace_id,
                })

            except Exception as e:
                console.print(f"[red]Failed to trace: {e}[/red]")
                results.append({
                    "text": text[:50],
                    "error": str(e),
                    "trace_id": None,
                })

            progress.remove_task(task)

    return results


def _trace_with_loaded_model(
    text: str,
    model,
    tokenizer,
    device: str = "cuda",
) -> Trace:
    """Trace using already-loaded model."""
    import torch

    inputs = tokenizer(text, return_tensors="pt", padding=True)
    if device == "cuda" and torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    token_ids = inputs["input_ids"][0].tolist()
    token_strs = [tokenizer.decode([tid]) for tid in token_ids]

    activations = []
    num_layers = 0
    num_experts = 0
    top_k = 0

    def capture_routing(module, input, output, layer_idx):
        nonlocal num_experts, top_k
        if hasattr(module, "gate") and hasattr(module.gate, "top_k"):
            top_k = module.gate.top_k

        topk_idx = getattr(output, "topk_idx", None)
        topk_weight = getattr(output, "topk_weight", None)

        if topk_idx is None and isinstance(output, tuple):
            topk_idx = output[1] if len(output) > 1 else None
            topk_weight = output[2] if len(output) > 2 else None

        if topk_idx is not None and topk_weight is not None:
            for token_pos in range(topk_idx.shape[0]):
                for expert_pos in range(topk_idx.shape[1]):
                    expert = topk_idx[token_pos, expert_pos].item()
                    weight = topk_weight[token_pos, expert_pos].item()
                    from .schema import TraceActivation
                    activations.append(TraceActivation(
                        layer=layer_idx,
                        token_idx=token_pos,
                        expert_idx=expert,
                        gate_weight=weight,
                    ))
                    if expert >= num_experts:
                        num_experts = expert + 1

    hooks = []
    for name, module in model.named_modules():
        if "moe" in name.lower() or "experts" in name.lower():
            layer_idx = num_layers
            hook = module.register_forward_hook(
                lambda m, i, o, idx=layer_idx: capture_routing(m, i, o, idx)
            )
            hooks.append(hook)
            num_layers += 1

    with torch.no_grad():
        model(**inputs)

    for hook in hooks:
        hook.remove()

    return Trace(
        model_id=model.config.name_or_path if hasattr(model, "config") else "unknown",
        model_class=model.__class__.__name__,
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        text=text,
        token_ids=token_ids,
        token_strs=token_strs,
        activations=activations,
    )

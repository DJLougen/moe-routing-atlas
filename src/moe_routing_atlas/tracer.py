"""Single-text tracer for MoE models."""

import time
from pathlib import Path

import httpx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import get_config
from .schema import Trace, TraceActivation


def trace_model(
    text: str,
    model_id: str = None,
    quant: str = "nf4",
    device: str = "cuda",
    backend_url: str = None,
) -> Trace:
    """Trace expert routing for a single text.

    Args:
        text: Input text to trace
        model_id: HuggingFace model ID (uses config default if None)
        quant: Quantization (nb4, nf4, int8, none)
        device: Compute device (cuda, mps, cpu)
        backend_url: URL to send trace to (saves locally if None)

    Returns:
        Trace object with all routing data
    """
    config = get_config()
    model_id = model_id or config.default_model
    backend_url = backend_url or config.backend_url

    print(f"[INFO] Loading {model_id} with {quant} quantization...")
    start = time.time()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization
    load_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": device if device != "cpu" else None,
        "trust_remote_code": True,
    }

    if quant == "nf4":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif quant == "nb4":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif quant == "int8":
        load_kwargs["load_in_8bit"] = True

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    print(f"[INFO] Model loaded in {time.time() - start:.1f}s")

    # Prepare input
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    if device == "cuda" and torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Tokenize to get strings
    token_ids = inputs["input_ids"][0].tolist()
    token_strs = [tokenizer.decode([tid]) for tid in token_ids]

    # Hook for routing capture
    activations = []
    num_layers = 0
    num_experts = 0
    top_k = 0

    def capture_routing(module, input, output, layer_idx):
        nonlocal num_experts, top_k
        # Extract routing info from MoE block
        # This is model-specific; see model compatibility notes
        if hasattr(module, "gate") and hasattr(module.gate, "top_k"):
            top_k = module.gate.top_k

        # Try to get routing weights
        if hasattr(output, "aux_loss"):
            # Mixtral / Qwen style
            topk_idx = output[1] if isinstance(output, tuple) else None
            topk_weight = output[2] if isinstance(output, tuple) else None
        elif hasattr(module, "topk_idx"):
            topk_idx = module.topk_idx
            topk_weight = module.topk_weight
        else:
            return

        if topk_idx is not None and topk_weight is not None:
            for token_pos in range(topk_idx.shape[0]):
                for expert_pos in range(topk_idx.shape[1]):
                    expert = topk_idx[token_pos, expert_pos].item()
                    weight = topk_weight[token_pos, expert_pos].item()
                    activations.append(TraceActivation(
                        layer=layer_idx,
                        token_idx=token_pos,
                        expert_idx=expert,
                        gate_weight=weight,
                    ))
                    if expert >= num_experts:
                        num_experts = expert + 1

    # Register hooks on MoE layers
    hooks = []
    for name, module in model.named_modules():
        if "moe" in name.lower() or "experts" in name.lower():
            layer_idx = num_layers
            hook = module.register_forward_hook(
                lambda m, i, o, idx=layer_idx: capture_routing(m, i, o, idx)
            )
            hooks.append(hook)
            num_layers += 1

    # Forward pass
    print(f"[INFO] Tracing {len(token_ids)} tokens through {num_layers} layers...")
    with torch.no_grad():
        outputs = model(**inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    print(f"[INFO] Captured {len(activations)} activations")

    # Build trace
    trace = Trace(
        model_id=model_id,
        model_class=model.__class__.__name__,
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        text=text,
        token_ids=token_ids,
        token_strs=token_strs,
        activations=activations,
        model_name=model_id.split("/")[-1],
    )

    # Send to backend if requested
    if backend_url:
        try:
            response = httpx.post(
                f"{backend_url}/traces",
                json=trace.model_dump(),
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json()
            trace_id = result.get("trace_id")
            print(f"[INFO] Trace sent to backend — ID {trace_id}")
        except Exception as e:
            print(f"[WARN] Failed to send to backend: {e}")

    return trace


def trace_model_to_file(
    text: str,
    output_path: str,
    model_id: str = None,
    quant: str = "nf4",
    device: str = "cuda",
) -> Trace:
    """Trace and save to file."""
    trace = trace_model(text, model_id=model_id, quant=quant, device=device, backend_url=None)
    Path(output_path).write_text(trace.model_dump_json(indent=2), encoding="utf-8")
    print(f"[INFO] Trace saved to {output_path}")
    return trace

"""Shared MoE routing capture utilities."""

from __future__ import annotations

import re
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .schema import Trace, TraceActivation

_MOE_BLOCK_HINTS = (
    "SparseMoeBlock",
    "MoEBlock",
    "MoeBlock",
    "MoELayer",
)


def is_moe_routing_block(name: str, module: nn.Module) -> bool:
    """Return True for top-level MoE router blocks, not individual experts."""
    cls_name = module.__class__.__name__
    if any(hint in cls_name for hint in _MOE_BLOCK_HINTS):
        return True
    if hasattr(module, "gate") and hasattr(module, "experts"):
        return not bool(re.search(r"experts\.\d+$", name))
    if "moe" in name.lower() and hasattr(module, "gate"):
        return not bool(re.search(r"experts\.\d+$", name))
    return False


def extract_layer_index(name: str, fallback: int) -> int:
    """Parse transformer layer index from module name."""
    for pattern in (r"\.layers\.(\d+)\.", r"\.layer\.(\d+)\.", r"\.h\.(\d+)\."):
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    return fallback


def extract_routing_from_output(
    module: nn.Module, output: Any
) -> tuple[Any, Any] | tuple[None, None]:
    """Extract top-k expert indices and weights from a MoE block output."""
    topk_idx = getattr(output, "topk_idx", None)
    topk_weight = getattr(output, "topk_weight", None)

    if topk_idx is None and isinstance(output, tuple):
        if len(output) > 1:
            topk_idx = output[1]
        if len(output) > 2:
            topk_weight = output[2]

    if topk_idx is None and hasattr(module, "topk_idx"):
        topk_idx = module.topk_idx
        topk_weight = module.topk_weight

    if topk_idx is not None and topk_weight is not None:
        return topk_idx, topk_weight
    return None, None


def register_moe_hooks(model: nn.Module) -> tuple[list[Any], dict[str, Any]]:
    """Register forward hooks on MoE router blocks."""
    state: dict[str, Any] = {
        "activations": [],
        "num_experts": 0,
        "top_k": 0,
        "layer_indices": set(),
    }

    def capture_routing(module: nn.Module, _input: Any, output: Any, layer_idx: int) -> None:
        if hasattr(module, "gate") and hasattr(module.gate, "top_k"):
            state["top_k"] = module.gate.top_k

        topk_idx, topk_weight = extract_routing_from_output(module, output)
        if topk_idx is None:
            return

        for token_pos in range(topk_idx.shape[0]):
            for expert_pos in range(topk_idx.shape[1]):
                expert = topk_idx[token_pos, expert_pos].item()
                weight = topk_weight[token_pos, expert_pos].item()
                state["activations"].append(
                    TraceActivation(
                        layer=layer_idx,
                        token_idx=token_pos,
                        expert_idx=expert,
                        gate_weight=weight,
                    )
                )
                if expert >= state["num_experts"]:
                    state["num_experts"] = expert + 1

    hooks: list[Any] = []
    fallback = 0
    for name, module in model.named_modules():
        if not is_moe_routing_block(name, module):
            continue
        layer_idx = extract_layer_index(name, fallback)
        fallback += 1
        state["layer_indices"].add(layer_idx)
        hook = module.register_forward_hook(
            lambda m, i, o, idx=layer_idx: capture_routing(m, i, o, idx)
        )
        hooks.append(hook)

    state["num_layers"] = len(state["layer_indices"]) or fallback
    return hooks, state


def remove_hooks(hooks: list[Any]) -> None:
    """Remove registered forward hooks."""
    for hook in hooks:
        hook.remove()


def move_inputs_to_device(inputs: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    """Move tokenized inputs to the requested device."""
    if device == "cpu":
        return inputs
    if device == "cuda" and not torch.cuda.is_available():
        return inputs
    if device == "mps" and not torch.backends.mps.is_available():
        return inputs
    return {key: value.to(device) for key, value in inputs.items()}


def load_moe_model(
    model_id: str,
    quant: str = "nf4",
    device: str = "cuda",
    trust_remote_code: bool = False,
) -> tuple[Any, Any]:
    """Load a HuggingFace MoE model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch.float16,
        "device_map": device if device != "cpu" else None,
        "trust_remote_code": trust_remote_code,
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
    return model, tokenizer


def run_trace_forward(
    model: Any,
    tokenizer: Any,
    text: str,
    model_id: str,
    device: str = "cuda",
) -> Trace:
    """Trace routing for one text using a loaded model."""
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    inputs = move_inputs_to_device(inputs, device)

    token_ids = inputs["input_ids"][0].tolist()
    token_strs = [tokenizer.decode([token_id]) for token_id in token_ids]

    hooks, state = register_moe_hooks(model)
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        remove_hooks(hooks)

    model_name = model_id.split("/")[-1]
    if hasattr(model, "config") and getattr(model.config, "name_or_path", None):
        model_name = model.config.name_or_path.split("/")[-1]

    return Trace(
        model_id=model_id,
        model_class=model.__class__.__name__,
        num_layers=state["num_layers"],
        num_experts=state["num_experts"],
        top_k=state["top_k"],
        text=text,
        token_ids=token_ids,
        token_strs=token_strs,
        activations=state["activations"],
        model_name=model_name,
    )
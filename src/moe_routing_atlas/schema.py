"""Data schema for the MoE Routing Atlas.

A trace captures the routing decisions for a single forward pass through a MoE model.
Traces are appendable — you can accumulate thousands of them for population-level analysis.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TraceActivation(BaseModel):
    """One expert activation at one layer for one token."""

    layer: int = Field(..., description="Layer index")
    token_idx: int = Field(..., description="Token position in sequence")
    expert_idx: int = Field(..., description="Expert index within layer")
    gate_weight: float = Field(..., description="Router gate weight for this expert")
    output_norm: float = Field(default=0.0, description="Output norm if available, else 0.0")


class Trace(BaseModel):
    """A single forward pass trace through a MoE model."""

    # Model identity
    model_id: str = Field(..., description="HuggingFace model ID")
    model_class: str = Field(default="", description="Model class name")
    num_layers: int = Field(..., description="Number of MoE layers")
    num_experts: int = Field(..., description="Number of experts per layer")
    top_k: int = Field(..., description="Top-k routing parameter")
    hidden_dim: int = Field(default=0, description="Hidden dimension size")

    # Input
    text: str = Field(default="", description="Input text")
    token_ids: list[int] = Field(default_factory=list, description="Token ID sequence")
    token_strs: list[str] = Field(default_factory=list, description="Token string sequence")

    # Routing decisions
    activations: list[TraceActivation] = Field(
        default_factory=list, description="All expert activations"
    )

    # Metadata
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO 8601 timestamp",
    )
    model_name: str = Field(default="", description="Model name for display")

    # Derived properties
    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    def to_json(self) -> dict[str, Any]:
        """Export to JSON-compatible dict."""
        return self.model_dump()

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Trace":
        """Import from JSON-compatible dict."""
        return cls.model_validate(data)

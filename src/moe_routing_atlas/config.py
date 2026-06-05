"""Configuration management for MoE Atlas."""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class AtlasConfig(BaseSettings):
    """Runtime configuration for MoE Atlas."""

    # Backend
    backend_host: str = Field(default="0.0.0.0", description="Backend server host")
    backend_port: int = Field(default=8000, description="Backend server port")
    backend_url: str = Field(default="http://localhost:8000", description="Full backend URL")

    # Database
    db_path: Path = Field(
        default=Path.home() / ".moe-atlas" / "atlas.db",
        description="SQLite database path",
    )

    # Model defaults
    default_model: str = Field(
        default="Qwen/Qwen1.5-MoE-A2.7B",
        description="Default HuggingFace model ID",
    )
    default_quantization: str = Field(
        default="nf4",
        description="Default quantization (nb4, nf4, int8, none)",
    )
    default_device: str = Field(
        default="cuda",
        description="Default compute device (cuda, mps, cpu)",
    )

    # Tracing
    max_experts_viz: int = Field(
        default=60,
        description="Max experts to visualize (performance limit)",
    )
    batch_size: int = Field(
        default=1,
        description="Batch size for tracing",
    )

    # Export
    export_dir: Path = Field(
        default=Path.home() / ".moe-atlas" / "exports",
        description="Directory for exported traces",
    )

    # Visualization
    visualizer_theme: str = Field(
        default="dark",
        description="Visualizer color theme (dark, light)",
    )
    auto_rotate: bool = Field(
        default=False,
        description="Auto-rotate camera by default",
    )

    class Config:
        env_prefix = "MOE_ATLAS_"
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global config instance
_config: Optional[AtlasConfig] = None


def get_config() -> AtlasConfig:
    """Get or create global config instance."""
    global _config
    if _config is None:
        _config = AtlasConfig()
        # Ensure directories exist
        _config.db_path.parent.mkdir(parents=True, exist_ok=True)
        _config.export_dir.mkdir(parents=True, exist_ok=True)
    return _config


def set_config(config: AtlasConfig) -> None:
    """Set global config instance."""
    global _config
    _config = config

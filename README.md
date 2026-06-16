# MoE Routing Atlas

> **Open-source toolkit for mapping and visualizing Mixture-of-Experts model routing patterns.**

## What Is This Repo?

This repository contains **MoE Atlas** — a command-line tool and web visualizer that lets you see inside Mixture-of-Experts (MoE) language models. 

MoE models (like Mixtral, Qwen-MoE, Gemma-4) use sparse routing: for each token, a "router" network picks a small subset of "expert" feed-forward networks to process it. This repo captures those routing decisions during inference and renders them as an interactive 3D scene you can explore token-by-token, layer-by-layer.

**Key idea:** Most of an MoE model's parameters are never touched for any given token. This tool makes that sparsity visible.

**Motivation:** [Cerebras](https://www.cerebras.ai/blog/moe-guide-why-moe) describes sparse MoE as the future of efficient large-scale AI — enabling trillion-parameter models without trillion-parameter compute. We built this to help researchers and practitioners understand what actually happens inside these routers.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/badge/pip-moe--atlas-blue)](https://pypi.org/project/moe-atlas/)
[![Release](https://img.shields.io/badge/release-v0.4.0-22c55e)](https://github.com/DJLougen/moe-routing-atlas/releases/latest)

See which experts activate for each token, layer by layer, in an interactive 3D visualization.

> **New in v0.4.0 — community trace files.** Routing traces are an *appendable* shared dataset. Pool contributors' files with `moe-atlas merge`, grow a canonical file with `moe-atlas export --append`, and gate pull requests with `moe-atlas validate` — all content-deduplicated and schema-validated so one bad file can't corrupt the set. See [Build a Community Trace Set](#build-a-community-trace-set).

---

## What This Does

Mixture-of-Experts (MoE) models like **Mixtral**, **Qwen2-MoE**, and **Gemma-4** route each token through a small subset of "expert" networks per layer. **MoE Atlas** captures these routing decisions and renders them as an explorable 3D scene.

### Visual Elements

| Element | What It Shows |
|---------|--------------|
| 🔵 **Expert Spheres** | Light up when activated by the router (brightness = gate weight) |
| 🔷 **Routers** | Spin only when their layer has active experts |
| 🔗 **Cyan Beams** | Show router → expert connections |
| 🟢 **Green Flow Lines** | Trace how tokens move between layers |
| ✨ **Pulse Particles** | Animate along active routing paths |

Every visual element is tied to **actual routing data** — nothing is decorative.

### Use It To

- **Understand** which experts handle which concepts
- **Compare** routing patterns across different inputs
- **Debug** MoE behavior layer-by-layer
- **Share** routing maps with collaborators
- **Research** population-level expert usage patterns
- **Contribute** traces to a shared, deduplicated community dataset

---

## Screenshots

### Main Visualization

*Interactive 3D scene showing 24 layers of expert routing. Click any token to see its path.*

### Token Routing Detail

![Token Routing](docs/screenshots/viz-token-detail.png)
*A single token's routing through all layers. Orange = active experts, cyan = routers, green = cross-layer flow.*

### Animation Playback

![Animation Playback](docs/screenshots/viz-animation.png)
*Auto-play through tokens to see how routing evolves across the sequence.*

### Trace Gallery

![Trace Gallery](docs/screenshots/viz-traces.png)
*Load and compare multiple traces side-by-side.*

> **Note:** Screenshots above are placeholders. Run the visualizer and take your own — the scene is fully interactive. See [Taking Screenshots](#taking-screenshots) below.

---

## Quick Start

### Install

```bash
pip install moe-atlas
```

Or from source:

```bash
git clone https://github.com/moe-atlas/moe-routing-atlas.git
cd moe-routing-atlas
pip install -e ".[dev]"
```

### 1. Start the Backend

```bash
moe-atlas serve
```

```
[INFO] Starting backend server on http://0.0.0.0:8000
[INFO] Database: ~/.moe-atlas/atlas.db
```

### 2. Trace Your First Model

In another terminal:

```bash
moe-atlas trace "The quick brown fox jumps over the lazy dog."
```

```
[INFO] Loading Qwen/Qwen1.5-MoE-A2.7B with nf4 quantization...
[INFO] Model loaded in 4.2s
[INFO] Tracing 10 tokens through 24 layers...
[INFO] Captured 960 activations
[INFO] Trace sent to backend — ID 1
```

### 3. Open the Visualizer

```bash
moe-atlas viz
```

Or open [http://localhost:8000/visualizer](http://localhost:8000/visualizer) manually.

### 4. Explore

1. Click a **trace** in the top-left grid
2. Click a **token** in the top bar
3. Watch the 3D scene light up
4. Hit **Play** to animate through tokens

**Mouse Controls:**
- Left-click + drag: Rotate camera
- Right-click + drag: Pan
- Scroll: Zoom
- Click token: Show routing for that token

**Keyboard:**
- `Space`: Play/Pause
- `R`: Toggle auto-rotate
- `Esc`: Reset camera

---

## Batch Tracing

Trace multiple texts efficiently (model loads once):

```bash
# Create input file
cat > texts.txt << 'EOF'
The quick brown fox jumps over the lazy dog.
Machine learning models are fascinating.
Mixture of experts enables efficient scaling.
Climate change is a pressing global issue.
EOF

# Trace all of them
moe-atlas batch texts.txt
```

Output:
```
Batch Trace Results
┏━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┓
┃ #  ┃ Tokens  ┃ Trace ID ┃ Status ┃
┡━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━┩
│ 1  │ 10      │ 42       │ OK     │
│ 2  │ 6       │ 43       │ OK     │
│ 3  │ 7       │ 44       │ OK     │
│ 4  │ 8       │ 45       │ OK     │
└────┴─────────┴──────────┴────────┘
```

---

## Supported Models

| Model | Experts | Layers | top_k | VRAM* | Status |
|-------|---------|--------|-------|-------|--------|
| `Qwen/Qwen1.5-MoE-A2.7B` | 64 | 24 | 4 | ~6GB | ✅ Tested |
| `mistralai/Mixtral-8x7B-v0.1` | 8 | 32 | 2 | ~24GB | ✅ Compatible |
| `mistralai/Mixtral-8x22B-v0.1` | 8 | 56 | 2 | ~80GB | ✅ Compatible |
| `google/gemma-4-26B-A4B-it` | 128 | 30 | 8 | ~20GB | 🔄 Ready |
| `deepseek-ai/deepseek-moe-16b-base` | 64 | 28 | 6 | ~12GB | ✅ Compatible |
| `Qwen/Qwen2-57B-A14B` | 128 | 28 | 8 | ~16GB | ✅ Compatible |

*VRAM with 4-bit quantization. Use `--quant nf4` to reduce memory.

**Adding a model:** If it uses HuggingFace `transformers` with standard MoE blocks, it likely works. The tracer auto-detects MoE layers.

---

## CLI Reference

```
moe-atlas --help
```

### Commands

| Command | Description | Example |
|---------|-------------|---------|
| `serve` | Start backend server | `moe-atlas serve --port 8000` |
| `trace` | Trace single text | `moe-atlas trace "Hello world"` |
| `batch` | Trace multiple texts | `moe-atlas batch texts.txt` |
| `viz` | Open visualizer in browser | `moe-atlas viz --browser` |
| `export` | Export traces to file | `moe-atlas export --format jsonl` |
| `import-traces` | Import traces from file | `moe-atlas import-traces traces.jsonl` |
| `merge` | Combine trace files (deduplicated) | `moe-atlas merge a.jsonl b.jsonl -o all.jsonl` |
| `validate` | Check a trace file before sharing | `moe-atlas validate traces.jsonl` |
| `init` | Create config directory | `moe-atlas init` |
| `config-show` | Show current config | `moe-atlas config-show` |

### Options

```bash
# Use a different model
moe-atlas trace "Hello" --model mistralai/Mixtral-8x7B-v0.1

# Use CPU instead of GPU
moe-atlas trace "Hello" --device cpu

# No quantization (slower, more VRAM)
moe-atlas trace "Hello" --quant none

# Send to friend's backend
moe-atlas trace "Hello" --backend http://friend-ip:8000

# Save trace to file instead of backend
moe-atlas trace "Hello" --output my-trace.json
```

### Environment Variables

All config options can be set via `MOE_ATLAS_*` variables:

```bash
export MOE_ATLAS_BACKEND_PORT=8080
export MOE_ATLAS_DEFAULT_MODEL="mistralai/Mixtral-8x7B-v0.1"
export MOE_ATLAS_DEFAULT_QUANTIZATION="nf4"
export MOE_ATLAS_DB_PATH="/custom/path/atlas.db"
export MOE_ATLAS_BACKEND_URL="http://192.168.1.100:8000"
```

---

## Sharing Traces

### Export Your Traces

```bash
# Export all traces to JSONL (recommended)
moe-atlas export --format jsonl
# → ~/.moe-atlas/exports/moe_atlas_export_20240605_143022.jsonl

# Or as a single JSON file
moe-atlas export --format json

# Or as Parquet for analysis
moe-atlas export --format parquet
```

### Share the File

Send the export file to anyone:

```bash
# Email, Slack, upload to GitHub, etc.
cp moe_atlas_export_20240605_143022.jsonl ~/Desktop/
```

### Friend Imports Your Traces

```bash
moe-atlas import-traces moe_atlas_export_20240605_143022.jsonl
```

```
[INFO] Importing 15 traces...
[INFO] Imported 15/15 traces successfully
```

### Direct Send to Backend

Skip the file and send directly:

```bash
# On your machine
moe-atlas trace "Your text" --backend http://friend-ip:8000
```

### File Format

Traces are standard JSON/JSONL — no proprietary format:

```json
{
  "model_id": "Qwen/Qwen1.5-MoE-A2.7B",
  "num_layers": 24,
  "num_experts": 64,
  "top_k": 4,
  "text": "The quick brown fox",
  "token_strs": ["The", " quick", " brown", " fox"],
  "activations": [
    {"layer": 0, "token_idx": 0, "expert_idx": 15, "gate_weight": 0.42},
    ...
  ],
  "timestamp": "2026-06-05T14:30:22.123Z"
}
```

### Build a Community Trace Set

Routing traces are most useful in bulk, so the JSONL format is **appendable** — one
self-contained trace per line — and the CLI makes a shared file easy to grow safely.

**Add your local traces to a shared file** (duplicates are skipped automatically):

```bash
moe-atlas export --append community.jsonl
```

**Combine files from several contributors** into one (deduplicated and validated):

```bash
moe-atlas merge community.jsonl alice.jsonl bob.jsonl --output community.jsonl
```

The output may also be one of the inputs, so a canonical file grows in place. Traces are
deduplicated by their *routing data* (model + input tokens + expert activations), so the
same prompt traced by two people is stored once — regardless of activation ordering,
display name, or timestamp.

**Validate before you share or open a pull request** (exits non-zero on bad records):

```bash
moe-atlas validate community.jsonl
# community.jsonl: 1280 valid, 12 duplicate, 0 invalid
```

Malformed or schema-invalid records are reported and skipped during a merge, so one bad
contribution can never corrupt the shared file.

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Your Prompt   │────▶│  MoE Atlas      │────▶│  SQLite DB      │
│                 │     │  Tracer         │     │  (traces +      │
└─────────────────┘     └─────────────────┘     │  activations)   │
                                                  └─────────────────┘
                                                           │
                                                           ▼
                                                  ┌─────────────────┐
                                                  │  FastAPI        │
                                                  │  Backend        │
                                                  └─────────────────┘
                                                           │
                                                           ▼
                                                  ┌─────────────────┐
                                                  │  Three.js       │
                                                  │  Visualizer     │
                                                  └─────────────────┘
```

**Data flow:**
1. **Tracer** hooks into MoE layers during inference
2. **Backend** stores traces in SQLite
3. **Visualizer** renders a 3D scene

---

## Taking Screenshots

To add your own screenshots to the docs:

```bash
# Start the backend and visualizer
moe-atlas serve &
moe-atlas viz

# Take screenshots from your browser
# Or use the built-in export:
# Press 'S' in the visualizer to save a PNG
```

Place screenshots in `docs/screenshots/` and they will appear in this README.

---

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for full setup.

Quick start:

```bash
git clone https://github.com/moe-atlas/moe-routing-atlas.git
cd moe-routing-atlas
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

Run tests:

```bash
pytest
```

---

## Troubleshooting

### Out of Memory

Use 4-bit quantization:

```bash
moe-atlas trace "Hello" --quant nf4
```

Reduces VRAM from ~24GB to ~6GB with minimal quality loss.

### Model Not Supported

The tracer auto-detects MoE layers. If your model fails:

1. Check it uses HuggingFace `transformers`
2. Check it has `moe` or `experts` in layer names
3. File an issue with the model ID

### Backend Won't Start

```bash
# Check port
lsof -i :8000
# Or use different port
moe-atlas serve --port 8080
```

### Visualizer is Blank

Make sure the backend is running and has traces:

```bash
moe-atlas serve &
moe-atlas trace "Test"
moe-atlas viz
```

### Import Errors

```bash
# Reinstall in development mode
pip install -e .

# Or upgrade dependencies
pip install -e ".[dev]" --upgrade
```

---

## References & Background

### Why MoE Routing Matters

Mixture-of-Experts (MoE) is the dominant architecture for scaling language models beyond dense parameter limits. Instead of activating every parameter for every token, MoE models use a learned router to dispatch each token to a small subset of expert feed-forward networks. This enables trillion-parameter models with sub-trillion compute costs.

**Cerebras on sparse MoE:**
- [MoE Fundamentals: Why Sparse Models Are the Future of AI](https://www.cerebras.ai/blog/moe-guide-why-moe) — Cerebras explains how sparse routing enables models like GPT-4 scale without proportional compute cost
- [MoE at Scale: Making Sparse Models Fast on Real Hardware](https://www.cerebras.ai/blog/moe-guide-scale) — How sparse routing subdivides batches across experts and the hardware challenges this creates

**Key insight from Cerebras:** "With a sparse MoE model, our routing subdivides the batch size across many experts. It results in most experts only seeing a tiny portion of the batch." This is exactly what MoE Atlas visualizes — which experts see which tokens.

### Related Work

| Resource | What It Covers |
|----------|---------------|
| [Cerebras MoE Guide](https://www.cerebras.ai/blog/moe-guide-why-moe) | Why sparse MoE is the future of efficient large-scale AI |
| [Representation Collapse of Sparse MoE](https://openreview.net/forum?id=mWaYC6CZf5) (Chi et al., NeurIPS 2022) | How naive routing causes token representations to cluster excessively |
| [Switch Transformers](https://arxiv.org/abs/2101.03961) (Fedus et al., 2021) | Google's trillion-parameter sparse model with top-1 routing |
| [Mixtral of Experts](https://arxiv.org/abs/2401.04088) (Jiang et al., 2024) | Open-source 8x7B sparse model, state-of-the-art for its size |

---

## Citation

If you use MoE Atlas in research:

```bibtex
@software{moe_atlas_2024,
  title = {MoE Routing Atlas: Open-source toolkit for mapping Mixture-of-Experts routing patterns},
  author = {MoE Atlas Contributors},
  year = {2024},
  url = {https://github.com/moe-atlas/moe-routing-atlas},
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md).

**Ways to help:**
- 🐛 Report bugs via [Issues](https://github.com/moe-atlas/moe-routing-atlas/issues)
- 💡 Suggest features
- 🔧 Add support for new MoE architectures
- 📊 Share interesting routing traces
- 🎨 Improve the 3D visualizer
- 📝 Improve documentation

---

## Roadmap

**Near term:**
- [x] `moe-atlas merge` command for combining trace files (with `export --append` and `validate`)
- [ ] Real-time tracing (watch routing as model generates)
- [ ] Expert usage heatmap overlay
- [ ] Export visualization as PNG/MP4

**Medium term:**
- [ ] Multi-model comparison (Qwen vs Mixtral vs Gemma)
- [ ] Statistical analysis dashboard
- [ ] Public trace gallery
- [ ] PyPI stable release

**Long term:**
- [ ] Research publication on MoE routing patterns
- [ ] Community-curated expert behavior dataset
- [ ] Interactive paper with embedded visualizations

---

**Made with ❤️ by the MoE Atlas community.**

*Mapping the hidden roads inside Mixture-of-Experts models.*

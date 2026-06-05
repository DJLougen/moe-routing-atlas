# MoE Routing Atlas

> **Open-source toolkit for mapping and visualizing Mixture-of-Experts model routing patterns.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/badge/pip-moe--atlas-blue)](https://pypi.org/project/moe-atlas/)

See which experts activate for each token, layer by layer, in an interactive 3D visualization.

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

---

## Screenshots

### Main Visualization

![Visualizer Overview](docs/screenshots/viz-overview.png)
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

**Future: File Combining**

We're working on a simpler way to combine trace files — `moe-atlas merge traces1.jsonl traces2.jsonl --output combined.jsonl`. For now, JSONL files can be concatenated directly: `cat traces1.jsonl traces2.jsonl > combined.jsonl`.

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
- [ ] `moe-atlas merge` command for combining trace files
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

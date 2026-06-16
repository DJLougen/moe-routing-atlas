# Contributing to MoE Routing Atlas

Thank you for your interest in contributing! This document will help you get started.

---

## Development Setup

### Prerequisites

- Python 3.10 or later
- Git
- A GPU with CUDA (optional but recommended for tracing)

### Clone and Install

```bash
git clone https://github.com/moe-atlas/moe-routing-atlas.git
cd moe-routing-atlas

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Verify Installation

```bash
moe-atlas --version
moe-atlas config-show
```

---

## Project Structure

```
moe-routing-atlas/
├── src/
│   └── moe_routing_atlas/
│       ├── __init__.py          # Package init
│       ├── __version__.py       # Version string
│       ├── cli.py               # Command-line interface
│       ├── config.py            # Configuration management
│       ├── schema.py            # Data models (Trace, Activation)
│       ├── backend.py           # FastAPI server
│       ├── tracer.py            # Single-text tracing
│       ├── batch_tracer.py      # Batch tracing
│       ├── exporter.py          # Export/import utilities
│       └── visualizer/
│           └── index.html       # Three.js 3D scene
├── tests/                       # Test suite
├── pyproject.toml              # Build config
├── README.md                   # User documentation
├── CONTRIBUTING.md             # This file
└── LICENSE                     # MIT License
```

---

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Follow existing code style (Black + Ruff)
- Add tests for new functionality
- Update documentation if needed

### 3. Run Quality Checks

```bash
# Formatting
black src/ tests/

# Linting
ruff check src/ tests/

# Type checking
mypy src/

# Tests
pytest
```

### 4. Commit

```bash
git add .
git commit -m "feat: add your feature description"
```

Follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation
- `test:` — tests
- `refactor:` — code refactoring
- `perf:` — performance improvement

### 5. Push and PR

```bash
git push origin feature/your-feature-name
```

Open a Pull Request on GitHub with:
- Clear description of changes
- Screenshots (for visual changes)
- Test results

---

## Adding Model Support

To add support for a new MoE architecture:

1. **Check if auto-detection works:**
   ```bash
   moe-atlas trace "Test" --model your-model-id
   ```
   If it works, great! If not:

2. **Update `tracer.py`:**
   - Add model-specific hook logic in the profiler
   - Handle fused vs non-fused expert blocks
   - Add to the supported models list in README.md

3. **Test:**
   ```bash
   python -m pytest tests/test_tracer.py -v -k your_model
   ```

4. **Document:**
   - Add to the model compatibility table in README.md
   - Note any special requirements (VRAM, quantization)

---

## Contributing Traces

The atlas is most useful when the community pools routing traces. The JSONL trace file is
**appendable** (one self-contained trace per line), and contributions are deduplicated and
validated automatically.

1. **Trace** one or more models locally:
   ```bash
   moe-atlas trace "your prompt" --model <moe-model-id>
   ```

2. **Append** your traces to a shared file (duplicates are skipped automatically):
   ```bash
   moe-atlas export --append community.jsonl
   ```
   To combine files from several people: `moe-atlas merge a.jsonl b.jsonl -o community.jsonl`.

3. **Validate** before opening a pull request:
   ```bash
   moe-atlas validate community.jsonl
   ```
   `validate` exits non-zero if any record is malformed or fails the `Trace` schema, so it
   doubles as a CI gate for trace pull requests.

4. **Open a pull request** adding or updating the shared file. Keep prompts non-sensitive
   and respect each model's license — traces store the input text alongside routing data.

Traces are identified by their routing data (model + input tokens + expert activations), so
re-tracing the same prompt never creates duplicates in the shared file.

---

## Improving the Visualizer

The visualizer is a standalone HTML file with embedded Three.js:

```
src/moe_routing_atlas/visualizer/index.html
```

### Making Changes

1. Edit `index.html` directly
2. Test by running `moe-atlas serve` and opening the visualizer
3. Take before/after screenshots for PRs

### Visualizer Guidelines

- Every visual element must represent actual trace data
- No purely decorative geometry (removed: attention diamonds, norm rings, skip connections)
- Active elements should glow; inactive elements should be dim
- Keep performance smooth (≤50,000 elements per trace)
- Support both dark and light themes

---

## Testing

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=moe_routing_atlas --cov-report=html

# Specific test
pytest tests/test_tracer.py -v

# Integration tests (need GPU)
pytest tests/test_integration.py -v
```

### Writing Tests

Place tests in `tests/` with the naming pattern `test_*.py`:

```python
import pytest
from moe_routing_atlas.schema import Trace


def test_trace_creation():
    trace = Trace(text="Hello", activations=[])
    assert trace.num_tokens == 1
```

---

## Code Style

We use:
- **Black** for formatting (line length 100)
- **Ruff** for linting
- **MyPy** for type checking

### IDE Setup

**VS Code:**
```json
{
  "python.formatting.provider": "black",
  "python.linting.ruffEnabled": true,
  "python.linting.mypyEnabled": true,
  "editor.formatOnSave": true
}
```

---

## Reporting Bugs

When reporting bugs, please include:

1. **Environment:**
   ```bash
   moe-atlas --version
   python --version
   pip show transformers torch
   ```

2. **Steps to reproduce**

3. **Expected vs actual behavior**

4. **Error messages** (full traceback)

5. **Screenshots** (for visual issues)

---

## Feature Requests

For feature requests, open a GitHub Issue with:

- Clear description of the feature
- Use case / motivation
- Mockups (for visual features)
- Whether you'd like to implement it

---

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers
- Focus on the code, not the person
- Assume good intent

---

## Questions?

- GitHub Discussions: General questions
- GitHub Issues: Bug reports and feature requests
- Discord: [invite link] (real-time chat)

Thank you for contributing to MoE Atlas!

# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-06-15

### Added
- `orjson` for fast JSON serialization (new required dependency).

### Changed
- Significantly reduced end-to-end trace pipeline latency (capture, store, query, export) while keeping routing activation counts and gate-weight values byte-for-byte identical:
  - `capture`: router decisions are extracted with bulk tensor `.tolist()` instead of per-element `.item()`, and MoE block discovery is memoized per model.
  - `schema.TraceActivation`: now a slotted `@dataclass` instead of a Pydantic model.
  - `backend.create_trace`: activations are stored with a single batched `executemany` instead of per-row inserts.
  - `backend`: trace reads and exports use tuple-row SQLite cursors instead of `sqlite3.Row`.
  - `exporter`: JSON/JSONL export builds rows from a single grouped SQLite scan; pandas is now used only for Parquet output.
- New SQLite databases use a leaner activations schema (single `trace_id` index, no `AUTOINCREMENT` surrogate key); existing databases remain readable.
- The backend now reports the installed package version instead of a hard-coded value.

## [0.2.1] - 2026-06-06

### Security
- `GET /traces` no longer returns full input text or token strings (preview only)
- Visualizer trace chips no longer show raw text in tooltips

## [0.2.0] - 2026-06-06

### Added
- Shared `capture.py` module for MoE hook registration and model loading
- Backend integration tests and exporter normalization tests
- `pydantic-settings` dependency and `.pre-commit-config.yaml`
- GitHub Actions CI and release workflows
- `CHANGELOG.md` and versioned GitHub Releases

### Changed
- API responses now expose `id`, `trace_id`, `token_strs`, and `token_ids` for visualizer compatibility
- Backend defaults to `127.0.0.1` with restricted CORS origins
- `trust_remote_code` defaults to `false` (opt in via `--trust-remote-code`)
- Visualizer uses relative API URLs and safe DOM rendering for token info
- Export/import pipeline maps DB fields to valid `Trace` payloads

### Fixed
- MoE hooks now target router blocks only (not individual expert submodules)
- CLI no longer crashes on missing `trace_id` after backend upload
- Missing traces return HTTP 404 instead of 200 with error JSON
- SQLite trace inserts wrapped in transactions with WAL and foreign keys
- MPS/CPU device handling in tracers
- GPU memory leak when reloading traces in the visualizer

### Security
- Request body size limits on trace uploads
- Pydantic validation caps on text and activation list size
- Content-Security-Policy headers for the visualizer

## [0.1.0] - 2026-06-05

### Added
- Initial MoE Routing Atlas toolkit
- CLI (`serve`, `trace`, `batch`, `viz`, `export`, `import-traces`)
- FastAPI backend with SQLite storage
- Three.js 3D visualizer
- Batch and single-text tracers for HuggingFace MoE models

[Unreleased]: https://github.com/DJLougen/moe-routing-atlas/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/DJLougen/moe-routing-atlas/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/DJLougen/moe-routing-atlas/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/DJLougen/moe-routing-atlas/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/DJLougen/moe-routing-atlas/releases/tag/v0.1.0
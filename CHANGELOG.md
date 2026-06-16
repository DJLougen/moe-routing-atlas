# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-06-16

### Added
- **Auto domain categorization.** Tag every trace with a subject domain so routing data can be sliced by field — making it easy to study which experts specialize where:
  - `moe-atlas categorize traces.jsonl -o tagged.jsonl -e ENDPOINT` adds a `domain` label to each record in a trace file.
  - `moe-atlas categorize --db -e ENDPOINT [--min-trace-id N]` adds and populates a `domain` column on the local database, optionally scoped to only newly added traces.
  - New `categorizer` module — `make_llm_classifier`, `classify_texts`, `categorize_records`, `categorize_file`, `categorize_db`, `normalize_label`, and a coarse 15-entry `DEFAULT_TAXONOMY`. The classifier is any OpenAI-compatible chat endpoint (configurable via `-e` or `MOE_ATLAS_CLASSIFIER_ENDPOINT`) and is injectable, so the library and its tests never require a live model.
  - `Trace.domain` field; exported and imported traces now carry their domain label.
- Categorization only *adds* labels — activation counts and gate weights are never read or modified, and the routing fingerprint (the dedup key) is unaffected.

## [0.4.0] - 2026-06-16

### Added
- **Appendable community trace files.** New tooling turns the JSONL trace format into a shared, append-only dataset that many people can grow safely:
  - `moe-atlas merge FILES... -o OUT` combines trace files into one, validating and (by default) deduplicating records. The output may be one of the inputs, so a canonical community file can grow in place; the file is written atomically.
  - `moe-atlas export --append FILE` appends the local database's traces to a shared JSONL file, skipping any already present.
  - `moe-atlas validate FILES...` reports valid/duplicate/invalid records and exits non-zero on any invalid record (usable as a CI gate for trace pull requests).
  - Library functions `trace_fingerprint`, `append_traces`, `merge_trace_files`, `validate_trace_file`, and `collect_traces` in `exporter`.
- Traces are deduplicated by their routing data (model identity, input tokens, and expert activations) — independent of activation ordering, display name, timestamp, or backend-assigned id.

### Changed
- Malformed or schema-invalid records are skipped and counted during merge/append/validate, so a single bad contribution cannot corrupt a shared file.

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

[Unreleased]: https://github.com/DJLougen/moe-routing-atlas/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/DJLougen/moe-routing-atlas/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/DJLougen/moe-routing-atlas/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/DJLougen/moe-routing-atlas/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/DJLougen/moe-routing-atlas/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/DJLougen/moe-routing-atlas/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/DJLougen/moe-routing-atlas/releases/tag/v0.1.0
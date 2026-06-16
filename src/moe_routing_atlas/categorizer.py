"""Auto domain-categorization for routing traces.

Classifies each trace's input text into a coarse subject domain using any
OpenAI-compatible chat endpoint (a local llama.cpp / vLLM server, for example).
The classifier is injectable, so the library and its tests never require a live
model. Categorization only *adds* a ``domain`` label; routing activations are
never read or modified, so trace data stays byte-for-byte correct.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
import orjson

from .exporter import _read_trace_records

# A batch classifier maps a sequence of texts to one domain label each, in order.
ClassifyBatch = Callable[[Sequence[str]], list[str]]

DEFAULT_TAXONOMY: tuple[str, ...] = (
    "Physics & Astronomy",
    "Chemistry & Materials",
    "Biology & Life Sciences",
    "Medicine & Health",
    "Mathematics & Statistics",
    "Computer Science & Technology",
    "Earth & Environmental Science",
    "Economics & Finance",
    "Law & Politics",
    "History & Archaeology",
    "Social Sciences",
    "Philosophy & Ethics",
    "Arts & Humanities",
    "Engineering & Applied",
    "Other",
)

_FALLBACK = "Other"

# Short or partial model answers mapped to a canonical category. Matched on word
# boundaries so "art" does not fire inside "earth", etc.
_KEYWORD_DOMAINS: dict[str, str] = {
    "physics": "Physics & Astronomy",
    "astronomy": "Physics & Astronomy",
    "cosmology": "Physics & Astronomy",
    "astrophysics": "Physics & Astronomy",
    "chemistry": "Chemistry & Materials",
    "materials": "Chemistry & Materials",
    "biology": "Biology & Life Sciences",
    "genetics": "Biology & Life Sciences",
    "ecology": "Biology & Life Sciences",
    "botany": "Biology & Life Sciences",
    "zoology": "Biology & Life Sciences",
    "neuroscience": "Biology & Life Sciences",
    "medicine": "Medicine & Health",
    "health": "Medicine & Health",
    "pharmacology": "Medicine & Health",
    "nutrition": "Medicine & Health",
    "mathematics": "Mathematics & Statistics",
    "math": "Mathematics & Statistics",
    "statistics": "Mathematics & Statistics",
    "computer science": "Computer Science & Technology",
    "machine learning": "Computer Science & Technology",
    "robotics": "Computer Science & Technology",
    "cybersecurity": "Computer Science & Technology",
    "cryptography": "Computer Science & Technology",
    "technology": "Computer Science & Technology",
    "geology": "Earth & Environmental Science",
    "oceanography": "Earth & Environmental Science",
    "meteorology": "Earth & Environmental Science",
    "climate": "Earth & Environmental Science",
    "geography": "Earth & Environmental Science",
    "economics": "Economics & Finance",
    "finance": "Economics & Finance",
    "law": "Law & Politics",
    "politics": "Law & Politics",
    "history": "History & Archaeology",
    "archaeology": "History & Archaeology",
    "psychology": "Social Sciences",
    "sociology": "Social Sciences",
    "anthropology": "Social Sciences",
    "linguistics": "Social Sciences",
    "philosophy": "Philosophy & Ethics",
    "ethics": "Philosophy & Ethics",
    "music": "Arts & Humanities",
    "art": "Arts & Humanities",
    "literature": "Arts & Humanities",
    "poetry": "Arts & Humanities",
    "engineering": "Engineering & Applied",
    "aviation": "Engineering & Applied",
}

_LINE_RE = re.compile(r"\s*(\d+)\s*[:.)]\s*(.+)")


def normalize_label(label: str, taxonomy: Sequence[str] = DEFAULT_TAXONOMY) -> str:
    """Map a raw model label to a canonical taxonomy entry.

    Resolution order: exact (case-insensitive) match, then a partial match
    against a category name (handles answers like ``"Physics"`` for
    ``"Physics & Astronomy"``), then a word-boundary keyword fallback. Anything
    unrecognized becomes ``Other``.
    """
    canon = {t.lower(): t for t in taxonomy}
    fallback = _FALLBACK if _FALLBACK in taxonomy else taxonomy[-1]
    s = label.strip().strip('".').strip().lower()
    if not s:
        return fallback
    if s in canon:
        return canon[s]
    for low, original in canon.items():
        if low in s or (len(s) > 3 and s in low):
            return original
    for keyword, domain in _KEYWORD_DOMAINS.items():
        if domain in taxonomy and re.search(rf"\b{re.escape(keyword)}\b", s):
            return domain
    return fallback


def build_prompt(texts: Sequence[str], taxonomy: Sequence[str] = DEFAULT_TAXONOMY) -> str:
    """Build a one-shot classification prompt for a batch of texts."""
    categories = "\n".join(f"- {t}" for t in taxonomy)
    numbered = "\n".join(f"{i + 1}: {t}" for i, t in enumerate(texts))
    return (
        "Classify each numbered sentence into exactly ONE category from this list "
        "(reply with the exact category name):\n"
        f"{categories}\n\n"
        'Output exactly one line per sentence in the format "N: Category" and nothing else.\n\n'
        f"Sentences:\n{numbered}"
    )


def parse_label_lines(
    content: str, n: int, taxonomy: Sequence[str] = DEFAULT_TAXONOMY
) -> list[str]:
    """Parse an ``N: Category`` response into exactly ``n`` normalized labels.

    Missing or unparseable entries default to ``Other``, so a malformed line can
    never drop a trace or misalign labels with their texts.
    """
    labels = [_FALLBACK] * n
    for line in content.splitlines():
        match = _LINE_RE.match(line)
        if not match:
            continue
        idx = int(match.group(1)) - 1
        if 0 <= idx < n:
            labels[idx] = normalize_label(match.group(2), taxonomy)
    return labels


def make_llm_classifier(
    endpoint: str,
    *,
    model: str | None = None,
    taxonomy: Sequence[str] = DEFAULT_TAXONOMY,
    enable_thinking: bool = False,
    temperature: float = 0.0,
    seed: int = 0,
    timeout: float = 300.0,
) -> ClassifyBatch:
    """Build a batch classifier backed by an OpenAI-compatible chat endpoint.

    ``endpoint`` may be a base URL (``http://host:port`` or ``.../v1``) or a full
    ``.../chat/completions`` URL. Classification is deterministic by default
    (temperature 0, fixed seed). For "thinking" models, reasoning is disabled via
    ``chat_template_kwargs`` so the reply is just the labels. A failed request
    yields ``Other`` for the batch rather than raising, so one flaky call cannot
    abort a large categorization run.
    """
    url = endpoint.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url if url.endswith("/v1") else f"{url}/v1"
        url = f"{url}/chat/completions"

    def classify_batch(texts: Sequence[str]) -> list[str]:
        if not texts:
            return []
        body: dict = {
            "messages": [{"role": "user", "content": build_prompt(texts, taxonomy)}],
            "max_tokens": 24 * len(texts) + 32,
            "temperature": temperature,
            "top_p": 1.0,
            "seed": seed,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if model:
            body["model"] = model
        try:
            response = httpx.post(url, json=body, timeout=timeout)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"].get("content") or ""
        except (httpx.HTTPError, KeyError, ValueError, IndexError):
            return [_FALLBACK] * len(texts)
        return parse_label_lines(content, len(texts), taxonomy)

    return classify_batch


def classify_texts(
    texts: Sequence[str], classify_batch: ClassifyBatch, *, batch_size: int = 12
) -> list[str]:
    """Classify every text, batching calls to ``classify_batch`` and preserving order."""
    labels: list[str] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        out = classify_batch(batch)
        if len(out) != len(batch):  # defensive: keep labels aligned with texts
            out = (list(out) + [_FALLBACK] * len(batch))[: len(batch)]
        labels.extend(out)
    return labels


@dataclass(slots=True)
class CategorizeResult:
    """Outcome of categorizing a set of traces."""

    total: int
    categorized: int
    distribution: dict[str, int]


def _classify_unique(
    texts: Iterable[str], classify_batch: ClassifyBatch, batch_size: int
) -> dict[str, str]:
    """Classify the distinct non-empty texts once; return a ``{text: domain}`` map."""
    unique = sorted({t for t in texts if t})
    labels = classify_texts(unique, classify_batch, batch_size=batch_size)
    return dict(zip(unique, labels, strict=False))


def _summarize(domains: Iterable[str]) -> CategorizeResult:
    distribution: dict[str, int] = {}
    total = categorized = 0
    for domain in domains:
        total += 1
        if domain and domain != _FALLBACK:
            categorized += 1
        distribution[domain or ""] = distribution.get(domain or "", 0) + 1
    return CategorizeResult(total=total, categorized=categorized, distribution=distribution)


def categorize_records(
    records: list[dict], classify_batch: ClassifyBatch, *, batch_size: int = 12
) -> CategorizeResult:
    """Add a ``domain`` field to each record in place. Routing data is untouched."""
    domain_by_text = _classify_unique(
        (r.get("text", "") for r in records), classify_batch, batch_size
    )
    domains: list[str] = []
    for record in records:
        domain = domain_by_text.get(record.get("text", ""), "")
        record["domain"] = domain
        domains.append(domain)
    return _summarize(domains)


def categorize_file(
    input_path: str | Path,
    output_path: str | Path,
    classify_batch: ClassifyBatch,
    *,
    batch_size: int = 12,
) -> CategorizeResult:
    """Read a trace file, add domain labels, and write a tagged JSONL file."""
    records, _ = _read_trace_records(Path(input_path))
    result = categorize_records(records, classify_batch, batch_size=batch_size)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as handle:
        for record in records:
            handle.write(orjson.dumps(record))
            handle.write(b"\n")
    return result


def categorize_db(
    db_path: str | Path,
    classify_batch: ClassifyBatch,
    *,
    batch_size: int = 12,
    min_trace_id: int | None = None,
) -> CategorizeResult:
    """Add and populate a ``domain`` column on the ``traces`` table.

    Classifies each distinct trace text and writes the label back per row.
    ``min_trace_id`` scopes the update to ``trace_id > min_trace_id`` (e.g. to
    backfill only newly added traces). The ``domain`` column is added
    idempotently; the activations table is never modified.
    """
    conn = sqlite3.connect(db_path)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(traces)")}
        if "domain" not in columns:
            conn.execute("ALTER TABLE traces ADD COLUMN domain TEXT")
        query = "SELECT trace_id, text FROM traces"
        params: tuple = ()
        if min_trace_id is not None:
            query += " WHERE trace_id > ?"
            params = (min_trace_id,)
        rows = conn.execute(query, params).fetchall()
        domain_by_text = _classify_unique((text for _, text in rows), classify_batch, batch_size)
        updates = [(domain_by_text.get(text or "", ""), trace_id) for trace_id, text in rows]
        conn.executemany("UPDATE traces SET domain = ? WHERE trace_id = ?", updates)
        conn.commit()
    finally:
        conn.close()
    return _summarize(domain for domain, _ in updates)

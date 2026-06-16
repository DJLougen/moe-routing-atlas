"""Tests for the auto domain-categorizer.

The LLM is never contacted: a deterministic offline ``classify_batch`` stub
classifies by keyword, and the one endpoint-parsing test monkeypatches
``httpx.post``. This keeps the suite fast, deterministic, and CI-safe.
"""

from __future__ import annotations

import sqlite3

import orjson

from moe_routing_atlas.categorizer import (
    build_prompt,
    categorize_db,
    categorize_file,
    categorize_records,
    classify_texts,
    normalize_label,
    parse_label_lines,
)


def _stub_classify(texts):
    """Offline classifier: keyword -> domain, no network."""
    out = []
    for text in texts:
        low = text.lower()
        if "photon" in low or "galaxy" in low:
            out.append("Physics & Astronomy")
        elif "gene" in low or "cell" in low:
            out.append("Biology & Life Sciences")
        else:
            out.append("Other")
    return out


def test_normalize_label_exact_partial_keyword_and_fallback():
    assert normalize_label("Physics & Astronomy") == "Physics & Astronomy"
    assert normalize_label("  Biology & Life Sciences.") == "Biology & Life Sciences"
    assert normalize_label("Physics") == "Physics & Astronomy"  # partial of a category
    assert normalize_label("astrophysics") == "Physics & Astronomy"  # keyword
    assert normalize_label("completely unrelated noise") == "Other"
    assert normalize_label("") == "Other"


def test_normalize_label_keyword_respects_word_boundaries():
    # "art" must not fire inside "earth"; the category resolves by its own name.
    assert normalize_label("Earth & Environmental Science") == "Earth & Environmental Science"
    assert normalize_label("visual art") == "Arts & Humanities"


def test_parse_label_lines_aligns_and_defaults_missing():
    content = "1: Physics & Astronomy\n2: biology\n"  # third entry absent
    assert parse_label_lines(content, 3) == [
        "Physics & Astronomy",
        "Biology & Life Sciences",
        "Other",
    ]


def test_parse_label_lines_ignores_junk_and_out_of_range():
    content = "preamble junk\n2: Mathematics\n9: Physics\n1: Chemistry"
    assert parse_label_lines(content, 2) == ["Chemistry & Materials", "Mathematics & Statistics"]


def test_build_prompt_numbers_every_text():
    prompt = build_prompt(["alpha", "beta"])
    assert "1: alpha" in prompt and "2: beta" in prompt
    assert "Physics & Astronomy" in prompt  # taxonomy listed


def test_classify_texts_batches_preserve_order_and_count():
    texts = [f"t{i}" for i in range(25)]
    sizes = []

    def stub(batch):
        sizes.append(len(batch))
        return ["Other"] * len(batch)

    labels = classify_texts(texts, stub, batch_size=10)
    assert len(labels) == 25
    assert sizes == [10, 10, 5]


def test_classify_texts_realigns_short_response():
    labels = classify_texts(["a", "b", "c"], lambda batch: ["Other"], batch_size=12)
    assert labels == ["Other", "Other", "Other"]  # padded back to 3


def test_categorize_records_adds_domain_without_touching_routing():
    records = [
        {
            "text": "A photon is a quantum of light.",
            "activations": [{"layer": 0, "token_idx": 0, "expert_idx": 3, "gate_weight": 0.5}],
        },
        {
            "text": "A gene encodes a protein.",
            "activations": [{"layer": 1, "token_idx": 0, "expert_idx": 7, "gate_weight": 0.25}],
        },
        {"text": "An unremarkable note.", "activations": []},
    ]
    before = [orjson.dumps(r["activations"]) for r in records]

    result = categorize_records(records, _stub_classify)

    assert [r["domain"] for r in records] == [
        "Physics & Astronomy",
        "Biology & Life Sciences",
        "Other",
    ]
    assert [orjson.dumps(r["activations"]) for r in records] == before  # routing untouched
    assert result.total == 3
    assert result.categorized == 2
    assert result.distribution["Physics & Astronomy"] == 1
    assert result.distribution["Other"] == 1


def test_categorize_file_roundtrip(tmp_path):
    src = tmp_path / "in.jsonl"
    src.write_bytes(
        orjson.dumps({"text": "A galaxy holds billions of stars.", "activations": []})
        + b"\n"
        + orjson.dumps({"text": "A cell is the unit of life.", "activations": []})
        + b"\n"
    )
    out = tmp_path / "nested" / "out.jsonl"

    result = categorize_file(src, out, _stub_classify)

    rows = [orjson.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert [r["domain"] for r in rows] == ["Physics & Astronomy", "Biology & Life Sciences"]
    assert result.categorized == 2


def test_categorize_db_adds_column_updates_and_scopes(tmp_path):
    db = tmp_path / "atlas.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE traces (trace_id INTEGER PRIMARY KEY, text TEXT)")
    conn.executemany(
        "INSERT INTO traces (trace_id, text) VALUES (?, ?)",
        [(1, "Old galaxy trace."), (2, "A photon carries energy."), (3, "A gene mutates.")],
    )
    conn.commit()
    conn.close()

    result = categorize_db(db, _stub_classify, min_trace_id=1)  # exclude trace_id 1

    conn = sqlite3.connect(db)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(traces)")}
    rows = dict(conn.execute("SELECT trace_id, domain FROM traces"))
    conn.close()
    assert "domain" in cols
    assert rows[1] is None  # scoped out by min_trace_id
    assert rows[2] == "Physics & Astronomy"
    assert rows[3] == "Biology & Life Sciences"
    assert result.total == 2


def test_categorize_db_column_add_is_idempotent(tmp_path):
    db = tmp_path / "atlas.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE traces (trace_id INTEGER PRIMARY KEY, text TEXT, domain TEXT)")
    conn.execute("INSERT INTO traces VALUES (1, 'A photon.', NULL)")
    conn.commit()
    conn.close()

    result = categorize_db(db, _stub_classify)  # domain column already exists

    conn = sqlite3.connect(db)
    assert dict(conn.execute("SELECT trace_id, domain FROM traces"))[1] == "Physics & Astronomy"
    conn.close()
    assert result.total == 1


def test_make_llm_classifier_parses_endpoint_response(monkeypatch):
    import moe_routing_atlas.categorizer as cat

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "1: physics\n2: gene biology"}}]}

    captured: dict = {}

    def fake_post(url, json, timeout):  # noqa: A002 - mirrors httpx.post signature
        captured["url"] = url
        captured["body"] = json
        return _Resp()

    monkeypatch.setattr(cat.httpx, "post", fake_post)

    classify = cat.make_llm_classifier("http://localhost:8899", model="qwen")
    labels = classify(["What is a photon?", "What is a gene?"])

    assert labels == ["Physics & Astronomy", "Biology & Life Sciences"]
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["body"]["model"] == "qwen"
    assert captured["body"]["temperature"] == 0.0
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_make_llm_classifier_full_url_preserved(monkeypatch):
    import moe_routing_atlas.categorizer as cat

    seen: dict = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "1: Other"}}]}

    monkeypatch.setattr(
        cat.httpx, "post", lambda url, json, timeout: seen.update(url=url) or _Resp()
    )
    cat.make_llm_classifier("http://host:1234/v1/chat/completions")(["x"])
    assert seen["url"] == "http://host:1234/v1/chat/completions"


def test_make_llm_classifier_returns_other_on_http_error(monkeypatch):
    import httpx

    import moe_routing_atlas.categorizer as cat

    def boom(url, json, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(cat.httpx, "post", boom)
    assert cat.make_llm_classifier("http://x")(["a", "b"]) == ["Other", "Other"]


def test_cli_categorize_file(tmp_path, monkeypatch):
    from click.testing import CliRunner

    import moe_routing_atlas.categorizer as cat
    from moe_routing_atlas.cli import cli

    monkeypatch.setattr(cat, "make_llm_classifier", lambda *a, **k: _stub_classify)
    src = tmp_path / "in.jsonl"
    src.write_bytes(orjson.dumps({"text": "A galaxy is vast.", "activations": []}) + b"\n")
    out = tmp_path / "out.jsonl"

    result = CliRunner().invoke(cli, ["categorize", str(src), "-o", str(out), "-e", "http://x"])

    assert result.exit_code == 0, result.output
    rows = [orjson.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert rows[0]["domain"] == "Physics & Astronomy"


def test_cli_categorize_requires_endpoint(tmp_path):
    from click.testing import CliRunner

    from moe_routing_atlas.cli import cli

    src = tmp_path / "in.jsonl"
    src.write_bytes(orjson.dumps({"text": "x", "activations": []}) + b"\n")
    # No -e and no MOE_ATLAS_CLASSIFIER_ENDPOINT -> usage error, non-zero exit.
    result = CliRunner(env={"MOE_ATLAS_CLASSIFIER_ENDPOINT": ""}).invoke(
        cli, ["categorize", str(src), "-o", str(tmp_path / "o.jsonl")]
    )
    assert result.exit_code != 0

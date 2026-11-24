import json
import math
import sys
from pathlib import Path

import pytest

import analyze
import audit
import fetch


def test_tokenize_removes_stopwords_short_tokens_and_handles_unicode():
    tokens = analyze.tokenize("The café and Bayes’ theorem explain naïve ideas in IT")
    assert "the" not in tokens
    assert "and" not in tokens
    assert "café" in tokens
    assert "bayes" in tokens
    assert "theorem" in tokens
    assert "naïve" in tokens
    assert "it" not in tokens


def test_add_bigrams_keeps_unigrams_and_adjacent_pairs():
    tokens = ["battery", "cycle", "life"]
    assert analyze.add_bigrams(tokens) == [
        "battery",
        "cycle",
        "life",
        "battery_cycle",
        "cycle_life",
    ]


def test_tfidf_matches_explicit_formula():
    docs = {
        "a": ["rare", "rare", "shared"],
        "b": ["other", "shared"],
    }
    ranked = {name: dict(pairs) for name, pairs in analyze.tf_idf(docs).items()}
    rare_idf = 1.0 + math.log(2.0)
    assert ranked["a"]["rare"] == pytest.approx((2 / 3) * rare_idf)
    assert ranked["a"]["shared"] == pytest.approx(1 / 3)  # df=N -> idf=1
    assert ranked["b"]["other"] == pytest.approx(0.5 * rare_idf)


def test_tfidf_ties_have_deterministic_alphabetical_order():
    ranked = analyze.tf_idf({"doc": ["zeta", "alpha"]})["doc"]
    assert [term for term, _ in ranked] == ["alpha", "zeta"]


def test_tfidf_rejects_empty_corpus_and_empty_documents():
    with pytest.raises(ValueError, match="At least one"):
        analyze.tf_idf({})
    with pytest.raises(ValueError, match="no usable tokens"):
        analyze.tf_idf({"empty": []})


def test_cosine_similarity_is_symmetric_and_self_similarity_is_one():
    docs = {
        "a": ["battery", "electrode", "charge", "storage"],
        "b": ["battery", "electrode", "capacity", "storage"],
        "c": ["football", "match", "pressing", "tactics"],
    }
    sim = analyze.cosine_similarity(docs)
    assert sim["a"]["a"] == pytest.approx(1.0)
    assert sim["a"]["b"] == pytest.approx(sim["b"]["a"])
    assert sim["a"]["b"] > sim["a"]["c"]
    assert all(0.0 <= value <= 1.0 + 1e-12 for row in sim.values() for value in row.values())


def test_bigram_order_audit_recovers_order_information():
    result = audit.bigram_order_audit()
    assert result["unigram_similarity"] == pytest.approx(1.0)
    assert result["unigram_plus_bigram_similarity"] == pytest.approx(0.3435051014342031)
    assert result["unigram_plus_bigram_similarity"] < result["unigram_similarity"]


def test_corpus_dependence_changes_similarity_without_changing_target_docs():
    result = audit.corpus_dependence_audit(background_documents=20)
    assert result["base_similarity"] == pytest.approx(0.25861529161577274)
    assert result["augmented_similarity"] == pytest.approx(0.05638050141068369)
    assert result["relative_change"] == pytest.approx(-0.7820, abs=5e-4)


def test_uniformly_duplicating_entire_corpus_preserves_idf_geometry():
    base = {"a": ["shared", "alpha"], "b": ["shared", "beta"]}
    duplicated = {
        "a1": base["a"],
        "b1": base["b"],
        "a2": base["a"],
        "b2": base["b"],
    }
    base_sim = analyze.cosine_similarity(base)["a"]["b"]
    dup_sim = analyze.cosine_similarity(duplicated)["a1"]["b1"]
    assert dup_sim == pytest.approx(base_sim)


def test_load_documents_and_bigrams(tmp_path):
    (tmp_path / "a.txt").write_text("Battery cycle life", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Catalyst reaction rate", encoding="utf-8")
    docs = analyze.load_documents(tmp_path, include_bigrams=True)
    assert "cycle_life" in docs["a"]
    assert "reaction_rate" in docs["b"]


def test_load_documents_rejects_missing_or_empty_after_cleaning(tmp_path):
    with pytest.raises(FileNotFoundError):
        analyze.load_documents(tmp_path)
    (tmp_path / "empty.txt").write_text("the and a to", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        analyze.load_documents(tmp_path)


def test_analyze_cli_exposes_bigrams_and_similarity(tmp_path, monkeypatch, capsys):
    (tmp_path / "a.txt").write_text("new york city", encoding="utf-8")
    (tmp_path / "b.txt").write_text("york new city", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyze.py", "--dir", str(tmp_path), "--bigrams", "--similarity", "--top", "10"],
    )
    analyze.main()
    output = capsys.readouterr().out
    assert "new_york" in output
    assert "Pairwise cosine similarity" in output
    assert "unigrams + adjacent bigrams" in output


def test_sklearn_validation_matches_explicit_convention_to_machine_precision():
    docs = analyze.load_documents(Path(__file__).resolve().parents[1] / "data" / "sample")
    result = audit.sklearn_validation(docs)
    assert result["max_abs_tfidf_weight_error"] < 1e-14
    assert result["max_abs_cosine_error"] < 1e-14


def test_api_url_encodes_path_sensitive_title_characters():
    assert fetch.api_url("Battery (electricity)").endswith("Battery_%28electricity%29")
    assert fetch.api_url("A/B").endswith("A%2FB")
    with pytest.raises(ValueError):
        fetch.api_url("   ")


def test_slug_rejects_title_with_no_filename_characters():
    assert fetch.slug("Battery (electricity)") == "battery_electricity"
    with pytest.raises(ValueError):
        fetch.slug("%%%")


def test_fetch_summary_uses_contact_user_agent_and_returns_provenance(monkeypatch):
    calls = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "title": "Battery (electricity)",
                "extract": "A battery stores energy.",
                "pageid": 123,
                "timestamp": "2026-01-02T03:04:05Z",
                "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Battery_(electricity)"}},
            }

    def fake_get(url, headers, timeout):
        calls.update(url=url, headers=headers, timeout=timeout)
        return Response()

    monkeypatch.setattr(fetch.requests, "get", fake_get)
    record = fetch.fetch_summary("Battery (electricity)")
    assert "ahmadnabileblagh.com" in calls["headers"]["User-Agent"]
    assert calls["timeout"] == 30
    assert record["extract"] == "A battery stores energy."
    assert record["pageid"] == 123
    assert record["source_url"].startswith("https://en.wikipedia.org/")
    assert "%28electricity%29" in record["api_url"]


def test_fetch_summary_rejects_empty_extract(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"title": "Empty", "extract": ""}

    monkeypatch.setattr(fetch.requests, "get", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="no usable summary"):
        fetch.fetch_summary("Empty")


def test_fetch_main_writes_text_and_manifest_without_sleep_after_last(tmp_path, monkeypatch):
    records = {
        "Alpha": {
            "requested_title": "Alpha",
            "resolved_title": "Alpha",
            "extract": "Alpha extract",
            "pageid": 1,
            "revision_timestamp": "2026-01-01T00:00:00Z",
            "source_url": "https://example.org/alpha",
            "api_url": "https://api.example/alpha",
        },
        "Beta": {
            "requested_title": "Beta",
            "resolved_title": "Beta",
            "extract": "Beta extract",
            "pageid": 2,
            "revision_timestamp": "2026-01-02T00:00:00Z",
            "source_url": "https://example.org/beta",
            "api_url": "https://api.example/beta",
        },
    }
    monkeypatch.setattr(fetch, "fetch_summary", lambda title, user_agent: dict(records[title]))
    sleeps = []
    monkeypatch.setattr(fetch.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(
        sys,
        "argv",
        ["fetch.py", "Alpha", "Beta", "--out-dir", str(tmp_path), "--delay", "0.25"],
    )
    fetch.main()

    assert (tmp_path / "alpha.txt").read_text(encoding="utf-8") == "Alpha extract"
    assert (tmp_path / "beta.txt").read_text(encoding="utf-8") == "Beta extract"
    manifest = json.loads((tmp_path / "_manifest.json").read_text(encoding="utf-8"))
    assert [entry["file"] for entry in manifest] == ["alpha.txt", "beta.txt"]
    assert all("retrieved_utc" in entry for entry in manifest)
    assert sleeps == [0.25]


def test_fetch_main_refuses_slug_collisions(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["fetch.py", "C++", "C#", "--out-dir", str(tmp_path)],
    )
    with pytest.raises(SystemExit, match="collide"):
        fetch.main()

"""Deterministic validation/audit experiments for the text-analysis project."""
from __future__ import annotations

import json
from pathlib import Path

from analyze import add_bigrams, cosine_similarity, load_documents, tf_idf

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "audit_summary.json"


def sklearn_validation(docs: dict[str, list[str]]) -> dict[str, float]:
    """Compare the explicit convention with a deliberately matched sklearn setup."""
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
    except ImportError as exc:  # pragma: no cover - exercised when optional validation deps absent
        raise RuntimeError(
            "Install requirements-validation.txt to run the sklearn validation"
        ) from exc

    names = sorted(docs)
    token_lists = [docs[name] for name in names]
    vectorizer = TfidfVectorizer(
        analyzer=lambda value: value,
        lowercase=False,
        smooth_idf=False,
        norm=None,
        sublinear_tf=False,
    )
    matrix = vectorizer.fit_transform(token_lists).toarray()
    terms = vectorizer.get_feature_names_out()

    ours = {name: dict(pairs) for name, pairs in tf_idf(docs).items()}
    max_weight_error = 0.0
    for row_index, name in enumerate(names):
        # sklearn gives count * idf; this project explicitly uses count/len(d) * idf.
        scaled = matrix[row_index] / len(docs[name])
        for column_index, term in enumerate(terms):
            max_weight_error = max(
                max_weight_error,
                abs(scaled[column_index] - ours[name].get(term, 0.0)),
            )

    reference_cosine = sklearn_cosine(matrix)
    ours_cosine = cosine_similarity(docs)
    max_cosine_error = 0.0
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            max_cosine_error = max(
                max_cosine_error,
                abs(reference_cosine[i, j] - ours_cosine[a][b]),
            )

    return {
        "max_abs_tfidf_weight_error": float(max_weight_error),
        "max_abs_cosine_error": float(max_cosine_error),
    }


def corpus_dependence_audit(background_documents: int = 20) -> dict[str, float]:
    """Show that TF-IDF cosine similarity is a property of a corpus, not just a pair."""
    base = {
        "a": ["shared", "alpha"],
        "b": ["shared", "beta"],
    }
    augmented = dict(base)
    for index in range(background_documents):
        augmented[f"background_{index}"] = ["shared", f"backgroundterm{index}"]

    base_similarity = cosine_similarity(base)["a"]["b"]
    augmented_similarity = cosine_similarity(augmented)["a"]["b"]
    return {
        "background_documents": background_documents,
        "base_similarity": base_similarity,
        "augmented_similarity": augmented_similarity,
        "relative_change": augmented_similarity / base_similarity - 1.0,
    }


def bigram_order_audit() -> dict[str, float]:
    """Same unigram bag, different order: bigrams recover order information."""
    unigram_docs = {
        "a": ["new", "york", "city"],
        "b": ["york", "new", "city"],
    }
    bigram_docs = {name: add_bigrams(tokens) for name, tokens in unigram_docs.items()}
    return {
        "unigram_similarity": cosine_similarity(unigram_docs)["a"]["b"],
        "unigram_plus_bigram_similarity": cosine_similarity(bigram_docs)["a"]["b"],
    }


def sample_summary(docs: dict[str, list[str]]) -> dict:
    ranked = tf_idf(docs)
    similarity = cosine_similarity(docs)
    return {
        "documents": len(docs),
        "tokens": sum(map(len, docs.values())),
        "top_term_by_document": {name: pairs[0][0] for name, pairs in ranked.items()},
        "cosine_similarity": similarity,
    }


def main() -> None:
    docs = load_documents(ROOT / "data" / "sample")
    report = {
        "tfidf_convention": "relative TF * (1 + ln(N/df)); no L2 normalisation",
        "sample": sample_summary(docs),
        "sklearn_validation": sklearn_validation(docs),
        "corpus_dependence": corpus_dependence_audit(),
        "bigram_order": bigram_order_audit(),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()

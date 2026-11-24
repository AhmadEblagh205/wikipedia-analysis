"""From-scratch text analysis for a small document collection.

The core TF-IDF convention is deliberately explicit:

    tf(t, d)  = count(t, d) / number_of_features_in_d
    idf(t)    = 1 + ln(N / df(t))
    tfidf     = tf * idf

No L2 normalisation is applied to the stored TF-IDF weights. Cosine
similarity normalises vectors when comparing them.

Usage:
    python src/analyze.py
    python src/analyze.py --bigrams --similarity
    python src/analyze.py --dir data/raw --similarity
"""
from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STOPWORDS = set(
    """a an and are as at be by for from has have in is it its
of on or that the this to was were which with their they when where why into through
how not over between during each such than then these those while""".split()
)

# Unicode letters, with optional internal ASCII/curly apostrophe groups.
# Digits/underscores are deliberately excluded from lexical tokens.
TOKEN_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lower-case text and keep Unicode word tokens after light filtering.

    This is intentionally a small educational tokenizer rather than a claim
    to production linguistic preprocessing. Curly apostrophes are normalised
    and possessive apostrophes do not survive as stray punctuation.
    """
    normalised = text.lower().replace("’", "'")
    tokens = [t.strip("'") for t in TOKEN_RE.findall(normalised)]
    return [t for t in tokens if t and t not in STOPWORDS and len(t) > 2]


def add_bigrams(tokens: list[str]) -> list[str]:
    """Return unigrams plus adjacent bigrams joined with an underscore."""
    return tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]


def _validate_docs(docs: dict[str, list[str]]) -> None:
    if not docs:
        raise ValueError("At least one document is required")
    empty = [name for name, tokens in docs.items() if not tokens]
    if empty:
        raise ValueError(
            "Documents contain no usable tokens after cleaning: " + ", ".join(sorted(empty))
        )


def tf_idf(docs: dict[str, list[str]]) -> dict[str, list[tuple[str, float]]]:
    """Compute relative-TF × unsmoothed log-IDF weights from scratch.

    idf(t) = 1 + ln(N / df(t)). Terms present in every document therefore
    retain weight 1 rather than being forced to zero.
    """
    _validate_docs(docs)
    n_docs = len(docs)
    document_frequency: Counter[str] = Counter()
    for tokens in docs.values():
        document_frequency.update(set(tokens))

    idf = {
        term: 1.0 + math.log(n_docs / frequency)
        for term, frequency in document_frequency.items()
    }

    out: dict[str, list[tuple[str, float]]] = {}
    for name, tokens in docs.items():
        counts = Counter(tokens)
        total = len(tokens)
        scores = {
            term: (count / total) * idf[term]
            for term, count in counts.items()
        }
        # Alphabetical secondary key makes exact-score ties deterministic.
        out[name] = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return out


def cosine_similarity(docs: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    """Pairwise cosine similarity on the same hand-built TF-IDF vectors.

    Full precision is returned. Formatting/rounding belongs in presentation,
    not in the analytical function.
    """
    ranked = tf_idf(docs)
    vectors = {name: dict(pairs) for name, pairs in ranked.items()}
    names = sorted(vectors)
    norms = {
        name: math.sqrt(sum(weight * weight for weight in vector.values()))
        for name, vector in vectors.items()
    }

    out = {name: {} for name in names}
    for a in names:
        for b in names:
            numerator = sum(
                weight * vectors[b].get(term, 0.0)
                for term, weight in vectors[a].items()
            )
            denominator = norms[a] * norms[b]
            value = numerator / denominator if denominator else 0.0
            # Floating-point round-off can otherwise produce 1 + epsilon on the diagonal.
            out[a][b] = max(-1.0, min(1.0, value))
    return out


def load_documents(directory: str | Path, include_bigrams: bool = False) -> dict[str, list[str]]:
    """Load UTF-8 ``*.txt`` documents from a directory."""
    paths = sorted(Path(directory).glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"No .txt files in {directory}")

    docs: dict[str, list[str]] = {}
    for path in paths:
        tokens = tokenize(path.read_text(encoding="utf-8"))
        if include_bigrams:
            tokens = add_bigrams(tokens)
        docs[path.stem] = tokens
    _validate_docs(docs)
    return docs


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _print_similarity(similarity: dict[str, dict[str, float]]) -> None:
    names = sorted(similarity)
    width = max(12, max(map(len, names), default=0) + 2)
    print("\nPairwise cosine similarity on the same TF-IDF feature space:")
    print("".ljust(width) + "".join(name[:10].rjust(11) for name in names))
    for a in names:
        row = "".join(f"{similarity[a][b]:11.4f}" for b in names)
        print(a[: width - 2].ljust(width) + row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=ROOT / "data" / "sample")
    parser.add_argument("--top", type=_positive_int, default=8)
    parser.add_argument(
        "--bigrams",
        action="store_true",
        help="add adjacent bigram features alongside unigrams",
    )
    parser.add_argument(
        "--similarity",
        action="store_true",
        help="print pairwise cosine similarity on the TF-IDF vectors",
    )
    args = parser.parse_args()

    try:
        docs = load_documents(args.dir, include_bigrams=args.bigrams)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(
        f"{len(docs)} documents, {sum(map(len, docs.values()))} "
        f"features after cleaning"
        + (" (unigrams + adjacent bigrams)" if args.bigrams else " (unigrams)")
        + "\n"
    )

    all_tokens: Counter[str] = Counter()
    for tokens in docs.values():
        all_tokens.update(tokens)
    print("Most frequent features across the collection:")
    for token, count in all_tokens.most_common(args.top):
        print(f"  {token:<24} {count}")

    print("\nHighest TF-IDF terms per document:")
    for name, ranked in tf_idf(docs).items():
        terms = ", ".join(term for term, _ in ranked[: args.top])
        print(f"  {name:<14} {terms}")

    if args.similarity:
        _print_similarity(cosine_similarity(docs))


if __name__ == "__main__":
    main()

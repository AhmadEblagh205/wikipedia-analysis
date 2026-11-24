# TF-IDF, Corpus Dependence & Wikipedia Summary Analysis

[![CI](https://github.com/AhmadEblagh205/wikipedia-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmadEblagh205/wikipedia-analysis/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A small text-analysis laboratory built to make **TF-IDF and cosine similarity inspectable rather than magical**. The core weighting is implemented from scratch, bigrams expose local word-order information, and deterministic audits show an important property: TF-IDF similarity is **corpus-relative**, not an intrinsic semantic distance between two documents.

The default offline demo uses three short **hand-written sample texts**. An optional fetcher downloads English-Wikipedia **summary extracts** into ignored local files; the repository does not pretend that its bundled sample is Wikipedia data or that three summaries characterise Wikipedia as a whole.

## What changed in the audit

The original rebuild had sound basic TF-IDF mechanics, but several claims/interfaces needed tightening:

- the README said bigrams and cosine similarity were delivered, although the CLI never exposed either feature;
- the exact TF-IDF convention was not stated, making comparisons with library defaults ambiguous;
- cosine values were rounded inside the analytical function rather than only for display;
- the tokenizer was ASCII-only and mishandled accented words / apostrophe variants;
- the live fetch User-Agent was descriptive but did not provide the contact route Wikimedia currently asks automated clients to include;
- the fetcher called short summary extracts “articles” and saved no retrieval provenance;
- the bundled demo's actual top terms were not the ones implied by the old prose.

V0.3 makes those boundaries explicit and adds controlled validation rather than stronger unsupported claims.

## TF-IDF convention

For term `t` in document `d`:

\[
TF(t,d)=\frac{c_{t,d}}{|d|}, \qquad
IDF(t)=1+\ln\frac{N}{df(t)}, \qquad
TFIDF=TF\times IDF.
\]

There is no L2 normalisation of the stored weights. This is a valid convention, but not the only convention called TF-IDF. Smoothing, tokenisation and normalisation choices differ across libraries.

`src/audit.py` configures scikit-learn to match this definition and checks the result rather than comparing against unrelated defaults.

## A known-truth lesson: similarity depends on the corpus

Hold two target documents fixed:

```text
a = shared alpha
b = shared beta
```

With only those two documents, TF-IDF cosine similarity is **0.2586**.

Now add 20 background documents that all contain `shared`. The target documents do not change, but their similarity falls to **0.0564**. The common token has become less informative relative to the enlarged corpus, while `alpha` and `beta` have become rarer.

That is a roughly **78% decrease in measured similarity without changing either target text**.

The point is not that TF-IDF is defective. It is that its definition of “distinctive” is conditional on the reference corpus.

## Why bigrams can matter

A unigram bag cannot distinguish these two sequences:

```text
new york city
york new city
```

They contain the same unigram multiset, so unigram TF-IDF cosine similarity is **1.0000**. When adjacent bigrams are added, the similarity falls to about **0.3435** because local order is now represented.

This is a mechanism demonstration, not evidence that bigrams universally improve a downstream model.

## Offline usage

```bash
pip install -r requirements.txt
python src/analyze.py
python src/analyze.py --similarity
python src/analyze.py --bigrams --similarity
```

The bundled sample currently contains **3 hand-written documents / 135 cleaned unigram tokens**. Under the explicit audited convention, the highest-scoring term is:

- battery sample: `energy`;
- catalysis sample: `catalyst`;
- probability sample: `explains`, `numbers`, `probability` and `theorem` are tied at the top; the deterministic alphabetical tie-break prints `explains` first.

Notice that a top TF-IDF term need not be unique to one document, and exact-score ties are not substantive rank differences: term frequency and document frequency trade off against each other.

## Optional Wikipedia summary fetching

```bash
python src/fetch.py "Battery (electricity)" Catalysis Probability
python src/analyze.py --dir data/raw --bigrams --similarity
```

`fetch.py`:

- fetches **summary extracts**, not full article bodies;
- uses a descriptive User-Agent with a website contact route;
- sends requests serially with a configurable conservative delay;
- URL-encodes titles;
- fails if no usable extract is returned;
- writes `_manifest.json` with page/source metadata and retrieval time;
- keeps downloaded content under ignored `data/raw/`.

If fetched Wikimedia content is republished, the applicable content-licence and attribution requirements still apply.

## Validation audit

For the optional scikit-learn cross-check:

```bash
pip install -r requirements-validation.txt
python src/audit.py
```

The audit records:

1. matched TF-IDF weights versus an equivalently configured `TfidfVectorizer`;
2. matched cosine similarities;
3. the controlled corpus-dependence experiment;
4. the unigram-versus-bigram word-order experiment.

Results are written to `reports/audit_summary.json`.

## Tests

```bash
pytest -q
```

Tests cover tokenisation, deterministic TF-IDF, formula identities, cosine properties, corpus dependence, bigram order sensitivity, CLI-visible feature behaviour, fetch URL/provenance logic and the optional scikit-learn equivalence check in CI.

## Interpretation boundary

Cosine similarity here means **lexical similarity in the selected TF-IDF feature space**. It is not a general semantic-similarity score, a relevance judgement or evidence of retrieval quality. For ranked search and labelled retrieval evaluation, see the separate inverted-index/search project in the wider portfolio.

More detail: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Licence

Code is MIT licensed; see [LICENSE](LICENSE). Fetched Wikimedia content is separate upstream content and is not included in the repository.

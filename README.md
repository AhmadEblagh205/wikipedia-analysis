# TF-IDF, Corpus Dependence & Wikipedia Summary Analysis

[![CI](https://github.com/AhmadEblagh205/wikipedia-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmadEblagh205/wikipedia-analysis/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

I built this small text-analysis project to work through TF-IDF and cosine similarity by implementing the weighting myself. It also adds bigrams, and a few checks that show TF-IDF similarity depends on the corpus it is computed in. The default demo uses three short hand-written sample texts, which are not Wikipedia data. An optional fetcher downloads English-Wikipedia summary extracts into local files that git ignores.

## What it does

- Computes TF-IDF weights from scratch, with TF as count over document length and IDF as 1 + ln(N / df), and no L2 normalisation.
- Prints the top terms per document, optionally with bigrams (`--bigrams`) and cosine similarity between documents (`--similarity`).
- Fetches Wikipedia summary extracts, not full articles, and writes a manifest with the source and retrieval time.
- Checks the weights and similarities against an equivalently configured scikit-learn `TfidfVectorizer` in `src/audit.py`.

## Results

These come from `reports/audit_summary.json` and use small controlled examples, not Wikipedia.

- Two documents, `shared alpha` and `shared beta`, have a TF-IDF cosine similarity of 0.2586. After I add 20 background documents that all contain `shared`, the similarity is 0.0564, about 78% lower, although neither document changed.
- The sequences `new york city` and `york new city` have unigram similarity 1.0000. With adjacent bigrams added it falls to about 0.3435.
- My weights match scikit-learn's to within 1.4e-17, and the cosine similarities to within 8.9e-16.
- The bundled sample has 3 documents and 135 cleaned unigram tokens. The top terms are `energy`, `catalyst` and, after a four-way tie broken alphabetically, `explains`.

These show how the method behaves. They do not show that bigrams improve a downstream model.

## How to run

```bash
pip install -r requirements.txt
python src/analyze.py
python src/analyze.py --similarity
python src/analyze.py --bigrams --similarity

# Optional: fetch Wikipedia summary extracts
python src/fetch.py "Battery (electricity)" Catalysis Probability
python src/analyze.py --dir data/raw --bigrams --similarity

# Optional: scikit-learn cross-check, writes reports/audit_summary.json
pip install -r requirements-validation.txt
python src/audit.py

pytest -q
```

The tests cover tokenisation, TF-IDF determinism and formula identities, cosine properties, corpus dependence, bigram order sensitivity, the CLI options, fetch URL and provenance logic, and the scikit-learn check in CI.

## Notes

Cosine similarity here is lexical similarity in the TF-IDF feature space, not a semantic score. Fetched Wikimedia content stays out of the repository, and if you republish it the content licence and attribution requirements still apply. The TF-IDF definition, the fixes I made in the audit and the fetcher details are in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Licence

Code is MIT licensed, see [LICENSE](LICENSE).

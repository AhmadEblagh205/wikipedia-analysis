# Changelog

## 0.3.0 — defensibility audit

- Wired bigrams and cosine similarity into the actual CLI.
- Documented the exact relative-TF / unsmoothed-IDF convention.
- Added deterministic tie-breaking and full-precision analytical cosine values.
- Upgraded tokenisation for Unicode letters and apostrophe variants.
- Added a controlled corpus-dependence audit and a bigram word-order audit.
- Added apples-to-apples validation against a matched scikit-learn TF-IDF configuration.
- Tightened the Wikipedia ingestion claim from full articles to summary extracts.
- Added compliant contact-route User-Agent, URL encoding, extract validation and a retrieval manifest.
- Added methodology/claim-boundary documentation and expanded tests.

## 0.2.0 — update round 4

Added cosine similarity between documents on the hand-built TF-IDF vectors and a bigram helper so multiword concepts can become visible.

## 0.1.0

Initial release.

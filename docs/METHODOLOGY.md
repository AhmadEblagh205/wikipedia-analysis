# Methodology and claim boundaries

## What the repository analyses

The default offline corpus is three short **hand-written sample texts** in `data/sample/`. They are not Wikipedia articles. `src/fetch.py` is an optional ingestion path that downloads short English-Wikipedia **summary extracts** from Wikimedia's REST endpoint into ignored local `data/raw/` files.

That distinction matters for provenance: results from the bundled demo are controlled educational examples, not measurements of Wikipedia as a platform.

## Tokenisation

The tokenizer lowercases text, recognises Unicode letter sequences, normalises curly apostrophes, removes a small explicit stopword set and removes tokens shorter than three characters. This is intentionally transparent and educational. It is not a claim to language-independent linguistic preprocessing, stemming or lemmatisation.

`--bigrams` adds adjacent token pairs alongside the unigrams. It does not discover phrases statistically; it simply makes local word order visible in the feature space.

## Exact TF-IDF convention

For term `t` in document `d`:

\[
\operatorname{tf}(t,d)=\frac{c_{t,d}}{|d|}
\]

where `|d|` is the number of features in that document under the selected unigram/bigram representation.

For a collection of `N` documents:

\[
\operatorname{idf}(t)=1+\ln\frac{N}{df(t)}
\]

and

\[
\operatorname{tfidf}(t,d)=\operatorname{tf}(t,d)\operatorname{idf}(t).
\]

There is no L2 normalisation of the stored weights. The `+1` means a term present in every document receives IDF 1 rather than zero.

This is one valid TF-IDF convention, not *the* unique definition of TF-IDF. Libraries differ in smoothing, sublinear TF, tokenisation and vector normalisation.

## scikit-learn validation

`src/audit.py` configures `TfidfVectorizer` to match this repository's unsmoothed IDF and disables L2 normalisation. Scikit-learn returns `count * idf`; dividing each row by its token count matches this repository's relative-TF convention. The audit compares both term weights and cosine similarities.

This validation is deliberately apples-to-apples. Comparing against scikit-learn defaults would test a different estimator configuration.

## Cosine similarity

For TF-IDF vectors `x` and `y`:

\[
\cos(x,y)=\frac{x^Ty}{\lVert x\rVert_2\lVert y\rVert_2}.
\]

The analytical function returns full precision. Rounding happens only when results are printed.

Cosine similarity here is **lexical similarity in the chosen TF-IDF feature space**. It is not a general semantic-similarity score and should not be interpreted as human relevance without labels.

## Why the corpus matters

IDF is estimated from the collection. Therefore the TF-IDF representation of a document can change when other documents are added, even if the document itself does not change.

The deterministic audit makes two target documents fixed:

- `a = [shared, alpha]`
- `b = [shared, beta]`

With only those two documents, cosine similarity is about 0.2586. Add 20 background documents that all contain `shared`, and the target texts themselves remain unchanged but their cosine similarity falls to about 0.0564. The shared token has become globally uninformative while the unique terms become rarer relative to the enlarged corpus.

The lesson is not that TF-IDF is unstable or wrong. The lesson is that its notion of distinctiveness is **corpus-relative**.

## What bigrams add

A bag of unigrams discards word order. In the audit:

- `new york city`
- `york new city`

have exactly the same unigram multiset, so unigram TF-IDF cosine similarity is 1.0. Adding adjacent bigrams reduces similarity to about 0.3435 because `new_york`, `york_city`, `york_new` and `new_city` encode different local order.

This is a controlled mechanism check, not evidence that bigrams always improve downstream prediction or retrieval.

## Wikimedia fetching and provenance

The fetcher uses serial requests, a descriptive User-Agent containing a website contact route, URL-encodes article titles, validates that a non-empty extract was returned and records local provenance in `_manifest.json`.

The default one-second delay is a conservative project choice, not a claimed Wikimedia numerical rate limit. Operators should obey Wikimedia's current API/User-Agent policies and any throttling instructions they receive.

Fetched Wikipedia content is not committed to this repository. Anyone redistributing fetched content is responsible for the applicable Wikimedia content licence and attribution requirements.

## Claim boundary

This project demonstrates:

- transparent tokenisation and n-gram feature construction;
- from-scratch TF-IDF mechanics;
- vector-space cosine similarity;
- corpus dependence of IDF;
- controlled validation against an equivalently configured library implementation;
- disciplined API/provenance handling.

It does **not** establish:

- semantic understanding of text;
- superiority of bigrams over unigrams on a labelled task;
- search/recommendation quality;
- empirical properties of Wikipedia as a whole;
- causal or behavioural conclusions from article text.

## Notes moved from the README

Fixes made in the audit of the original rebuild:

- The old README said bigrams and cosine similarity were delivered, but the CLI exposed neither. Both are now available through `--bigrams` and `--similarity`.
- The exact TF-IDF convention was not stated. It is now TF = count / document length and IDF = 1 + ln(N / df), with no L2 normalisation, and `src/audit.py` configures scikit-learn to match it.
- Cosine values were rounded inside the analytical function. They are now rounded only for display.
- The tokeniser was ASCII-only and mishandled accented words and apostrophe variants.
- The fetch User-Agent did not include the contact route Wikimedia asks automated clients to give. The fetcher also called short summary extracts "articles" and saved no retrieval provenance. It now sends a descriptive User-Agent with a website contact route, requests serially with a configurable delay, URL-encodes titles, fails if no usable extract is returned and writes `_manifest.json` with source metadata and retrieval time.
- The bundled demo's top terms did not match the old prose. On the 3 hand-written documents (135 cleaned unigram tokens) the top terms are `energy` (battery), `catalyst` (catalysis) and a four-way tie in the probability sample (`explains`, `numbers`, `probability`, `theorem`), where the alphabetical tie-break prints `explains`. A top TF-IDF term need not be unique to one document, and exact ties are not substantive rank differences.
- Cosine similarity here is lexical similarity in the chosen TF-IDF feature space. It is not a semantic-similarity score or evidence of retrieval quality. For ranked search with labelled evaluation, see the separate grep-search-engine project.

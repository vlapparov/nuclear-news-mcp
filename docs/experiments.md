# Experiments, evaluation method and benchmarks

Supporting detail for the [README](../README.md): what was tried during development, how
retrieval was measured, and the full timings. Raw outputs of every run are in
[`results/`](../results/).

**Machines.** Timings come from two machines, always labelled:

| | Author's Intel Mac (target) | Apple M3 Max (development) |
|---|---|---|
| Processor | Intel Core i9-9880H @ 2.30 GHz, 16 logical CPUs | Apple M3 Max, 14 cores |
| Memory, OS | 16 GB, macOS 15.7.9 (x86_64) | 36 GB, macOS 26.5.2 (arm64) |
| Python · torch · transformers · sentence-transformers | 3.12.12 · 2.2.2 · 4.57.6 · 5.7.0 | 3.12.4 · 2.14.1 · 5.20.1 · 6.1.0 |
| torch CPU threads | 8 | 10 |
| Raw files | [`benchmark-intel.json`](../results/benchmark-intel.json), [`build-manifest-intel.json`](../results/build-manifest-intel.json) | [`benchmark-arm64.json`](../results/benchmark-arm64.json), [`build-manifest-default-index.json`](../results/build-manifest-default-index.json) |

Timings of the two machines are not interchangeable (different hardware and library versions).
Retrieval quality of the default index is identical on both: all 36 values match. The chunk-size
and model comparisons were run on the M3 only.

## Contents

- [Evaluation method](#evaluation-method)
- [1. Chunking](#1-chunking)
- [2. Embedding model](#2-embedding-model)
- [3. Keyword, semantic or hybrid](#3-keyword-semantic-or-hybrid)
- [4. Intel Mac notes](#4-intel-mac-notes)
- [5. Performance](#5-performance)
- [6. Reproducing the results](#6-reproducing-the-results)

---

## Evaluation method

No labelled questions exist for this dataset, so retrieval is measured with a **known-item
benchmark** ([`scripts/eval_retrieval.py`](../scripts/eval_retrieval.py)). Every query is built
from, or written for, one *target* article, and the question is where that article ranks. The
sample is fixed (random seed 0), so every configuration sees the same queries.

| Query set | n | How a query is made | What it tests |
|---|---|---|---|
| headline | 200 | the target article's own title | same-language, short query |
| partial sentence | 200 | a sentence from the middle third of the article, with every third word removed | same-language "half-remembered fact" |
| English → native | 16 | an English question written for a specific Polish, Czech, Slovak or Hungarian article (listed in the script) | cross-lingual search, the case of an English-speaking client |

Metrics over the top 10 results: **top-1** (target ranked first), **top-10** (target anywhere in
the top 10) and **MRR** (average of 1/rank, 0 when missing).

**How much a difference means.** With 200 queries, one standard error of a difference is at most
about 5 points, so differences above ~10 points are meaningful and a few points are not (no paired
significance test was run). With **16 queries, one query is 6.25 points** and the standard error
is up to about 18 points, so the English-set results are **indicative only**. This applies to
every table below.

**Limits.** Known-item search is a proxy. The headline and partial-sentence queries copy words from
the article, so they favour keyword search. None of this measures how well an LLM client chooses
tools or how faithful its answer is (see "Future work" in the README).

---

## 1. Chunking

**Goal.** Chunks are what gets searched and what the client sees as evidence: understandable on
their own, small enough to pinpoint a fact, comparable across four languages, traceable to their
article.

**Starting point (v0, not in this repository).** A hand-written splitter packed whole sentences into
passages of up to 800 characters, embedded with multilingual-e5-small. Hybrid search ranked the
target first for 56.5% of partial-sentence queries. That run used unrepaired article text, so it is
only indicative.

**Approach.** LangChain's `RecursiveCharacterTextSplitter` with a structure-first separator order.
Article bodies have almost no paragraph breaks (only Blikk_hun uses line breaks, plus 50 articles
elsewhere), so the usable structure is title → body → sentences → clauses:

```python
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
```

Sizes are counted in **tokens of the embedding model's own tokenizer**, so a chunk holds a similar
amount of text in every language. Each chunk gets its article title as a header
(`"<title>\n\n<chunk>"`) and stores `article_id`, `chunk_index` and character offsets.

**Results** ([`results/experiments.jsonl`](../results/experiments.jsonl), static model, hybrid unless stated):

| Chunk / overlap (tokens) | Chunks | Headline, semantic top-1 | Partial sentence, hybrid top-1 | Partial sentence, hybrid top-10 | English → native, hybrid top-10 |
|---|---|---|---|---|---|
| **128 / 16** (adopted) | 223,653 | **91.0%** | **80.5%** | **97.5%** | 50% (8 of 16) |
| 256 / 0 | 106,688 | 79.5% | 65.5% | 93.5% | 38% (6 of 16) |
| 256 / 32 | 109,710 | 79.0% | 66.5% | 94.5% | 44% (7 of 16) |
| 512 / 64 | 58,836 | 57.0% | 55.5% | 87.5% | 38% (6 of 16) |

**Conclusions.**

- **Smaller chunks win clearly:** 128/16 beats 256 and 512 on every set, by 14–25 points on
  partial-sentence top-1. Likely reason: the model averages its token vectors, so distinguishing
  words get diluted in long chunks.
- **Overlap** makes no measurable difference (about 1 point). It was kept at 16 tokens so a fact on a
  boundary appears whole in at least one chunk (not tested separately).
- **Not specific to one model:** e5 also scored higher at 128 than at 256 tokens (headline semantic
  top-1 93.0% → 96.5%, partial-sentence hybrid top-1 55.0% → 69.0%).
- **Adopted:** 128 tokens, 16 overlap. Chunks average 344 characters (max 659).
- **Semantic chunking** (splitting where the topic changes) was not tried: it costs an extra
  embedding pass and should be judged on a labelled question set first.

---

## 2. Embedding model

**Goal.** In v0, embedding the corpus with multilingual-e5-small was the slow step and ran slowly on
the author's Intel Mac (torch 2.2.2 is the last Intel-Mac version; GPU acceleration there could not
be tested). How much build time can a CPU-only build save, and at what cost in quality?

**What a "static" model is.** A transformer (e5, MiniLM) passes text through 12 attention layers, so
each word's vector depends on its context. A static model is a tokenizer plus a lookup table: a
text's vector is the average of its word pieces' rows. It ignores word order and context, but needs
no network, so it is fast on any CPU. The chosen one,
`static-similarity-mrl-multilingual-v1`, has 105,879 word pieces × 1,024 numbers, trained on
translation and paraphrase pairs in 50 languages including the four used here. It is
Matryoshka-trained, so the first 256 dimensions are used (4× less memory).

**Method.** Quality: one full build and evaluation per model, all at 128/16 chunks, transformers on
the Apple GPU (MPS). Speed: CPU throughput on the same 2,000 chunk texts for every model, after a
warm-up (5 runs on the Intel Mac, 3 on the M3 Max).

**Results.**

| Model (preset) | Type | CPU chunks/s (median): Intel / M3 | Embed full corpus on Intel CPU (extrapolated) | Headline semantic top-1 | Partial sent. hybrid top-1 | English → native semantic top-10 | English → native hybrid top-10 |
|---|---|---|---|---|---|---|---|
| static-similarity-mrl-multilingual-v1 (`fast`) | lookup table | ~9,500 / ~26,000 (noisy) | **~24 s** (real build: 27.6 s) | 91.0% | **80.5%** | 50% (8) | 50% (8) |
| paraphrase-multilingual-MiniLM-L12-v2 (`crosslingual`) | transformer, ≤128 tokens | 41.8 / 249 | ~89 min | 78.5% | 54.5% | **87.5% (14)** | **81% (13)** |
| multilingual-e5-small (v0's model) | transformer, ≤512 tokens | 34.7 / 213 | ~107 min | **96.5%** | 69.0% | 62.5% (10) | 50% (8) |

Notes: brackets are counts out of 16. "Extrapolated" is 223,653 chunks divided by the median rate; the
transformer indexes actually have 204,771 chunks (different tokenizers), so about 82 min (MiniLM)
and 98 min (e5). On the M3's GPU the real transformer embedding times were 222 s (MiniLM) and 256 s
(e5), one run each.

**Conclusions.**

- **Static** is roughly 110–250× faster than MiniLM on the Intel CPU and 40–105× faster on the M3. It
  is the best on same-language partial-sentence queries (80.5% vs 54.5% for MiniLM, clearly
  meaningful) and the weakest at matching English questions to non-English articles.
- **MiniLM** leads on the English set (13 vs 8 of 16 in hybrid): suggestive, not established.
- **e5** is best on headline-style semantic queries (96.5% vs 91.0%, modest), but hybrid keyword
  matching covers much of that case and e5 was the slowest. It was dropped as a preset;
  `scripts/experiment.py` still supports it.
- **Decision:** `fast` is the default. The whole default build on the Intel Mac takes about 4.2 min,
  against ~1.5 h for MiniLM's embedding step alone. `crosslingual` stays available
  (`news-mcp-ingest --embedding crosslingual`) for English-speaking clients where a GPU or a long
  build is acceptable.
- On the Intel Mac only the default model's quality was re-measured, and it was identical.

---

## 3. Keyword, semantic or hybrid

The same index (the delivered default) was searched in each mode with `scripts/eval_retrieval.py`.

| Query set | Keyword | Semantic | **Hybrid** (default) | Hybrid MRR |
|---|---|---|---|---|
| headline (n = 200) | 98.0% / 100% | 91.0% / 96.5% | **97.0% / 100%** | 0.983 |
| partial sentence (n = 200) | 93.5% / 99.0% | 62.0% / 83.0% | **80.5% / 97.5%** | 0.856 |
| English → native (n = 16) | 0% / 6.3% | 31.3% / 50.0% | **6.3% / 50.0%** | 0.190 |

(top-1 / top-10; from [`results/eval-default-index.json`](../results/eval-default-index.json), and
identical in [`results/eval-intel.json`](../results/eval-intel.json).)

- **Keyword search is the strongest single retriever on same-language queries**, because those
  queries reuse the article's own words.
- **Hybrid costs rank-1 accuracy there** (80.5% vs 93.5% on partial-sentence top-1): Reciprocal Rank
  Fusion weights the weaker semantic ranking equally. The target stays in the top 10.
- **Keyword fails on English → native** (1 of 16 in the top 10). That is what semantic search is
  for, and hybrid keeps 8 of 16.
- **Decision:** hybrid is the default, because an English-speaking client is the realistic case.
  `mode="keyword"` remains available. Fusion weights and the RRF constant were not tuned: doing so on
  this benchmark alone would overfit to queries that copy the article's words.

---

## 4. Intel Mac notes

The pinned Intel stack was checked on the Apple Silicon machine in a separate x86_64 environment
under Rosetta 2 (`UV_PROJECT_ENVIRONMENT=.venv-x86 uv sync --python cpython-3.12-macos-x86_64`),
then on the author's Intel Mac itself. Two problems were found and fixed:

1. **Install failure:** `cryptography` 49+ (needed by the MCP SDK) has no Intel-Mac wheel, so install
   tried to compile it with Rust. Fix: `cryptography<49` on Intel Macs only.
2. **Build crash:** sentence-transformers picks Apple's GPU (MPS) when available, and torch 2.2.2
   cannot run the static model's `EmbeddingBag` there. Intel Macs with a Metal GPU also report MPS,
   so this most likely affects them too (not verified). Fix: the `fast` preset always runs on the CPU.

With both fixes all 37 tests pass on the Intel stack, and build → serve → query works end to end. On
the Intel Mac the full index (223,653 chunks) gave exactly the same retrieval results as on Apple
Silicon, so the older libraries do not change the results. No test output from that machine is
stored in `results/`; timings taken under Rosetta are emulation and are not reported.

---

## 5. Performance

**Full build of the default index.** One run per machine: the chunking and embedding phases recorded
by `news-mcp-ingest` (dataset download and loading not included).

| Phase | Intel i9 | M3 Max |
|---|---|---|
| Chunking 26,483 articles | 223.5 s | 87.0 s |
| Embedding 223,653 chunks (`fast`, CPU) | 27.6 s | 12.5 s |
| **Total of the two** | **251.1 s (≈ 4.2 min)** | **99.5 s** |

**Benchmark** (`scripts/benchmark_speed.py`; 5 timed runs on Intel, 3 on the M3, after a warm-up;
embeddings on the CPU). Throughput is the median with min–max; latency is the median and p95 of
single searches (20 queries × 5 runs on Intel, 20 × 3 on the M3).

| Measurement | Intel i9 | M3 Max | M3 faster by |
|---|---|---|---|
| Chunking, as in `news-mcp-ingest` | 138 articles/s (123–139) | 302 articles/s (301–304) | 2.2× |
| Embedding, static (`fast`) | ~9,500 chunks/s (4,805–10,124) | ~26,000 chunks/s (9,753–26,086) | ~2.7× (noisy: runs < 0.5 s) |
| Embedding, MiniLM (`crosslingual`) | 41.8 chunks/s (41.0–42.4) | 249 chunks/s (247–252) | 6.0× |
| Embedding, e5-small | 34.7 chunks/s (34.1–35.7) | 213 chunks/s (212–215) | 6.2× |
| Hybrid search | 48.7 ms (p95 319 ms) | 21.8 ms (p95 43.8 ms) | 2.2× |
| Hybrid search with filters (Poland, from 2020) | 68.5 ms (p95 244 ms) | 32.3 ms (p95 47.3 ms) | 2.1× |
| Keyword search | 10.2 ms (p95 52.6 ms) | 4.4 ms (p95 26.1 ms) | 2.3× |
| Semantic search | 36.3 ms (p95 38.1 ms) | 16.5 ms (p95 16.9 ms) | 2.2× |
| `aggregate_articles`, country × year (100 groups) | 1,157 ms (1,134–2,220) | 599 ms (597–600) | 1.9× |

**What this establishes.**

- On the Intel Mac the default build takes about 4 minutes, and 89% of that is chunking. The next
  build-time gain is cheaper token counting, not a faster embedding model.
- A transformer model is impractical to build on this Intel CPU (~1.5 h), confirming the default.
- Interactive use is fast: typical searches take 10–70 ms on the Intel Mac.

**What it does not establish.**

- Why transformers are ~6× slower on the Intel Mac while everything else is ~2× slower: hardware and
  library versions differ, so the gap cannot be attributed to the CPU alone.
- Where the Intel hybrid-search latency tail comes from (p95 319 ms vs a 48.7 ms median). First-run
  slowdowns are visible elsewhere in the same file, but the cause was not investigated.
- How exact the extrapolations are: on the Intel Mac extrapolated chunking (192 s) was 14% below the
  real build (223.5 s). Treat extrapolated times as lower bounds. Build-to-build variation is not
  measured.

---

## 6. Reproducing the results

| Result | Command | Output | Time on the M3 Max |
|---|---|---|---|
| Retrieval quality of the default index | `uv run python scripts/eval_retrieval.py --out results/eval-default-index.json` | `results/eval-default-index.json` | ~1 min |
| Chunk-size and model comparisons | `uv run python scripts/experiment.py --model <fast\|crosslingual\|e5> --chunk-tokens N --overlap M` (the seven commands used are in the script's docstring) | `results/experiments.jsonl`, one line per run | ~2 min per static run; ~7 min per transformer run on the Apple GPU |
| Speed on your machine | `uv run python scripts/benchmark_speed.py --runs 3 --models fast crosslingual e5` (the Intel run used `--runs 5`) | `results/benchmark-<machine>.json` | 1.5 min |
| The brief's sample questions over MCP | `uv run news-mcp-serve`, then in another terminal `uv run python scripts/sample_questions.py` | printed | < 1 min |

Not reproducible from this repository alone: the **v0 baseline** numbers (the v0 code is not part of
it), the **Rosetta compatibility check** (needs an Apple Silicon Mac), and the **full-build
timings**, which come from one `news-mcp-ingest` run per machine (a new build writes its own timing
to `data/manifest.json`).

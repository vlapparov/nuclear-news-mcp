"""Measure speed on this machine and record the hardware and software it ran on.

Needs a built index (uv run news-mcp-ingest). Each measurement is repeated --runs times:
  1. chunking:   articles split per second (fast-preset tokenizer), on a fixed sample of articles
  2. embedding:  chunks embedded per second on the CPU, per model, on a fixed sample of chunks
  3. search:     latency in ms of single searches (hybrid / keyword / semantic, with and without filters)
  4. aggregate:  latency in ms of aggregate_articles grouped by country and year

    uv run python scripts/benchmark_speed.py --runs 3
    uv run python scripts/benchmark_speed.py --runs 3 --models fast crosslingual e5 --out results/benchmark-<name>.json
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime
from importlib import metadata

import torch
from experiment import MODELS
from news_mcp import config, server
from news_mcp.chunking import make_splitter, split_article, text_to_embed
from news_mcp.embeddings import SentenceEmbeddings

QUERIES = [
    "LNG world trade 2021", "Paks II construction permit", "Dukovany tender", "Chernobyl tourists",
    "radioactive waste storage", "Mochovce third unit", "Temelín expansion", "elektrownia jądrowa Pomorze",
    "atomfegyver Oroszország", "jadrové zbrane Bielorusko", "Fukushima evacuation", "uranium mining",
    "small modular reactors", "nuclear energy EU taxonomy", "Zaporizhzhia power plant shelling",
    "energetická bezpečnost plyn", "atomerőmű bővítés", "energetyka jądrowa koszty", "Iran nuclear deal",
    "nuclear power public opinion",
]


def environment() -> dict:
    def sysctl(key):
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
        except OSError:
            return None

    memory = sysctl("hw.memsize")
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "machine": platform.machine(),
        "processor": sysctl("machdep.cpu.brand_string") or platform.processor(),
        "cpu_count": os.cpu_count(),
        "memory_gb": round(int(memory) / 2**30, 1) if memory else None,
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch_threads": torch.get_num_threads(),
        "packages": {p: metadata.version(p) for p in
                     ["torch", "numpy", "transformers", "sentence-transformers", "langchain-text-splitters", "mcp"]},
    }


def summary(values: list) -> dict:
    return {
        "median": round(statistics.median(values), 3),
        "mean": round(statistics.mean(values), 3),
        "stdev": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
        "runs": [round(v, 3) for v in values],
    }


def time_it(function) -> float:
    started = time.perf_counter()
    function()
    return time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--sample", type=int, default=2000, help="chunks per embedding run")
    parser.add_argument("--articles", type=int, default=500, help="articles per chunking run")
    parser.add_argument("--models", nargs="+", default=["fast", "crosslingual"], choices=list(MODELS))
    parser.add_argument("--out", default=f"results/benchmark-{platform.machine()}.json")
    args = parser.parse_args()

    server.load_index(args.data_dir)
    store = server.store
    n_articles = store.query("SELECT COUNT(*) AS n FROM articles")[0]["n"]
    n_chunks = store.query("SELECT COUNT(*) AS n FROM chunks")[0]["n"]
    report = {"environment": environment(), "config": vars(args), "index": {"articles": n_articles, "chunks": n_chunks}}
    print(json.dumps(report["environment"], indent=2))

    # 1. chunking: same splitter settings as news-mcp-ingest, tokenizer of the fast preset
    fast = SentenceEmbeddings(MODELS["fast"]["model"], MODELS["fast"])
    step = max(1, n_articles // args.articles)
    articles = store.query(f"SELECT id, title, body FROM articles WHERE rowid % {step} = 0 LIMIT {args.articles}")
    splitter = make_splitter(fast.count_tokens)
    seconds = [time_it(lambda: [split_article(a, splitter) for a in articles]) for _ in range(args.runs)]
    rates = [len(articles) / s for s in seconds]
    report["chunking_articles_per_s"] = summary(rates)
    report["chunking_full_corpus_s_extrapolated"] = round(n_articles / statistics.median(rates), 1)

    # 2. embedding on the CPU: the same chunk texts for every model
    step = max(1, n_chunks // args.sample)
    rows = store.query(
        "SELECT articles.title, chunks.text FROM chunks JOIN articles ON articles.id = chunks.article_id "
        f"WHERE chunk_id % {step} = 0 LIMIT {args.sample}"
    )
    texts = [text_to_embed(r["title"], r["text"]) for r in rows]
    report["embedding"] = {}
    for name in args.models:
        model = SentenceEmbeddings(MODELS[name]["model"], {**MODELS[name], "device": "cpu"})
        model.model.encode(texts[:64])  # warm-up, not timed
        seconds = [time_it(lambda: model.model.encode(texts, batch_size=64)) for _ in range(args.runs)]
        rates = [len(texts) / s for s in seconds]
        report["embedding"][name] = {
            "chunks_per_s": summary(rates),
            "full_corpus_s_extrapolated": round(n_chunks / statistics.median(rates), 1),
        }

    # 3. search latency, one query at a time
    server.searcher.search("warm-up", {}, 10)
    report["search_ms"] = {}
    for label, mode, filters in [
        ("hybrid", "hybrid", {}),
        ("hybrid_filtered", "hybrid", {"country": ["Poland"], "date_from": "2020-01-01"}),
        ("keyword", "keyword", {}),
        ("semantic", "semantic", {}),
    ]:
        latencies = [1000 * time_it(lambda: server.searcher.search(q, filters, 10, mode))
                     for _ in range(args.runs) for q in QUERIES]
        report["search_ms"][label] = {**summary(latencies), "p95": round(sorted(latencies)[int(0.95 * len(latencies)) - 1], 3)}
        del report["search_ms"][label]["runs"]  # one value per query and run: too long to keep
        report["search_ms"][label]["n"] = len(latencies)

    # 4. aggregation latency
    latencies = [1000 * time_it(lambda: store.aggregate({"nuclear_related_only": True}, ["country", "year"]))
                 for _ in range(args.runs)]
    report["aggregate_country_year_ms"] = summary(latencies)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "environment"}, indent=2))
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()

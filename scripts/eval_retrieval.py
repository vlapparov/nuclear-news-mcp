"""Measure how often search finds the right article ("known-item" retrieval).

Three query sets, each with one known target article:
  title    - the article's own headline (same language)
  passage  - a sentence from the middle of the article with every third word removed
  english  - hand-written English questions about Polish/Czech/Slovak/Hungarian articles

    uv run python scripts/eval_retrieval.py [--data-dir data] [--n 200]
"""

import argparse
import json
import random
import re

from news_mcp import config, server

# English question -> id of the (non-English) article that answers it
ENGLISH_QUESTIONS = {
    "Which nationality was second among tourists visiting Chernobyl in 2019?": "30216",
    "Radioactive beef was sold in Japanese shops after Fukushima": "8968",
    "South Korean ship carrying 6,200 tonnes of fuel oil to North Korea": "14291",
    "Catholic charity raised 38,000 euros for victims of the Japan earthquake": "15656",
    "Could solar power in the Czech Republic match seven Temelin plants within seven years?": "2075",
    "Austria is worried about restarting the Bohunice V1 reactor": "16229",
    "Russia offers to store spent nuclear fuel from six countries": "26371",
    "Radiation-eating fungus from Chernobyl sent to the space station with SpaceX": "27619",
    "Japan announces a phase-out of nuclear power by 2040": "1777",
    "North Korea writes its nuclear weapons policy into the constitution": "22389",
    "Spain plans to close all its nuclear power plants": "918",
    "Is Brazil planning to build an atomic bomb?": "26603",
    "Soviet nuclear warheads were secretly stored at a base in Hungary": "32240",
    "Poland plans six nuclear reactors by 2040": "63",
    "Russian occupiers force captive staff to work at the Chernobyl power plant": "24057",
    "Public hearing on the Temelin expansion lasted late into the night": "5800",
}


def make_queries(n: int) -> dict:
    """{query set name: [(query, target article id), ...]}"""
    rng = random.Random(0)
    articles = server.store.query("SELECT id, title, body FROM articles WHERE length(body) > 1500")
    sample = rng.sample(articles, min(n, len(articles)))
    passage_queries = []
    for article in sample:
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", article["body"]) if 8 <= len(s.split()) <= 40]
        if sentences:
            words = rng.choice(sentences[len(sentences) // 3:] or sentences).split()
            passage_queries.append((" ".join(w for i, w in enumerate(words) if i % 3 != 2), article["id"]))
    known_ids = set(server.store.article_ids({}))  # a --limit build may not contain the English targets
    return {
        "title": [(a["title"], a["id"]) for a in sample],
        "passage": passage_queries,
        "english": [(q, i) for q, i in ENGLISH_QUESTIONS.items() if i in known_ids],
    }


def score(queries: list, mode: str) -> dict:
    """Rank of the target article for every query -> top-1, top-10 and MRR@10."""
    ranks = []
    for query, target in queries:
        ids = [r["article_id"] for r in server.searcher.search(query, {}, limit=10, mode=mode)]
        ranks.append(ids.index(target) + 1 if target in ids else None)
    return {
        "n": len(ranks),
        "top1": round(sum(r == 1 for r in ranks) / len(ranks), 3),
        "top10": round(sum(r is not None for r in ranks) / len(ranks), 3),
        "mrr": round(sum(1 / r for r in ranks if r) / len(ranks), 3),
    }


def evaluate(n: int = 200) -> list:
    """All query sets x all search modes, on the index loaded in `server`."""
    return [
        {"query_set": name, "mode": mode, **score(queries, mode)}
        for name, queries in make_queries(n).items()
        if queries  # a --limit build may contain none of the English targets
        for mode in ("keyword", "semantic", "hybrid")
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--out", help="also write the results to this JSON file")
    args = parser.parse_args()
    server.load_index(args.data_dir)

    rows = evaluate(args.n)
    print(f"{'query set':<10}{'mode':<10}{'n':>5}{'top-1':>8}{'top-10':>8}{'MRR':>8}")
    for r in rows:
        print(f"{r['query_set']:<10}{r['mode']:<10}{r['n']:>5}{r['top1']:>8.3f}{r['top10']:>8.3f}{r['mrr']:>8.3f}")
    if args.out:
        with open(args.data_dir + "/manifest.json") as f:
            manifest = json.load(f)
        with open(args.out, "w") as f:
            json.dump({"index": manifest, "results": rows}, f, indent=2)


if __name__ == "__main__":
    main()

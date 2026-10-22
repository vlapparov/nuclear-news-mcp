"""One retrieval experiment: build a full index with a given embedding model and chunk size
in a temporary folder, run the retrieval benchmark on it, and append one JSON line of results.

This is how the chunk-size and model comparisons in the README were produced:

    uv run python scripts/experiment.py --model fast --chunk-tokens 128 --overlap 16
    uv run python scripts/experiment.py --model fast --chunk-tokens 256 --overlap 0
    uv run python scripts/experiment.py --model fast --chunk-tokens 256 --overlap 32
    uv run python scripts/experiment.py --model fast --chunk-tokens 512 --overlap 64
    uv run python scripts/experiment.py --model crosslingual --chunk-tokens 128 --overlap 16
    uv run python scripts/experiment.py --model e5 --chunk-tokens 128 --overlap 16
    uv run python scripts/experiment.py --model e5 --chunk-tokens 256 --overlap 32
"""

import argparse
import json
import platform
import tempfile
from datetime import datetime

import eval_retrieval
from news_mcp import chunking, config, server
from news_mcp.embeddings import SentenceEmbeddings
from news_mcp.ingest import build_index, load_dataset_rows

# The two presets of the server, plus the model used in v0 (not offered as a preset any more).
MODELS = {
    **config.EMBEDDING_PRESETS,
    "e5": {
        "model": "intfloat/multilingual-e5-small",
        "truncate_dim": None,
        "device": None,
        "query_prefix": "query: ",  # E5 models are trained with these prefixes
        "passage_prefix": "passage: ",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), default="fast")
    parser.add_argument("--chunk-tokens", type=int, default=config.CHUNK_TOKENS)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP_TOKENS)
    parser.add_argument("--limit", type=int, help="random sample of N articles (quick check only)")
    parser.add_argument("--out", default="results/experiments.jsonl")
    args = parser.parse_args()

    chunking.CHUNK_TOKENS, chunking.CHUNK_OVERLAP_TOKENS = args.chunk_tokens, args.overlap
    preset = MODELS[args.model]
    model = SentenceEmbeddings(preset["model"], preset)
    rows = load_dataset_rows(args.limit)

    with tempfile.TemporaryDirectory() as data_dir:
        stats = build_index(rows, data_dir, model)
        server.load_index(data_dir, model=model)
        results = eval_retrieval.evaluate()

    line = {
        "model": args.model,
        "model_name": preset["model"],
        "chunk_tokens": args.chunk_tokens,
        "overlap_tokens": args.overlap,
        **stats,
        "embedding_device": str(model.model.device),
        "machine": platform.machine(),
        "date": datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }
    with open(args.out, "a") as f:
        f.write(json.dumps(line) + "\n")
    for r in results:
        print(f"{r['query_set']:<10}{r['mode']:<10}{r['n']:>5}{r['top1']:>8.3f}{r['top10']:>8.3f}{r['mrr']:>8.3f}")


if __name__ == "__main__":
    main()

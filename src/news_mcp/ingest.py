"""Build the search index once: dataset -> SQLite (articles, chunks, keyword index) + embeddings.

    uv run news-mcp-ingest                       # full dataset, "fast" embedding preset
    uv run news-mcp-ingest --limit 500           # quick test build
    uv run news-mcp-ingest --embedding crosslingual  # slower, better for English questions
"""

import argparse
import json
import os
import re
import sqlite3
import time
from datetime import datetime
from statistics import mean
from typing import List

import numpy as np

from news_mcp.chunking import make_splitter, split_article, text_to_embed
from news_mcp.config import DATA_DIR, DATASET, DATASET_SPLIT, EMBEDDING_PRESET, EMBEDDING_PRESETS
from news_mcp.embeddings import SentenceEmbeddings
from news_mcp.store import SCHEMA

MODELS = ["gemini", "deepseek", "gpt"]
RAW_SCORE_COLUMNS = [f"{m}_{k}" for m in MODELS for k in ("article_rating", "headline_sentiment", "headline_alarmistic")]


def load_dataset_rows(limit: int = None) -> List[dict]:
    """Download the dataset from Hugging Face (only network access of the project)."""
    from datasets import load_dataset

    dataset = load_dataset(DATASET, split=DATASET_SPLIT)
    if limit:
        dataset = dataset.shuffle(seed=0).select(range(limit))
    return list(dataset)


def repair_text(text: str) -> str:
    """Re-insert spaces lost at scraping time after a sentence end ("rokov.Projekt" -> "rokov. Projekt")."""
    return re.sub(r"([a-zà-ž][.!?])([A-ZÁÄČĎÉĚÍĹĽŇÓÔÖŐŘŔŠŤÚŮÜŰÝŽĄĆĘŁŃŚŹŻ])", r"\1 \2", (text or "").strip())


def clean_article(row: dict) -> dict:
    """
    Normalise one dataset row and add consensus scores of the three annotator models.
    - date "DD-MM-YYYY" -> "YYYY-MM-DD"; language "cz" -> "cs" (the standard code)
    - nuclear_related: at least 2 of 3 models gave article_rating > 0 (0 = "not about nuclear power")
    - nuclear_sentiment: mean of the non-zero article ratings (1..5)
    - headline_alarmism: 6 - mean(raw), because in the raw data 5 means "NOT alarmist"
    """
    date = datetime.strptime(row["date"], "%d-%m-%Y")
    ratings = [row[f"{m}_article_rating"] for m in MODELS if row[f"{m}_article_rating"] > 0]
    related = len(ratings) >= 2
    article = {
        "id": str(row["id"]),
        "country": row["country"],
        "language": "cs" if row["language"] == "cz" else row["language"],
        "source": row["source"],
        "date": date.strftime("%Y-%m-%d"),
        "year": date.year,
        "month": date.strftime("%Y-%m"),
        "url": row["url"],
        "title": (row["title"] or "").strip(),
        "body": repair_text(row["article body"]),
        "nuclear_related": int(related),
        "nuclear_sentiment": round(mean(ratings), 3) if related else None,
        "headline_sentiment": round(mean(row[f"{m}_headline_sentiment"] for m in MODELS), 3),
        "headline_alarmism": round(6 - mean(row[f"{m}_headline_alarmistic"] for m in MODELS), 3),
    }
    for column in RAW_SCORE_COLUMNS:
        article[column] = row[column]
    return article


def build_index(rows: List[dict], data_dir: str, model: SentenceEmbeddings) -> dict:
    """
    Write data_dir/news.sqlite and data_dir/embeddings.npy.
    :return: build statistics
    """
    os.makedirs(data_dir, exist_ok=True)
    db_path = os.path.join(data_dir, "news.sqlite")
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)

    splitter = make_splitter(model.count_tokens)
    texts_to_embed = []
    started = time.time()
    for row in rows:
        article = clean_article(row)
        columns = list(article)
        conn.execute(
            f"INSERT INTO articles ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
            [article[c] for c in columns],
        )
        for chunk in split_article(article, splitter):
            chunk_id = len(texts_to_embed)
            conn.execute(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                (chunk_id, chunk["article_id"], chunk["chunk_index"], chunk["start_char"], chunk["end_char"],
                 chunk["text"]),
            )
            conn.execute("INSERT INTO chunks_fts (rowid, title, text) VALUES (?, ?, ?)",
                         (chunk_id, article["title"], chunk["text"]))
            texts_to_embed.append(text_to_embed(article["title"], chunk["text"]))
    conn.commit()
    conn.close()
    chunking_seconds = time.time() - started

    print(f"Embedding {len(texts_to_embed)} chunks of {len(rows)} articles...")
    started = time.time()
    embeddings = model.encode_passages(texts_to_embed)
    np.save(os.path.join(data_dir, "embeddings.npy"), embeddings.astype(np.float16))
    return {
        "articles": len(rows),
        "chunks": len(texts_to_embed),
        "chunking_seconds": round(chunking_seconds, 1),
        "embedding_seconds": round(time.time() - started, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Build the search index for the news MCP server.")
    parser.add_argument("--limit", type=int, help="only use a random sample of N articles")
    parser.add_argument("--embedding", default=EMBEDDING_PRESET, choices=list(EMBEDDING_PRESETS))
    parser.add_argument("--data-dir", default=DATA_DIR)
    args = parser.parse_args()

    rows = load_dataset_rows(args.limit)
    print(f"Loaded {len(rows)} articles from {DATASET} ({DATASET_SPLIT} split)")
    preset = EMBEDDING_PRESETS[args.embedding]
    model = SentenceEmbeddings(preset["model"], preset)
    stats = build_index(rows, args.data_dir, model)

    # Keep a local copy of the model, so the server never needs the network.
    model.save(os.path.join(args.data_dir, "embedding_model"))
    stats["embedding_preset"] = args.embedding
    with open(os.path.join(args.data_dir, "manifest.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

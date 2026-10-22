"""Hybrid search: keyword (BM25) + semantic (embeddings), merged with Reciprocal Rank Fusion.

Keyword search is precise for names, numbers and acronyms ("LNG", "Dukovany").
Semantic search finds the same meaning in other words or another language.
"""

from typing import Dict, List

import numpy as np

from news_mcp.config import CANDIDATES_PER_RETRIEVER, RRF_K
from news_mcp.embeddings import SentenceEmbeddings
from news_mcp.store import Store


def reciprocal_rank_fusion(rankings: Dict[str, List[int]]) -> Dict[int, float]:
    """
    Merge ranked lists using only the ranks: score = sum of 1 / (RRF_K + rank).
    :param rankings: {"keyword": [chunk ids, best first], "semantic": [...]}
    :return: {chunk_id: score}, best first
    """
    scores = {}
    for ranked_ids in rankings.values():
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


class HybridSearch:
    def __init__(self, store: Store, embeddings: np.ndarray, model: SentenceEmbeddings):
        """
        :param embeddings: one normalised vector per chunk (row i = chunk_id i)
        """
        self.store = store
        self.embeddings = embeddings
        self.model = model
        self.chunk_article_ids = np.array(store.chunk_article_ids())

    def semantic_search(self, text: str, filters: dict, limit: int) -> List[int]:
        """Chunk ids with the highest cosine similarity, within the filters."""
        scores = self.embeddings @ self.model.encode_queries([text])[0]
        if filters:
            allowed = np.isin(self.chunk_article_ids, self.store.article_ids(filters))
            scores[~allowed] = -np.inf
        best = np.argsort(-scores)[:limit]
        return [int(i) for i in best if scores[i] > -np.inf]

    def search(self, text: str, filters: dict, limit: int = 10, mode: str = "hybrid") -> List[dict]:
        """
        :param mode: "hybrid", "keyword" or "semantic"
        :return: up to `limit` articles, each with its best-matching passage
        """
        rankings = {}
        if mode in ("hybrid", "keyword"):
            rankings["keyword"] = self.store.keyword_search(text, filters, CANDIDATES_PER_RETRIEVER)
        if mode in ("hybrid", "semantic"):
            rankings["semantic"] = self.semantic_search(text, filters, CANDIDATES_PER_RETRIEVER)
        fused = reciprocal_rank_fusion(rankings)

        # Keep only the best passage of each article.
        best_chunk_per_article = {}
        for chunk_id in fused:
            article_id = self.chunk_article_ids[chunk_id]
            if article_id not in best_chunk_per_article:
                best_chunk_per_article[article_id] = chunk_id
            if len(best_chunk_per_article) == limit:
                break

        results = []
        for chunk_id in best_chunk_per_article.values():
            result = self.store.passage(chunk_id)
            result["score"] = round(fused[chunk_id], 4)
            result["matched_by"] = [name for name, ids in rankings.items() if chunk_id in ids]
            results.append(result)
        return results

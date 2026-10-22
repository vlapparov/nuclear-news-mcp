"""All settings in one place. Most of them can be overridden with environment variables."""

import json
import os

# ---- data -----------------------------------------------------------------------------
DATASET = "eoplumbum/v4_nuclear_power_articles"
DATASET_SPLIT = "train"
DATA_DIR = os.environ.get("NEWS_MCP_DATA_DIR", "data")

# ---- embedding models -----------------------------------------------------------------
# Two presets; see README section A2 for the measured speed / quality trade-off.
EMBEDDING_PRESETS = {
    # Static embeddings: a lookup table instead of a neural network (much faster on a CPU, README A2).
    "fast": {
        "model": "sentence-transformers/static-similarity-mrl-multilingual-v1",
        "truncate_dim": 256,  # this model allows shorter vectors with little quality loss
        "device": "cpu",  # a table lookup gains nothing from a GPU (and older torch lacks it on MPS)
        "query_prefix": "",
        "passage_prefix": "",
    },
    # A small transformer trained on translation pairs: better when the question is in English
    # and the article is not, but far slower to embed the corpus on a CPU (README A2).
    "crosslingual": {
        "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "truncate_dim": None,
        "device": None,  # None = use a GPU (CUDA / Apple MPS) when there is one
        "query_prefix": "",
        "passage_prefix": "",
    },
}
EMBEDDING_PRESET = os.environ.get("NEWS_MCP_EMBEDDING", "fast")

# ---- chunking (measured in tokens of the embedding model's tokenizer) -------------------
# 128 / 16 won a comparison of 128, 256 and 512-token chunks (README section A1).
CHUNK_TOKENS = 128
CHUNK_OVERLAP_TOKENS = 16

# ---- search ---------------------------------------------------------------------------
CANDIDATES_PER_RETRIEVER = 200  # passages each retriever contributes before fusion
RRF_K = 60  # standard Reciprocal Rank Fusion constant

# ---- server ---------------------------------------------------------------------------
HOST = os.environ.get("NEWS_MCP_HOST", "127.0.0.1")
PORT = int(os.environ.get("NEWS_MCP_PORT", "8000"))

# Bearer tokens and the scopes (permissions) each one has.
#   news:search     -> every tool
#   news:read_full  -> additionally, the full text of articles (get_article)
TOKENS = {
    "test-token": {"client_id": "test-client", "scopes": ["news:search", "news:read_full"]},
}
if os.environ.get("NEWS_MCP_TOKENS_FILE"):
    with open(os.environ["NEWS_MCP_TOKENS_FILE"]) as f:
        TOKENS = json.load(f)

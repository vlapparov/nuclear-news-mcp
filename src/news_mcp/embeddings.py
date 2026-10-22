import os
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceEmbeddings:
    def __init__(self, model_path: str, preset: dict):
        """
        :param model_path: Hugging Face model id, or a local folder with a saved copy
        :param preset: one of config.EMBEDDING_PRESETS (prefixes, vector size)
        """
        self.preset = preset
        self.model = SentenceTransformer(
            model_path,
            truncate_dim=preset["truncate_dim"],
            device=preset["device"],
            local_files_only=os.path.isdir(model_path),  # a saved copy: never contact the Hub
        )

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        texts = [self.preset["query_prefix"] + q for q in queries]
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def encode_passages(self, passages: List[str]) -> np.ndarray:
        texts = [self.preset["passage_prefix"] + p for p in passages]
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)

    def count_tokens(self, text: str) -> int:
        """Number of tokens the model sees for this text (used to size chunks)."""
        tokenizer = self.model.tokenizer
        if hasattr(tokenizer, "model_max_length"):  # Hugging Face transformers tokenizer
            return len(tokenizer.encode(text, add_special_tokens=False, verbose=False))
        return len(tokenizer.encode(text, add_special_tokens=False).ids)  # `tokenizers` tokenizer

    def save(self, path: str) -> None:
        self.model.save(path)

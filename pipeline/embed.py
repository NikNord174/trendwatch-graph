"""Sentence embeddings for story titles. Local model, no API calls."""

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def embed_titles(titles: list[str]) -> np.ndarray:
    # Deferred import: torch is heavy and only this pipeline step needs it.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(
        titles, batch_size=256, show_progress_bar=True, normalize_embeddings=True
    )
    return np.asarray(vectors, dtype=np.float32)

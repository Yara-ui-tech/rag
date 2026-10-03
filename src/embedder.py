"""
Embedding generation and vector normalization module.
"""

import os
import numpy as np
from typing import List, Union, Optional
from dotenv import load_dotenv

load_dotenv()


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """Normalizes vector to L2 unit length."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


class EmbeddingClient:
    """
    Client for generating embeddings using OpenAI API (text-embedding-3-small).
    Features:
      - Automatic batching for high throughput
      - Vector normalization for fast cosine similarity dot-product math
      - Safe educational fallback if OPENAI_API_KEY is not configured
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_EMBEDDING_MODEL", model)
        self.dimensions = dimensions
        self._openai_client = None

        if self.api_key and not self.api_key.startswith("your-openai-api-key"):
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=self.api_key)
            except Exception as e:
                print(f"[!] Warning: Could not initialize OpenAI client: {e}")

    @property
    def is_live(self) -> bool:
        """Returns True if live OpenAI API client is ready."""
        return self._openai_client is not None

    def embed_text(self, text: str) -> np.ndarray:
        """Embeds a single string into a normalized NumPy float32 vector."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str], batch_size: int = 64) -> List[np.ndarray]:
        """
        Embeds a list of texts in batches.
        Returns a list of normalized 1D NumPy arrays.
        """
        if not texts:
            return []

        if not self.is_live:
            # Educational mock embedding generator
            return [self._generate_mock_embedding(t) for t in texts]

        results = []
        for i in range(0, len(texts), batch_size):
            batch = [t.replace("\n", " ").strip() for t in texts[i : i + batch_size]]
            # Replace empty strings to avoid OpenAI API error
            batch = [t if t else " " for t in batch]

            response = self._openai_client.embeddings.create(
                input=batch,
                model=self.model,
            )

            for item in response.data:
                vec = np.array(item.embedding, dtype=np.float32)
                results.append(normalize_vector(vec))

        return results

    def _generate_mock_embedding(self, text: str) -> np.ndarray:
        """Deterministic pseudo-semantic vector for educational offline exploration."""
        seed = abs(hash(text)) % (2**32)
        rng = np.random.RandomState(seed)
        mock_vec = rng.randn(self.dimensions).astype(np.float32)
        return normalize_vector(mock_vec)

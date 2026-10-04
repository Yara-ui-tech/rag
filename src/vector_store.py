"""
Vector Database storage and similarity search module using NumPy.
"""

import os
import json
from typing import List, Dict, Any, Optional, Callable
import numpy as np

from .chunker import DocumentChunk


class VectorStore:
    """
    Lightweight, fast in-memory vector database with JSON persistence.
    Uses normalized cosine dot-product matrix multiplication.
    """

    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self._matrix: Optional[np.ndarray] = None  # Shape (N, D)

    def __len__(self) -> int:
        return len(self.chunks)

    def add_chunks(
        self, chunks: List[DocumentChunk], embeddings: Optional[List[np.ndarray]] = None
    ) -> None:
        """Adds chunks and their corresponding embedding vectors to the store."""
        if not chunks:
            return

        new_vectors = []
        for i, chunk in enumerate(chunks):
            if embeddings is not None and i < len(embeddings):
                vec = embeddings[i]
                chunk.embedding = vec.tolist()
            elif chunk.embedding is not None:
                vec = np.array(chunk.embedding, dtype=np.float32)
            else:
                raise ValueError(f"Chunk {chunk.chunk_id} has no embedding provided!")

            self.chunks.append(chunk)
            new_vectors.append(vec)

        stacked_new = np.vstack(new_vectors)
        if self._matrix is None:
            self._matrix = stacked_new
        else:
            self._matrix = np.vstack([self._matrix, stacked_new])

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
        filter_metadata: Optional[Dict[str, Any]] = None,
        min_score: float = -1.0,
    ) -> List[Dict[str, Any]]:
        """
        Calculates cosine similarity dot products against all vectors.
        Returns top_k matching chunks with similarity scores.
        """
        if self._matrix is None or len(self.chunks) == 0:
            return []

        # Vector dot-product: (N, D) @ (D,) -> (N,)
        scores = self._matrix @ query_vector

        # Sort descending
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        for idx in ranked_indices:
            if len(results) >= top_k:
                break

            score = float(scores[idx])
            if score < min_score:
                continue

            chunk = self.chunks[idx]

            # Metadata filtering if specified
            if filter_metadata:
                match = True
                for k, v in filter_metadata.items():
                    if chunk.metadata.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            results.append({
                "rank": len(results) + 1,
                "score": score,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            })

        return results

    def save(self, filepath: str) -> None:
        """Persists the vector index and chunks to disk as JSON."""
        try:
            dir_name = os.path.dirname(os.path.abspath(filepath))
            os.makedirs(dir_name, exist_ok=True)

            payload = {
                "version": "1.0",
                "count": len(self.chunks),
                "chunks": [c.to_dict() for c in self.chunks],
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except OSError as e:
            print(f"[!] Warning: Could not persist VectorStore to disk ({e}). Operating in memory.")

    @classmethod
    def load(cls, filepath: str) -> "VectorStore":
        """Reconstructs a VectorStore from a saved JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Index file does not exist: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)

        store = cls()
        vectors = []
        for item in payload.get("chunks", []):
            chunk = DocumentChunk.from_dict(item)
            store.chunks.append(chunk)
            if chunk.embedding is not None:
                vectors.append(np.array(chunk.embedding, dtype=np.float32))

        if vectors:
            store._matrix = np.vstack(vectors)

        return store

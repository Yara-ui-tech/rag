"""
Unit tests for pure NumPy Vector Store.
"""

import os
import tempfile
import unittest
import numpy as np
from src.chunker import DocumentChunk
from src.vector_store import VectorStore
from src.embedder import normalize_vector


class TestVectorStore(unittest.TestCase):
    def test_top_k_cosine_search(self):
        store = VectorStore()

        # Create 3 synthetic orthogonal or known vectors
        v1 = normalize_vector(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v2 = normalize_vector(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        v3 = normalize_vector(np.array([0.9, 0.1, 0.0], dtype=np.float32))

        chunks = [
            DocumentChunk(chunk_id="c1", text="Target match", metadata={"cat": "test"}),
            DocumentChunk(chunk_id="c2", text="Unrelated match", metadata={"cat": "test"}),
            DocumentChunk(chunk_id="c3", text="Close runner up", metadata={"cat": "test"}),
        ]

        store.add_chunks(chunks, embeddings=[v1, v2, v3])
        self.assertEqual(len(store), 3)

        # Query aligned with v1
        query_vec = normalize_vector(np.array([1.0, 0.05, 0.0], dtype=np.float32))
        results = store.search(query_vec, top_k=2)

        self.assertEqual(len(results), 2)
        # c1 should be rank 1
        self.assertEqual(results[0]["chunk_id"], "c1")
        self.assertGreater(results[0]["score"], results[1]["score"])

    def test_save_and_reload(self):
        store = VectorStore()
        v = normalize_vector(np.array([0.5, 0.5], dtype=np.float32))
        chunks = [DocumentChunk(chunk_id="x1", text="Persistent text", metadata={"author": "Vance"})]
        store.add_chunks(chunks, [v])

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            store.save(tmp_path)
            loaded_store = VectorStore.load(tmp_path)
            self.assertEqual(len(loaded_store), 1)
            self.assertEqual(loaded_store.chunks[0].chunk_id, "x1")
            self.assertEqual(loaded_store.chunks[0].metadata["author"], "Vance")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()

"""
================================================================================
LESSON 03: BUILDING A VECTOR STORE & RETRIEVER FROM SCRATCH
================================================================================

What is a Vector Store?
-----------------------
A Vector Database (like Pinecone, ChromaDB, Weaviate, or Qdrant) is simply:
1. An index of embedding vectors
2. The original raw text chunks associated with each vector
3. Metadata (source filename, chunk ID, section title, timestamps)
4. A similarity search engine (e.g., Cosine Similarity or Approximate Nearest Neighbors)
5. Disk persistence (saving and loading)

In this lesson, you will build an in-memory Vector Store from pure Python 
and NumPy. You will understand how databases like Chroma or FAISS work under the hood!

In this lesson, you will learn:
  1. How to structure a Document Chunk with metadata
  2. Batch embedding ingestion
  3. Top-K similarity retrieval with score filtering
  4. Saving to disk and reloading an existing index
================================================================================
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# Import our helper from lesson 2 (or use the embedder directly)
from lesson_02_embeddings import get_embedding, normalize_vector


@dataclass
class DocumentChunk:
    """Represents a discrete text snippet with tracking metadata."""
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class SimpleVectorStore:
    """
    An educational, production-ready, pure NumPy vector database.
    Supports indexing, metadata filtering, top-k retrieval, and serialization.
    """

    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.vectors: Optional[np.ndarray] = None  # Shape: (N, Dimensions)

    def add_chunks(self, chunks: List[DocumentChunk], batch_size: int = 16) -> None:
        """
        Embeds and registers a list of DocumentChunks into the store.
        """
        print(f"[+] Indexing {len(chunks)} chunks into VectorStore...")
        new_embeddings = []

        for i, chunk in enumerate(chunks):
            if chunk.embedding is None:
                # Compute embedding and normalize for fast dot-product search
                vec = normalize_vector(get_embedding(chunk.text))
                chunk.embedding = vec.tolist()
            else:
                vec = np.array(chunk.embedding, dtype=np.float32)
                vec = normalize_vector(vec)

            new_embeddings.append(vec)
            self.chunks.append(chunk)

        # Update the NumPy matrix of vectors
        new_matrix = np.vstack(new_embeddings)
        if self.vectors is None:
            self.vectors = new_matrix
        else:
            self.vectors = np.vstack([self.vectors, new_matrix])

        print(f"[OK] Successfully indexed! Store now contains {len(self.chunks)} vectors.")

    def search(self, query: str, top_k: int = 3, min_similarity: float = 0.0) -> List[Dict[str, Any]]:
        """
        Performs cosine similarity search for the given query.
        Returns the top_k best matching chunks with their similarity scores.
        """
        if self.vectors is None or len(self.chunks) == 0:
            return []

        # 1. Embed and normalize user query
        query_vec = normalize_vector(get_embedding(query))

        # 2. Fast dot-product against all stored vectors: (N, D) @ (D,) -> (N,)
        similarities = self.vectors @ query_vec

        # 3. Sort indices descending by score
        ranked_indices = np.argsort(similarities)[::-1]

        # 4. Collect top_k results
        results = []
        for idx in ranked_indices[:top_k]:
            score = float(similarities[idx])
            if score < min_similarity:
                continue
            chunk = self.chunks[idx]
            results.append({
                "rank": len(results) + 1,
                "score": score,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata
            })

        return results

    def save(self, filepath: str) -> None:
        """Saves the index and metadata to a single JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "chunks": [asdict(c) for c in self.chunks]
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[OK] VectorStore persisted to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "SimpleVectorStore":
        """Loads a saved VectorStore from disk."""
        store = cls()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        vectors = []
        for item in data["chunks"]:
            chunk = DocumentChunk(**item)
            store.chunks.append(chunk)
            vectors.append(np.array(chunk.embedding, dtype=np.float32))

        if vectors:
            store.vectors = np.vstack(vectors)
        print(f"[OK] Loaded {len(store.chunks)} vectors from: {filepath}")
        return store


# ----------------------------------------------------------------------
# RUNNING THE LESSON DEMO
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("LESSON 3 DEMONSTRATION: VECTOR STORE & RETRIEVAL")
    print("=" * 70)

    # 1. Create sample document chunks (from Lesson 1)
    raw_texts = [
        ("The QuantumOrbit X-9 nominal warp frequency is 44.8 THz with Helium-3 coolant.", "manual_sec_2"),
        ("Emergency containment breach protocol requires depressurizing secondary plasma manifold within 30s.", "manual_sec_3"),
        ("All full-time engineers get $1,850 home office setup allowance for ergonomic desks and chairs.", "policy_sec_1"),
        ("Weekend on-call standby rate is $150 per 12-hour shift with a $65 meal allowance for incidents.", "policy_sec_2"),
        ("Recharge Friday is the last Friday of every quarter with zero meetings.", "policy_sec_4"),
    ]

    chunks = [
        DocumentChunk(
            chunk_id=f"doc_{i}",
            text=text,
            metadata={"source": doc_ref, "category": "engineering" if "manual" in doc_ref else "hr"}
        )
        for i, (text, doc_ref) in enumerate(raw_texts)
    ]

    # 2. Ingest chunks into our vector store
    store = SimpleVectorStore()
    store.add_chunks(chunks)

    # 3. Perform a semantic query
    query = "How much can I spend on my desk and chair setup at home?"
    print(f"\n[?] Query: \"{query}\"")
    results = store.search(query, top_k=2)

    print("\n--- Retrieved Top 2 Chunks ---")
    for r in results:
        print(f"Rank {r['rank']} | Score: {r['score']:.4f} | Source: {r['metadata']['source']}")
        print(f"  Content: \"{r['text']}\"\n")

    # 4. Demonstrate Persistence (Save & Reload)
    storage_path = os.path.join(os.path.dirname(__file__), "..", "data", "demo_index.json")
    store.save(storage_path)

    print("\n--- Reloading VectorStore from disk ---")
    reloaded_store = SimpleVectorStore.load(storage_path)
    reloaded_results = reloaded_store.search("What frequency does the warp drive run at?", top_k=1)
    print(f"Query against reloaded store: Warp frequency")
    print(f"  Best Match: \"{reloaded_results[0]['text']}\" (Score: {reloaded_results[0]['score']:.4f})")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS FOR STUDENTS:")
    print("1. A Vector Store pairs embeddings with chunks and source metadata.")
    print("2. Search is fast mathematical ranking using vector dot products.")
    print("3. Persisting the index means embeddings only need to be computed ONCE.")
    print("4. Next step: Lesson 04 will inject these retrieved chunks into an LLM prompt!")
    print("=" * 70)

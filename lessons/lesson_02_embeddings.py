"""
================================================================================
LESSON 02: EMBEDDINGS & VECTOR MATHEMATICS (How Computers 'Understand' Text)
================================================================================

What is an Embedding?
----------------------
Computers cannot understand words; they understand numbers.
An embedding model converts a piece of text into a high-dimensional vector 
(a list of numbers, e.g. 1,536 dimensions for OpenAI's `text-embedding-3-small`).

The magic of embeddings is *semantic geometry*:
Texts with similar meanings point in similar directions in vector space,
EVEN IF THEY DO NOT SHARE A SINGLE WORD!
  - "The dog barked" & "A canine made a noise" -> Very high similarity!
  - "The dog barked" & "Quantum particle spin"  -> Very low similarity!

In this lesson, you will learn:
  1. How to generate embeddings using the OpenAI API
  2. The mathematics of Cosine Similarity from scratch (using NumPy)
  3. Why normalized vectors make search lightning-fast
================================================================================
"""

import os
import math
import numpy as np
from typing import List, Union
from dotenv import load_dotenv

# Load environment variables (such as OPENAI_API_KEY from .env)
load_dotenv()


# ----------------------------------------------------------------------
# 1. PURE MATHEMATICS: COSINE SIMILARITY FROM FIRST PRINCIPLES
# ----------------------------------------------------------------------
def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Computes the dot product: A . B = sum(A_i * B_i)
    Measures how much two vectors point in the same direction.
    """
    return float(np.sum(v1 * v2))


def vector_norm(v: np.ndarray) -> float:
    """
    Computes the Euclidean magnitude (length): ||v|| = sqrt(sum(v_i^2))
    """
    return float(np.sqrt(np.sum(v ** 2)))


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Cosine Similarity = (v1 . v2) / (||v1|| * ||v2||)
    
    Returns a score between -1.0 and 1.0:
      1.0  = Exactly identical direction (identical semantic meaning)
      0.0  = Orthogonal (completely unrelated)
     -1.0  = Diametrically opposite
    """
    norm1 = vector_norm(v1)
    norm2 = vector_norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    return dot_product(v1, v2) / (norm1 * norm2)


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalizes a vector to unit length (length = 1.0).
    TRICK USED IN PRODUCTION:
    If all vectors have length 1.0, then cosine_similarity(A, B) = dot_product(A, B)!
    This turns search into a single fast matrix multiplication.
    """
    norm = vector_norm(v)
    if norm == 0:
        return v
    return v / norm


# ----------------------------------------------------------------------
# 2. GENERATING REAL EMBEDDINGS (OpenAI text-embedding-3-small)
# ----------------------------------------------------------------------
def get_embedding(text: str, model: str = "text-embedding-3-small") -> np.ndarray:
    """
    Calls the OpenAI Embeddings API to generate a 1,536-dimensional vector.
    Includes a fallback educational simulator if no API key is provided yet.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key or api_key.startswith("your-openai-api-key"):
        # EDUCATIONAL FALLBACK: When learner hasn't added their API key yet,
        # generate a deterministic pseudo-semantic vector so the math can still be studied!
        print("  [!] Notice: No valid OPENAI_API_KEY found in .env; using educational mock vector.")
        print("      To use real OpenAI embeddings, add your key to .env!")
        np.random.seed(abs(hash(text)) % (2**32))
        mock_vec = np.random.randn(1536)
        return normalize_vector(mock_vec)

    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    
    # Strip newline characters as recommended by OpenAI for embedding models
    clean_text = text.replace("\n", " ")
    response = client.embeddings.create(input=[clean_text], model=model)
    embedding = response.data[0].embedding
    return np.array(embedding, dtype=np.float32)


# ----------------------------------------------------------------------
# RUNNING THE LESSON DEMO
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("LESSON 2 DEMONSTRATION: EMBEDDINGS & VECTOR MATH")
    print("=" * 70)

    # 1. Compare three sentences
    sentences = [
        "The puppy was playing in the backyard with a ball.",       # Sentence A
        "A young dog is outside having fun with a toy sphere.",      # Sentence B (Semantically identical to A!)
        "Quantum superposition in transmon qubits at 15 millikelvin." # Sentence C (Completely unrelated)
    ]

    print("\n[+] Generating embeddings for 3 test sentences...\n")
    for i, s in enumerate(sentences):
        print(f"  Sentence {chr(65+i)}: \"{s}\"")

    vectors = [get_embedding(s) for s in sentences]

    print(f"\n[+] Vector Dimensionality: {len(vectors[0])} dimensions per sentence.")
    print(f"    Sample values from Vector A: {vectors[0][:5]} ...")

    # 2. Calculate Cosine Similarities
    sim_A_B = cosine_similarity(vectors[0], vectors[1])
    sim_A_C = cosine_similarity(vectors[0], vectors[2])
    sim_B_C = cosine_similarity(vectors[1], vectors[2])

    print("\n--- Cosine Similarity Results ---")
    print(f"  Sim(A, B) [Puppy vs. Young Dog]:       {sim_A_B:+.4f} (High semantic overlap)")
    print(f"  Sim(A, C) [Puppy vs. Quantum Qubits]:  {sim_A_C:+.4f} (Nearly orthogonal / unrelated)")
    print(f"  Sim(B, C) [Young Dog vs. Quantum]:     {sim_B_C:+.4f} (Nearly orthogonal / unrelated)")

    # 3. Fast Matrix Search Demonstration
    print("\n--- Fast Batch Retrieval Using Matrix Dot Product ---")
    # Stack normalized vectors into a matrix (N x D)
    matrix = np.vstack([normalize_vector(v) for v in vectors])
    query = "Where is the little dog playing?"
    print(f"  Query: \"{query}\"")
    
    query_vector = normalize_vector(get_embedding(query))
    
    # In one single matrix multiplication: (N x D) @ (D,) -> (N,) similarities!
    scores = matrix @ query_vector
    for idx, score in enumerate(scores):
        print(f"  Match #{idx+1} (Sentence {chr(65+idx)}): Similarity Score = {score:+.4f}")

    best_match_idx = int(np.argmax(scores))
    print(f"\n  -> Top retrieved document: Sentence {chr(65+best_match_idx)}")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS FOR STUDENTS:")
    print("1. Text embeddings map semantic concepts into geometric space.")
    print("2. Cosine similarity measures angle, ignoring magnitude differences.")
    print("3. Pre-normalizing vectors turns vector retrieval into simple matrix multiplication.")
    print("4. Next step: Lesson 03 will build a persistent Vector Store database!")
    print("=" * 70)

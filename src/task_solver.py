"""
Task Solver & Step-by-Step RAG Solution Generator.
Analyzes user-provided tasks and generates structured step-by-step guides, 
production code solutions, tests, and best practices.
"""

import os
import re
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


class TaskSolver:
    """
    Expert RAG Mentor that translates tasks into clear, executable steps,
    code solutions, and test plans.
    """

    SYSTEM_PROMPT = """You are an elite AI Architect and Master RAG Instructor.
A student or engineer has given you a specific RAG task.
Your job is to provide a complete, clear, production-grade guide with:

1. 🎯 ARCHITECTURE & STRATEGY (1-2 sentences explaining the approach)
2. 📋 STEP-BY-STEP EXECUTION PLAN (Numbered list of exact steps)
3. 💻 COMPLETE PYTHON IMPLEMENTATION (Clean, runnable, well-commented code, ZERO placeholders like '... implement here ...')
4. 🧪 VERIFICATION & TEST CODE (How to run it and verify it works)
5. ⚠️ COMMON PITFALLS & GOTCHAS (What commonly goes wrong and how to fix it)

Format your response cleanly with clear Markdown headers and code blocks.
Prioritize clean architecture, high readability, and production standards."""

    BUILTIN_PATTERNS = {
        "chunk": {
            "title": "Document Loading & Recursive Text Splitting with Overlap",
            "steps": [
                "1. Load raw text or markdown document into memory.",
                "2. Define tiered separators (Paragraphs '\\n\\n' -> Sentences '\\n' -> Phrases ' ' -> Characters '').",
                "3. Recursively split text until each segment is smaller than the target chunk size.",
                "4. Merge small segments with chunk_overlap to maintain cross-boundary context.",
                "5. Attach unique chunk_id and metadata (source document, index, token count).",
            ],
            "code": '''import tiktoken
from typing import List, Dict, Any

class RecursiveSemanticChunker:
    """Production recursive text chunker preserving natural boundaries."""
    
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def split_text(self, text: str) -> List[str]:
        separators = ["\\n\\n", "\\n", ". ", " ", ""]
        splits = []
        self._recursive_split(text, separators, splits)
        return self._merge_with_overlap(splits)

    def _recursive_split(self, text: str, seps: List[str], accumulator: List[str]):
        if not seps or len(text) <= self.chunk_size:
            accumulator.append(text.strip())
            return
        sep = seps[0]
        pieces = text.split(sep) if sep else list(text)
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if len(piece) <= self.chunk_size:
                accumulator.append(piece)
            else:
                self._recursive_split(piece, seps[1:], accumulator)

    def _merge_with_overlap(self, pieces: List[str]) -> List[str]:
        chunks = []
        current = ""
        for piece in pieces:
            if not current:
                current = piece
            elif len(current) + 1 + len(piece) <= self.chunk_size:
                current += "\\n" + piece
            else:
                chunks.append(current)
                overlap = current[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                current = (overlap + " " + piece).strip()
        if current:
            chunks.append(current)
        return chunks

# Test the chunker
sample_doc = "QuantumOrbit X-9 uses Helium-3 coolant.\\n\\nWarp frequency is 44.8 THz."
chunker = RecursiveSemanticChunker(chunk_size=50, chunk_overlap=15)
chunks = chunker.split_text(sample_doc)
for i, c in enumerate(chunks):
    print(f"Chunk {i+1} ({chunker.count_tokens(c)} tokens): {repr(c)}")
''',
            "test": "Run: python -c 'from script import RecursiveSemanticChunker; c = RecursiveSemanticChunker(50, 15); print(c.split_text(\"A quick test.\"))'",
            "pitfalls": "Setting overlap too high (creates duplicate chunks) or too low (severs critical names and technical numbers across chunk cuts).",
        },
        "vector": {
            "title": "Cosine Similarity & Pure NumPy Vector Database",
            "steps": [
                "1. Convert text chunks to high-dimensional embedding vectors.",
                "2. Normalize all vectors to unit L2 length (||v|| = 1.0).",
                "3. Convert query string to normalized embedding vector.",
                "4. Compute dot product of matrix (N, D) against query (D,) to obtain all cosine similarities in 1 operation.",
                "5. Use np.argsort()[::-1] to extract top-K highest scoring chunks with metadata.",
            ],
            "code": '''import numpy as np
from typing import List, Dict, Any

def normalize_vector(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

class PureNumPyVectorStore:
    """In-memory vector database with fast dot-product search."""
    
    def __init__(self):
        self.chunks = []
        self.matrix = None

    def add(self, text: str, embedding: np.ndarray, metadata: Dict[str, Any]):
        norm_vec = normalize_vector(embedding)
        self.chunks.append({"text": text, "metadata": metadata})
        if self.matrix is None:
            self.matrix = norm_vec.reshape(1, -1)
        else:
            self.matrix = np.vstack([self.matrix, norm_vec])

    def search(self, query_vec: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.matrix is None:
            return []
        query_norm = normalize_vector(query_vec)
        # Fast cosine similarity: (N, D) @ (D,) -> (N,)
        scores = self.matrix @ query_norm
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {"rank": i + 1, "score": float(scores[idx]), "chunk": self.chunks[idx]}
            for i, idx in enumerate(ranked_indices)
        ]

# Demonstration
store = PureNumPyVectorStore()
dim = 1536
v1 = np.random.randn(dim).astype(np.float32)
v2 = np.random.randn(dim).astype(np.float32)
store.add("Coolant is Helium-3", v1, {"source": "manual.md"})
store.add("Remote work reimbursement $1850", v2, {"source": "policy.md"})

results = store.search(v1, top_k=1)
print(f"Top Match: {results[0]['chunk']['text']} (Score: {results[0]['score']:.4f})")
''',
            "test": "Verify that searching with the exact vector of document #1 produces a similarity score of exactly 1.0000.",
            "pitfalls": "Forgetting to normalize vectors before dot-product search, which causes larger magnitude vectors to artificially dominate scores.",
        },
        "hybrid": {
            "title": "Hybrid Search: Combining BM25 Keyword Search + Dense Vector Search",
            "steps": [
                "1. Build a dense semantic vector index (for capturing concepts and synonyms).",
                "2. Build an inverted index / BM25 keyword index (for exact acronyms, model IDs, part numbers).",
                "3. On query, execute both dense vector search and BM25 keyword search independently.",
                "4. Normalize scores using Reciprocal Rank Fusion (RRF): RRF_Score = 1 / (60 + rank).",
                "5. Combine scores, sort descending, and return the fused top-K results.",
            ],
            "code": '''import math
from collections import Counter
from typing import List, Dict, Any

class BM25KeywordIndex:
    """Lightweight BM25 keyword search implementation."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = []
        self.doc_lens = []
        self.avg_len = 0
        self.doc_freqs = Counter()

    def fit(self, docs: List[str]):
        self.docs = docs
        self.doc_lens = [len(d.lower().split()) for d in docs]
        self.avg_len = sum(self.doc_lens) / max(len(docs), 1)
        for d in docs:
            words = set(d.lower().split())
            for w in words:
                self.doc_freqs[w] += 1

    def score(self, query: str) -> List[float]:
        q_words = query.lower().split()
        n_docs = len(self.docs)
        scores = [0.0] * n_docs
        for qw in q_words:
            df = self.doc_freqs.get(qw, 0)
            if df == 0:
                continue
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for i, doc in enumerate(self.docs):
                words = doc.lower().split()
                tf = words.count(qw)
                denom = tf + self.k1 * (1 - self.b + self.b * (self.doc_lens[i] / self.avg_len))
                scores[i] += idf * ((tf * (self.k1 + 1)) / max(denom, 1e-6))
        return scores

def reciprocal_rank_fusion(dense_ranks: List[int], bm25_ranks: List[int], k: int = 60) -> float:
    """Reciprocal Rank Fusion (RRF) algorithm."""
    score = 0.0
    for r in dense_ranks:
        score += 1.0 / (k + r)
    for r in bm25_ranks:
        score += 1.0 / (k + r)
    return score

# Example
corpus = ["Model SR-90 warp drive", "Helium-3 coolant pump flow rate", "Nexus Horizon remote work"]
bm25 = BM25KeywordIndex()
bm25.fit(corpus)
print("BM25 scores for 'SR-90':", bm25.score("SR-90"))
''',
            "test": "Search for an exact model number like 'SR-90' and verify BM25 scores the exact document highest.",
            "pitfalls": "Pure vector search often fails on arbitrary part numbers ('QO-ENG-2026') or codes; Hybrid search fixes this completely.",
        },
    }

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self._openai_client = None
        if self.api_key and not self.api_key.startswith("your-openai-api-key"):
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=self.api_key)
            except Exception:
                pass

    @property
    def is_live(self) -> bool:
        return self._openai_client is not None

    def solve(self, task_description: str) -> Dict[str, Any]:
        """
        Takes a user's task and returns a complete step-by-step tutorial,
        implementation, and test suite.
        """
        task_clean = task_description.strip()
        if not task_clean:
            return {"error": "Task description cannot be empty."}

        # If live OpenAI client is available, use GPT-4o to generate deep custom solution
        if self.is_live:
            try:
                model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
                response = self._openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"Please provide the complete step-by-step engineering plan, "
                                f"working Python implementation, and test suite for this RAG task:\n\n"
                                f"TASK: {task_clean}"
                            ),
                        },
                    ],
                    temperature=0.1,
                )
                solution_markdown = response.choices[0].message.content
                return {
                    "task": task_clean,
                    "is_live_ai": True,
                    "solution_markdown": solution_markdown,
                }
            except Exception as e:
                print(f"[!] Warning: OpenAI call failed ({e}), falling back to pattern solver.")

        # Fallback to intelligent pattern solver
        pattern_key = self._match_pattern(task_clean)
        pattern = self.BUILTIN_PATTERNS.get(pattern_key, self.BUILTIN_PATTERNS["chunk"])

        solution_md = f"""# [TASK]: {task_clean}

## 1. Architecture & Strategy
To successfully solve this task, we follow a modular design that isolates document ingestion, representation, and retrieval.

## 2. Step-by-Step Execution Plan
{chr(10).join(pattern['steps'])}

## 3. Complete Python Implementation
```python
{pattern['code']}
```

## 4. How to Verify and Test
{pattern['test']}

## 5. Common Pitfalls & Gotchas
{pattern['pitfalls']}
"""

        return {
            "task": task_clean,
            "is_live_ai": False,
            "solution_markdown": solution_md,
        }

    def _match_pattern(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["hybrid", "bm25", "keyword", "fusion", "rrf"]):
            return "hybrid"
        if any(w in q for w in ["vector", "cosine", "similarity", "embed", "math", "store", "database"]):
            return "vector"
        return "chunk"

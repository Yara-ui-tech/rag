"""
================================================================================
LESSON 05: THE COMPLETE END-TO-END RAG PIPELINE
================================================================================

Putting It All Together:
------------------------
You have learned all 4 core components individually:
  1. Document Loading & Recursive Chunking (Lesson 1)
  2. Embeddings & Cosine Vector Mathematics (Lesson 2)
  3. Vector Store Indexing, Persistence, & Search (Lesson 3)
  4. Prompt Augmentation & Grounded LLM Generation (Lesson 4)

Now, we assemble these components into a clean, reusable `RAGPipeline` class!
This is the standard architectural pattern used in modern production AI systems.

In this lesson, you will learn:
  - How to wire ingestion -> chunking -> vector indexing -> query -> generation
  - Measuring retrieval quality and relevance scores
  - Providing source citations in responses
================================================================================
"""

import os
import glob
import time
from typing import List, Dict, Any, Optional

from lesson_01_chunking import RecursiveCharacterTextSplitter, count_tokens
from lesson_03_vector_store import SimpleVectorStore, DocumentChunk
from lesson_04_augmented_prompt import format_context_prompt, generate_answer


class RAGPipeline:
    """
    Orchestrates the complete Retrieval-Augmented Generation lifecycle.
    """

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        top_k: int = 3,
        storage_path: Optional[str] = None
    ):
        self.chunker = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vector_store = SimpleVectorStore()
        self.top_k = top_k
        self.storage_path = storage_path

    def ingest_files(self, file_paths: List[str]) -> None:
        """
        Loads, chunks, and indexes all documents from the provided file paths.
        """
        all_chunks: List[DocumentChunk] = []
        print(f"\n[+] Beginning ingestion of {len(file_paths)} document(s)...")

        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"  [!] Warning: File {file_path} does not exist, skipping.")
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            filename = os.path.basename(file_path)
            raw_splits = self.chunker.split_text(content)
            print(f"  -> File: {filename} ({len(content)} chars) -> {len(raw_splits)} chunks")

            for idx, split_text in enumerate(raw_splits):
                chunk = DocumentChunk(
                    chunk_id=f"{filename}#part{idx+1}",
                    text=split_text,
                    metadata={
                        "source": filename,
                        "file_path": file_path,
                        "chunk_index": idx + 1,
                        "token_count": count_tokens(split_text)
                    }
                )
                all_chunks.append(chunk)

        # Index all chunks into the vector store
        self.vector_store.add_chunks(all_chunks)

        if self.storage_path:
            self.vector_store.save(self.storage_path)

    def query(self, question: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Executes the full RAG query flow:
          1. Retrieve top_k relevant chunks from vector store
          2. Assemble augmented context prompt
          3. Generate grounded answer via LLM
        """
        start_time = time.time()

        # Step 1: Retrieve
        retrieved_chunks = self.vector_store.search(question, top_k=self.top_k)
        retrieval_ms = (time.time() - start_time) * 1000

        # Step 2: Augment
        augmented_prompt = format_context_prompt(question, retrieved_chunks)

        # Step 3: Generate
        gen_start = time.time()
        answer = generate_answer(augmented_prompt)
        generation_ms = (time.time() - gen_start) * 1000

        result = {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "retrieval_time_ms": retrieval_ms,
            "generation_time_ms": generation_ms,
            "total_time_ms": (time.time() - start_time) * 1000
        }

        if verbose:
            self._print_query_summary(result)

        return result

    def _print_query_summary(self, result: Dict[str, Any]) -> None:
        print("\n" + "=" * 70)
        print(f"USER QUESTION: {result['question']}")
        print("=" * 70)
        print(f"[Stats] Retrieved {len(result['retrieved_chunks'])} chunks in {result['retrieval_time_ms']:.1f}ms | Generation: {result['generation_time_ms']:.1f}ms")
        print("\n--- RETRIEVED SOURCES ---")
        for chunk in result["retrieved_chunks"]:
            source = chunk["metadata"]["source"]
            chunk_idx = chunk["metadata"]["chunk_index"]
            score = chunk["score"]
            print(f"  - [{source} Part {chunk_idx}] (Score: {score:.4f})")
            snippet = chunk["text"][:120].replace("\n", " ")
            print(f"    Snippet: \"{snippet}...\"")

        print("\n--- SYNTHESIZED RAG ANSWER ---")
        print(result["answer"])
        print("=" * 70 + "\n")


# ----------------------------------------------------------------------
# RUNNING THE LESSON DEMO
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("LESSON 5 DEMONSTRATION: FULL RAG PIPELINE IN ACTION")
    print("=" * 70)

    # 1. Locate sample data files
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    sample_files = glob.glob(os.path.join(data_dir, "*.md"))

    # 2. Instantiate and ingest documents
    index_file = os.path.join(data_dir, "rag_pipeline_index.json")
    pipeline = RAGPipeline(chunk_size=400, chunk_overlap=80, top_k=2, storage_path=index_file)
    pipeline.ingest_files(sample_files)

    # 3. Test queries across technical and HR documents
    questions = [
        "What are the emergency containment breach steps if RED-ECHO-7 sounds?",
        "What is the budget for remote work home office desks and equipment?",
        "What is the policy for attending international technical conferences?"
    ]

    for q in questions:
        pipeline.query(q)

    print("\n" + "=" * 70)
    print("CONGRATULATIONS!")
    print("You have mastered the foundational concepts of RAG:")
    print("  [1] Chunking & Text Splitting")
    print("  [2] High-Dimensional Embeddings & Cosine Mathematics")
    print("  [3] Vector Databases & Fast Top-K Retrieval")
    print("  [4] Prompt Augmentation & Anti-Hallucination Guardrails")
    print("  [5] Full Pipeline Architecture & Lifecycle")
    print("=" * 70)

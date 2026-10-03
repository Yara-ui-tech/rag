"""
Unified RAG Pipeline connecting Chunking, Embeddings, Vector Store, and Generation.
"""

import os
import glob
import time
from typing import List, Dict, Any, Optional, Generator

from .chunker import RecursiveCharacterTextSplitter, DocumentChunk, load_document
from .embedder import EmbeddingClient
from .vector_store import VectorStore
from .generator import PromptGenerator, LLMGenerator


class RAGPipeline:
    """
    Modular, end-to-end RAG orchestrator.
    Handles document ingestion, vector indexing, retrieval, and grounded generation.
    """

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        top_k: int = 3,
        chat_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        index_file: Optional[str] = None,
    ):
        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.embedder = EmbeddingClient(model=embedding_model)
        self.vector_store = VectorStore()
        self.generator = LLMGenerator(model=chat_model)
        self.top_k = top_k
        self.index_file = index_file

        if self.index_file and os.path.exists(self.index_file):
            print(f"[+] Found existing index at {self.index_file}, loading...")
            self.vector_store = VectorStore.load(self.index_file)
            print(f"[OK] Loaded {len(self.vector_store)} chunks from disk.")

    def ingest_files(self, file_paths: List[str]) -> int:
        """Loads and indexes files. Returns number of chunks created."""
        all_chunks: List[DocumentChunk] = []

        for path in file_paths:
            if not os.path.isfile(path):
                continue
            text = load_document(path)
            doc_name = os.path.basename(path)
            chunks = self.chunker.chunk_document(
                text=text,
                source_name=doc_name,
                extra_metadata={"path": os.path.abspath(path)},
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        # Compute embeddings in batches
        texts = [c.text for c in all_chunks]
        embeddings = self.embedder.embed_batch(texts)
        self.vector_store.add_chunks(all_chunks, embeddings)

        if self.index_file:
            self.vector_store.save(self.index_file)

        return len(all_chunks)

    def ingest_directory(
        self,
        dir_path: str,
        extensions: Optional[List[str]] = None,
        pattern: Optional[str] = None,
    ) -> int:
        """Finds all document files in dir_path and ingests them."""
        allowed_exts = extensions or [".md", ".txt", ".rst", ".doc"]
        all_files = []
        for root, _, filenames in os.walk(dir_path):
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext in allowed_exts:
                    all_files.append(os.path.join(root, fname))
        return self.ingest_files(all_files)

    def query(self, question: str) -> Dict[str, Any]:
        """
        Executes a RAG query:
          1. Embeds question
          2. Retrieves top-k matching chunks
          3. Builds grounded prompt
          4. Generates LLM completion
        """
        t0 = time.time()

        # Step 1: Embed query
        query_vec = self.embedder.embed_text(question)

        # Step 2: Retrieve
        t_ret_start = time.time()
        retrieved = self.vector_store.search(query_vec, top_k=self.top_k)
        retrieval_ms = (time.time() - t_ret_start) * 1000

        # Step 3: Prompt augmentation
        prompt = PromptGenerator.build_user_prompt(question, retrieved)

        # Step 4: Generation
        t_gen_start = time.time()
        answer = self.generator.generate(prompt)
        generation_ms = (time.time() - t_gen_start) * 1000

        total_ms = (time.time() - t0) * 1000

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved,
            "prompt": prompt,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms": total_ms,
        }

    def query_stream(self, question: str) -> Generator[Dict[str, Any], None, None]:
        """
        Yields status metadata first, then stream deltas, and finally full summary.
        """
        query_vec = self.embedder.embed_text(question)
        retrieved = self.vector_store.search(query_vec, top_k=self.top_k)
        prompt = PromptGenerator.build_user_prompt(question, retrieved)

        yield {"type": "retrieval", "retrieved_chunks": retrieved, "prompt": prompt}

        full_answer = []
        for delta in self.generator.generate_stream(prompt):
            full_answer.append(delta)
            yield {"type": "token", "delta": delta}

        yield {"type": "done", "full_answer": "".join(full_answer)}

"""
Document loading and semantic chunking module.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import tiktoken


@dataclass
class DocumentChunk:
    """Represents a text chunk with unique ID, text, metadata, and optional embedding."""
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        return cls(
            chunk_id=data["chunk_id"],
            text=data["text"],
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
        )


class TokenCounter:
    """Calculates tokens using OpenAI's tokenizer."""

    def __init__(self, model_encoding: str = "cl100k_base"):
        self.encoding = tiktoken.get_encoding(model_encoding)

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text))


class RecursiveCharacterTextSplitter:
    """
    Recursively splits text using natural structural separators:
    Paragraphs -> Sentences -> Words -> Characters.
    """

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        self.token_counter = TokenCounter()

    def split_text(self, text: str) -> List[str]:
        raw_splits: List[str] = []
        self._split_recursive(text, self.separators, raw_splits)
        return self._merge_splits(raw_splits)

    def _split_recursive(
        self, text: str, separators: List[str], final_splits: List[str]
    ) -> None:
        if not separators:
            final_splits.append(text)
            return

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            for char in text:
                final_splits.append(char)
            return

        pieces = text.split(sep)
        for piece in pieces:
            piece_clean = piece.strip()
            if not piece_clean:
                continue

            if len(piece_clean) <= self.chunk_size:
                final_splits.append(piece_clean)
            else:
                self._split_recursive(piece_clean, remaining_seps, final_splits)

    def _merge_splits(self, pieces: List[str]) -> List[str]:
        chunks = []
        current = ""

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue

            if not current:
                current = piece
            elif len(current) + 1 + len(piece) <= self.chunk_size:
                current += "\n" + piece
            else:
                chunks.append(current)
                overlap = current[-self.chunk_overlap :] if self.chunk_overlap > 0 else ""
                current = (overlap + " " + piece).strip()

        if current:
            chunks.append(current)

        return chunks

    def chunk_document(
        self, text: str, source_name: str, extra_metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """Convenience method that creates fully formed DocumentChunk objects."""
        raw_chunks = self.split_text(text)
        result = []
        extra_meta = extra_metadata or {}

        for idx, chunk_text in enumerate(raw_chunks):
            chunk_id = f"{source_name}#part{idx + 1}"
            meta = {
                "source": source_name,
                "chunk_index": idx + 1,
                "token_count": self.token_counter.count(chunk_text),
                "char_count": len(chunk_text),
                **extra_meta,
            }
            result.append(DocumentChunk(chunk_id=chunk_id, text=chunk_text, metadata=meta))

        return result


def load_document(file_path: str) -> str:
    """Reads a text or markdown document from disk."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

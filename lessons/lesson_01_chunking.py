"""
================================================================================
LESSON 01: DOCUMENT LOADING & CHUNKING (The Foundation of RAG)
================================================================================

Why do we need chunking in RAG?
--------------------------------
1. Semantic Specificity: An embedding of an entire 100-page document is a 
   "blurry average" of everything in it. An embedding of a 300-word paragraph 
   captures a specific concept with high fidelity.
2. Context Window & Cost: Even modern LLMs with 128k context windows suffer from 
   "lost in the middle" degradation when given too much noise. Finding the 
   exact 2-3 paragraphs saves money and increases accuracy.
3. Chunk Overlap: When we split text, critical context might be sliced right at 
   the cut point. Overlapping chunks ensure no sentence or thought is severed.

In this lesson, you will learn:
  - Naive splitting vs. Recursive splitting
  - The role of chunk overlap
  - Token counting using tiktoken
================================================================================
"""

import os
from typing import List, Dict, Any
import tiktoken

# Initialize tokenizer for OpenAI models (cl100k_base used by text-embedding-3 / gpt-4)
tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the exact number of tokens in a text string."""
    return len(tokenizer.encode(text))


# ----------------------------------------------------------------------
# 1. NAIVE CHUNKER (Fixed character count)
# ----------------------------------------------------------------------
def naive_chunk_text(text: str, chunk_size: int = 150) -> List[str]:
    """
    Splits text every `chunk_size` characters, without caring about words or lines.
    NOTE: This is what you should NEVER do in production! Notice how it cuts words.
    """
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


# ----------------------------------------------------------------------
# 2. RECURSIVE TEXT SPLITTER (Production Quality from Scratch)
# ----------------------------------------------------------------------
class RecursiveCharacterTextSplitter:
    """
    Recursively splits text by prioritizing natural semantic boundaries:
      1. Paragraph breaks ("\\n\\n")
      2. Line breaks ("\\n")
      3. Sentence or phrase spaces (" ")
      4. Raw characters ("")
    """

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        separators: List[str] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        final_chunks: List[str] = []
        self._split_recursive(text, self.separators, final_chunks)
        return self._merge_splits(final_chunks)

    def _split_recursive(
        self, text: str, separators: List[str], final_chunks: List[str]
    ) -> None:
        if not separators:
            final_chunks.append(text)
            return

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # Character level fallback
            for char in text:
                final_chunks.append(char)
            return

        splits = text.split(separator)
        for piece in splits:
            if not piece.strip():
                continue
            # If the piece is small enough, keep it; otherwise subdivide with next separator
            if len(piece) <= self.chunk_size:
                final_chunks.append(piece)
            else:
                self._split_recursive(piece, remaining_separators, final_chunks)

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """Merges small pieces into chunks up to chunk_size, applying chunk_overlap."""
        merged_chunks = []
        current_chunk = ""

        for piece in splits:
            piece = piece.strip()
            if not piece:
                continue

            if not current_chunk:
                current_chunk = piece
            elif len(current_chunk) + 1 + len(piece) <= self.chunk_size:
                current_chunk += "\n" + piece
            else:
                merged_chunks.append(current_chunk)
                # Apply overlap by taking the trailing slice of the current chunk
                overlap_text = current_chunk[-self.chunk_overlap :] if self.chunk_overlap > 0 else ""
                current_chunk = (overlap_text + " " + piece).strip()

        if current_chunk:
            merged_chunks.append(current_chunk)

        return merged_chunks


def load_file(filepath: str) -> str:
    """Reads a file from disk."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


# ----------------------------------------------------------------------
# RUNNING THE LESSON DEMO
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("LESSON 1 DEMONSTRATION: CHUNKING IN ACTION")
    print("=" * 70)

    # 1. Load sample document
    sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "quantum_orbit_manual.md")
    sample_text = load_file(sample_path)
    total_tokens = count_tokens(sample_text)
    print(f"\n[+] Loaded document: {os.path.basename(sample_path)}")
    print(f"    Total characters: {len(sample_text)}")
    print(f"    Total tokens:     {total_tokens}")

    # 2. Show the flaw in Naive Splitting
    print("\n--- 1. Why Naive Character Splitting Fails ---")
    naive_chunks = naive_chunk_text(sample_text[:350], chunk_size=120)
    for i, c in enumerate(naive_chunks):
        print(f"  [Naive Chunk {i+1}] ({len(c)} chars): {repr(c)}")
    print("  -> Notice how words like 'Antimatter' or 'localized' get cut in half!")

    # 3. Show Recursive Splitting with Overlap
    print("\n--- 2. Recursive Semantic Splitting with Overlap ---")
    splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=70)
    semantic_chunks = splitter.split_text(sample_text)

    print(f"Created {len(semantic_chunks)} clean chunks.\n")
    for idx, chunk in enumerate(semantic_chunks[:3]):  # display first 3
        tokens = count_tokens(chunk)
        print(f"{'-'*50}\nCHUNK #{idx+1} | Chars: {len(chunk)} | Tokens: {tokens}\n{'-'*50}")
        print(chunk)

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS FOR STUDENTS:")
    print("1. Chunks must preserve natural semantic units (paragraphs, sentences).")
    print("2. Overlap ensures entities near boundary edges aren't severed.")
    print("3. Next step: Lesson 02 will convert these text chunks into mathematical vectors!")
    print("=" * 70)

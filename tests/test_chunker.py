"""
Unit tests for document chunker and splitter.
"""

import unittest
from src.chunker import RecursiveCharacterTextSplitter, DocumentChunk


class TestChunker(unittest.TestCase):
    def test_recursive_splitting_respects_size(self):
        text = "Paragraph 1 is here.\n\nParagraph 2 is slightly longer and contains more details.\n\nParagraph 3 is also here."
        splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=15)
        chunks = splitter.split_text(text)

        self.assertGreater(len(chunks), 1)
        for c in chunks:
            # Check chunks are roughly within desired size
            self.assertTrue(len(c) <= 90)

    def test_chunk_document_metadata(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        doc = "Heading\n\nThis is a sample document for testing chunk metadata attribution."
        chunks = splitter.chunk_document(doc, source_name="test_doc.md")

        self.assertGreater(len(chunks), 0)
        first_chunk = chunks[0]
        self.assertEqual(first_chunk.metadata["source"], "test_doc.md")
        self.assertEqual(first_chunk.metadata["chunk_index"], 1)
        self.assertIn("token_count", first_chunk.metadata)


if __name__ == "__main__":
    unittest.main()

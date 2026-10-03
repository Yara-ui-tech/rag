"""
Unit tests for PromptGenerator and anti-hallucination formatting.
"""

import unittest
from src.generator import PromptGenerator


class TestPromptGenerator(unittest.TestCase):
    def test_build_user_prompt_with_context(self):
        chunks = [
            {
                "chunk_id": "c1",
                "text": "The emergency warp limit is 52.3 THz.",
                "score": 0.95,
                "metadata": {"source": "quantum_manual.md"},
            }
        ]
        prompt = PromptGenerator.build_user_prompt(
            question="What is the warp limit?", retrieved_chunks=chunks
        )

        self.assertIn("quantum_manual.md", prompt)
        self.assertIn("52.3 THz", prompt)
        self.assertIn("What is the warp limit?", prompt)
        self.assertIn("[Source 1: quantum_manual.md", prompt)

    def test_build_user_prompt_empty(self):
        prompt = PromptGenerator.build_user_prompt(
            question="Unknown question?", retrieved_chunks=[]
        )
        self.assertIn("No reference documents found", prompt)


if __name__ == "__main__":
    unittest.main()

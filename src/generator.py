"""
Prompt generation and LLM interaction module using OpenAI Chat Completions.
"""

import os
from typing import List, Dict, Any, Optional, Generator
from dotenv import load_dotenv

load_dotenv()


class PromptGenerator:
    """Constructs grounded context prompts for RAG."""

    SYSTEM_PROMPT = (
        "You are an expert enterprise research assistant. "
        "Answer the user's question accurately and concisely, relying STRICTLY and ONLY "
        "on the provided Reference Context. "
        "Do NOT invent facts, extrapolate, or rely on outside speculation. "
        "If the answer cannot be determined from the provided context, state: "
        "'I do not have enough information in the provided documents to answer this question.' "
        "Always cite source documents using square brackets (e.g. [Source 1])."
    )

    @classmethod
    def build_user_prompt(cls, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Assembles retrieved chunks into formatted context blocks."""
        if not retrieved_chunks:
            return (
                f"No reference documents found.\n\n"
                f"User Question: {question}\n\n"
                f"Please respond stating that no relevant documents were found."
            )

        context_lines = []
        for i, chunk in enumerate(retrieved_chunks):
            source = chunk.get("metadata", {}).get("source", "Unknown Document")
            score = chunk.get("score", 0.0)
            context_lines.append(
                f"--- [Source {i + 1}: {source} | Similarity: {score:.3f}] ---\n"
                f"{chunk['text'].strip()}"
            )

        formatted_context = "\n\n".join(context_lines)

        return f"""Use the following reference documents to answer the question below.
Adhere strictly to the facts provided. Cite your sources.

REFERENCE CONTEXT:
{formatted_context}

QUESTION:
{question}

GROUNDED ANSWER (with citations):"""


class LLMGenerator:
    """Manages chat completions with OpenAI models (e.g. gpt-4o, gpt-4o-mini)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_CHAT_MODEL", model)
        self.temperature = temperature
        self._openai_client = None

        if self.api_key and not self.api_key.startswith("your-openai-api-key"):
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=self.api_key)
            except Exception as e:
                print(f"[!] Warning: Could not initialize OpenAI client: {e}")

    @property
    def is_live(self) -> bool:
        return self._openai_client is not None

    def generate(self, prompt: str) -> str:
        """Generates a complete answer from the LLM."""
        if not self.is_live:
            return (
                "[SIMULATED LLM RESPONSE]\n"
                "According to [Source 1], the answer has been extracted from the reference context. "
                "(Set OPENAI_API_KEY in .env for live GPT-4o output)"
            )

        response = self._openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": PromptGenerator.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """Streams the LLM response tokens as they arrive."""
        if not self.is_live:
            yield self.generate(prompt)
            return

        stream = self._openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": PromptGenerator.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

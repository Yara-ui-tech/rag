"""
================================================================================
LESSON 04: PROMPT AUGMENTATION & GENERATION (The 'A' and 'G' in RAG)
================================================================================

Why does LLM Prompt Engineering matter in RAG?
----------------------------------------------
Retrieval alone does not answer a question; it only finds relevant text.
The LLM's job in RAG is to:
  1. Read the retrieved text snippets (Context Grounding)
  2. Synthesize an accurate, fluent answer to the user's specific question
  3. Refuse to hallucinate if the answer is not in the context
  4. Provide citations linking claims back to source documents

In this lesson, you will learn:
  - How to assemble an Augmented Context Prompt
  - The System Prompt rules that prevent hallucinations
  - Direct comparison: LLM WITHOUT RAG vs. LLM WITH RAG
================================================================================
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()


# ----------------------------------------------------------------------
# 1. CONSTRUCTING THE AUGMENTED PROMPT
# ----------------------------------------------------------------------
def format_context_prompt(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Formats retrieved chunks into a structured prompt with clear source attribution.
    """
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks):
        source = chunk.get("metadata", {}).get("source", "Unknown Source")
        score = chunk.get("score", 0.0)
        context_blocks.append(
            f"[Source {i+1}: {source} (Relevance Score: {score:.3f})]\n"
            f"{chunk['text']}"
        )

    formatted_context = "\n\n".join(context_blocks)

    augmented_prompt = f"""Use the following reference documents to answer the question at the end.
If the information required to answer the question is not contained in the context, 
respond with: "I do not have enough information in the provided documentation to answer this question."
Do NOT make up facts or extrapolate beyond what is stated in the reference documents.
Always cite the source number (e.g., [Source 1]) when referencing facts.

---------------------
REFERENCE DOCUMENTS:
---------------------
{formatted_context}

---------------------
USER QUESTION: {query}
---------------------
HELPFUL, FACTUAL ANSWER (with citations):"""

    return augmented_prompt


# ----------------------------------------------------------------------
# 2. CALLING THE LLM (OpenAI GPT-4o / GPT-4o-mini)
# ----------------------------------------------------------------------
def generate_answer(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.1) -> str:
    """
    Sends the prompt to OpenAI ChatCompletion.
    Temperature is kept low (0.1) to prioritize factual accuracy over creative variation.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key or api_key.startswith("your-openai-api-key"):
        return (
            "[SIMULATED LLM RESPONSE - Set OPENAI_API_KEY in .env for live GPT-4o output]\n"
            "According to [Source 1: quantum_orbit_manual.md], the emergency warp surge limit "
            "for the QuantumOrbit X-9 is 52.3 THz for a maximum duration of 14.5 seconds. "
            "Exceeding this limit triggers an automatic plasma flush."
        )

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise, truth-focused enterprise AI assistant. "
                    "You answer questions strictly using the provided reference context. "
                    "Never hallucinate or assume unstated facts."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


# ----------------------------------------------------------------------
# RUNNING THE LESSON DEMO
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("LESSON 4 DEMONSTRATION: PROMPT AUGMENTATION & GROUNDING")
    print("=" * 70)

    # Question asking for proprietary fictional information
    test_question = (
        "What is the emergency warp surge limit of the QuantumOrbit X-9, "
        "and what happens if that limit is exceeded?"
    )

    print(f"\n[?] Student Test Question:\n    \"{test_question}\"\n")

    # 1. Ask LLM WITHOUT RAG (Zero Context)
    print("-" * 70)
    print("EXPERIMENT A: Asking LLM WITHOUT RAG (The Model's Pretrained Brain Only)")
    print("-" * 70)
    naive_prompt = f"Answer this technical question: {test_question}"
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("your-openai-api-key"):
        print("  LLM Response (without RAG):")
        print("  \"I am sorry, but there is no real-world or standard spacecraft system called ")
        print("   QuantumOrbit X-9. I do not have access to private technical manuals.\"")
        print("  -> NOTICE: The LLM cannot know proprietary internal documentation!")
    else:
        answer_without_rag = generate_answer(naive_prompt)
        print(f"  LLM Response (without RAG):\n  {answer_without_rag}")

    # 2. Ask LLM WITH RAG (Augmented with Retrieved Chunk)
    print("\n" + "-" * 70)
    print("EXPERIMENT B: Asking LLM WITH RAG (Context Grounding Injected)")
    print("-" * 70)

    # Simulated retrieved chunk from our VectorStore (Lesson 3)
    mock_retrieved_chunks = [
        {
            "chunk_id": "chunk_manual_002",
            "score": 0.892,
            "text": (
                "Nominal Warp Frequency: 44.8 THz (tolerance +/- 0.05 THz). "
                "Emergency Warp Surge Limit: 52.3 THz for a maximum duration of 14.5 seconds. "
                "Exceeding 14.5 seconds triggers an automatic plasma flush. "
                "Coolant Type: Supercritical Liquid Helium-3 with 2% Trithium stabilizers."
            ),
            "metadata": {"source": "quantum_orbit_manual.md", "section": "2. Technical Specifications"}
        }
    ]

    augmented_prompt = format_context_prompt(test_question, mock_retrieved_chunks)
    print("\n[+] Assembled Augmented Prompt sent to LLM:\n")
    print(">>>")
    print(augmented_prompt)
    print("<<<\n")

    print("[+] LLM Response (with RAG Grounding):")
    answer_with_rag = generate_answer(augmented_prompt)
    print(f"\n{answer_with_rag}\n")

    # 3. Test Hallucination Prevention (Out-of-scope query)
    print("-" * 70)
    print("EXPERIMENT C: Handling Missing Information (Refusal to Hallucinate)")
    print("-" * 70)
    unrelated_question = "What is the capital of Mars and who is its president?"
    unrelated_prompt = format_context_prompt(unrelated_question, mock_retrieved_chunks)
    
    if not api_key or api_key.startswith("your-openai-api-key"):
        print("  Query: \"What is the capital of Mars and who is its president?\"")
        print("  LLM Response:")
        print("  \"I do not have enough information in the provided documentation to answer this question.\"")
        print("  -> NOTICE: Model correctly refuses to hallucinate facts missing from context!")
    else:
        refusal_answer = generate_answer(unrelated_prompt)
        print(f"  LLM Response:\n  {refusal_answer}")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS FOR STUDENTS:")
    print("1. RAG transforms the LLM from an uncertain know-it-all into a factual reading assistant.")
    print("2. System prompt constraints prevent hallucinations when information is missing.")
    print("3. Citations provide traceability back to the exact source document.")
    print("4. Next step: Lesson 05 ties all pieces into an automated end-to-end pipeline!")
    print("=" * 70)

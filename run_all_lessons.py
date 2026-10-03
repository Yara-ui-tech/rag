"""
================================================================================
MASTER LESSON RUNNER & LEARNER ROADMAP
================================================================================
Runs all 5 educational lessons sequentially, printing explanations and 
allowing students to step through the curriculum one concept at a time.
================================================================================
"""

import os
import sys
import subprocess
import time

LESSONS = [
    {
        "num": "01",
        "file": "lessons/lesson_01_chunking.py",
        "title": "Document Loading & Chunking",
        "concept": "Why splitting documents matters, chunk sizes, overlap, and token counting.",
    },
    {
        "num": "02",
        "file": "lessons/lesson_02_embeddings.py",
        "title": "Embeddings & Vector Mathematics",
        "concept": "How semantic meaning is converted to geometry, dot products, and cosine similarity.",
    },
    {
        "num": "03",
        "file": "lessons/lesson_03_vector_store.py",
        "title": "Building a Vector Store from Scratch",
        "concept": "Storing chunks, metadata, top-K nearest-neighbor search, and disk persistence.",
    },
    {
        "num": "04",
        "file": "lessons/lesson_04_augmented_prompt.py",
        "title": "Prompt Augmentation & Grounding",
        "concept": "Context injection, system prompts, anti-hallucination guardrails, and citations.",
    },
    {
        "num": "05",
        "file": "lessons/lesson_05_complete_pipeline.py",
        "title": "End-to-End RAG Pipeline",
        "concept": "Assembling the full production-style system and querying real documents.",
    },
]


def run_lesson(idx: int, info: dict) -> bool:
    print("\n" + "#" * 78)
    print(f"# LESSON {info['num']}: {info['title'].upper()}")
    print(f"# CORE CONCEPT: {info['concept']}")
    print("#" * 78 + "\n")

    filepath = os.path.join(os.path.dirname(__file__), info["file"])
    if not os.path.exists(filepath):
        print(f"[!] Error: File {filepath} not found!")
        return False

    cmd = [sys.executable, filepath]
    result = subprocess.run(cmd)

    return result.returncode == 0


def main():
    print("=" * 78)
    print("     WELCOME TO THE RAG CURRICULUM: FIRST PRINCIPLES TO FULL SYSTEM")
    print("=" * 78)
    print("This course teaches you how RAG actually works under the hood:")
    for l in LESSONS:
        print(f"  [{l['num']}] {l['title']:<35} - {l['concept']}")
    print("=" * 78)

    for i, lesson in enumerate(LESSONS):
        success = run_lesson(i, lesson)
        if not success:
            print(f"\n[!] Lesson {lesson['num']} exited with an error. Halting.")
            sys.exit(1)

        if i < len(LESSONS) - 1:
            print(f"\n[OK] Lesson {lesson['num']} completed!")
            print("-" * 78)
            time.sleep(1)

    print("\n" + "=" * 78)
    print("CONGRATULATIONS! You have completed all 5 RAG lessons.")
    print("Next steps:")
    print("  1. Add your real OpenAI API key to .env")
    print("  2. Run the interactive workbench: python run_rag.py --interactive")
    print("  3. Add your own documents into the data/ directory and query them!")
    print("=" * 78)


if __name__ == "__main__":
    main()

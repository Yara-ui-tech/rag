"""
================================================================================
INTERACTIVE RAG ASSISTANT & RUNNER
================================================================================
Use this script to test, inspect, and run RAG queries against your documents.

Usage:
  1. Ask a single question:
     python run_rag.py --query "What is the warp surge limit?"

  2. Enter interactive chat mode:
     python run_rag.py --interactive

  3. Ingest custom files or change index:
     python run_rag.py --data ./data --interactive
================================================================================
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Ensure local src directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.pipeline import RAGPipeline

load_dotenv()


def print_banner():
    print("=" * 70)
    print("         * RAG TRAINING SUITE - INTERACTIVE WORKBENCH *")
    print("=" * 70)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("your-openai-api-key"):
        print("[!] NOTICE: Running in OFFLINE / SIMULATION mode.")
        print("    To enable live GPT-4o & text-embedding-3-small, set your OPENAI_API_KEY in .env")
    else:
        print("[+] STATUS: Connected to OpenAI API (text-embedding-3-small + gpt-4o)")
    print("=" * 70)


def format_response_display(result: dict, show_debug: bool = False):
    print("\n" + "-" * 70)
    print(f"[?] QUESTION: {result['question']}")
    print("-" * 70)
    print(f"[Timing] Retrieval: {result['retrieval_ms']:.1f}ms | Generation: {result['generation_ms']:.1f}ms | Total: {result['total_ms']:.1f}ms")
    
    print("\n--- [1] RETRIEVED CONTEXT SNIPPETS ---")
    for i, c in enumerate(result["retrieved_chunks"]):
        source = c["metadata"].get("source", "unknown")
        part = c["metadata"].get("chunk_index", 1)
        score = c["score"]
        snippet = c["text"][:140].replace("\n", " ")
        print(f"  [Source {i+1}] {source} (Part {part}) | Relevance Score: {score:+.4f}")
        print(f"    Excerpt: \"{snippet}...\"")

    print("\n--- [2] GROUNDED ANSWER (with citations) ---")
    print(result["answer"])

    if show_debug:
        print("\n--- [DEBUG] INJECTED PROMPT SENT TO LLM ---")
        print(result["prompt"])

    print("-" * 70 + "\n")


def interactive_loop(pipeline: RAGPipeline):
    print("\n[+] Entering Interactive Mode. Type your question, or commands:")
    print("    /debug   - Toggle raw prompt inspection")
    print("    /stats   - View vector store stats")
    print("    /exit    - Exit the program\n")

    show_debug = False

    while True:
        try:
            user_input = input("RAG > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "/exit", "/quit"]:
            print("Exiting RAG Workbench. Happy coding!")
            break

        if user_input.lower() == "/debug":
            show_debug = not show_debug
            print(f"[*] Debug mode: {'ON (Full prompt display)' if show_debug else 'OFF'}")
            continue

        if user_input.lower() == "/stats":
            print(f"[*] Chunks in Vector Store: {len(pipeline.vector_store)}")
            print(f"[*] Top-K retrieval setting: {pipeline.top_k}")
            continue

        result = pipeline.query(user_input)
        format_response_display(result, show_debug=show_debug)


def main():
    parser = argparse.ArgumentParser(description="RAG Interactive Workbench & Runner")
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data"),
        help="Directory containing documents to ingest",
    )
    parser.add_argument(
        "--index",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "storage", "rag_index.json"),
        help="Path to save or load the vector store index JSON",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a single question and exit",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve per question",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive conversational loop",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-indexing even if saved index exists",
    )

    args = parser.parse_args()
    print_banner()

    index_exists = os.path.exists(args.index)

    if index_exists and not args.reindex:
        pipeline = RAGPipeline(top_k=args.top_k, index_file=args.index)
    else:
        pipeline = RAGPipeline(top_k=args.top_k, index_file=args.index)
        print(f"[+] Ingesting documents from: {args.data}")
        count = pipeline.ingest_directory(args.data, pattern="*.*")
        print(f"[OK] Ingested and indexed {count} chunks into {args.index}")

    if args.query:
        result = pipeline.query(args.query)
        format_response_display(result)
    elif args.interactive or len(sys.argv) == 1:
        interactive_loop(pipeline)


if __name__ == "__main__":
    main()

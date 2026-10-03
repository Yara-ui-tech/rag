"""
================================================================================
RAG TASK SOLVER & STEP-BY-STEP TERMINAL
================================================================================
Paste any RAG task or exercise, and this terminal will generate:
  1. Architecture & Strategy
  2. Step-by-Step Execution Plan
  3. Complete, Runnable Python Code Solution
  4. Verification & Testing Code
  5. Common Pitfalls & How to Avoid Them

Usage:
  python task_solver.py
  (Type or paste your task, then press Enter twice or type /solve)
================================================================================
"""

import os
import sys
from dotenv import load_dotenv

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure local src directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.task_solver import TaskSolver

load_dotenv()


def print_terminal_banner():
    print("\n" + "=" * 76)
    print("      [*] RAG TASK SOLVER & STEP-BY-STEP SOLUTION ENGINE [*]")
    print("=" * 76)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("your-openai-api-key"):
        print("[!] Mode: Offline Intelligent Solver (Set OPENAI_API_KEY in .env for custom GPT-4o)")
    else:
        print("[+] Mode: Connected to OpenAI GPT-4o Task Architect")
    print("=" * 76)
    print("Instructions:")
    print("  - Paste or type your RAG task / question below.")
    print("  - When done, type '/solve' or press Enter on an empty line.")
    print("  - Type 'exit' or 'quit' to close.")
    print("=" * 76 + "\n")


def interactive_terminal():
    print_terminal_banner()
    solver = TaskSolver()

    while True:
        print("+--[Enter Your Task] (Paste multiple lines, type '/solve' to run):")
        lines = []
        while True:
            try:
                line = input("| ")
            except (KeyboardInterrupt, EOFError):
                print("\n\nExiting Task Solver. Happy building!")
                return

            if line.strip() in ["/solve", "/run", "DONE"]:
                break
            if line.strip().lower() in ["exit", "quit"]:
                print("\nExiting Task Solver. Happy building!")
                return
            if line == "" and len(lines) > 0 and lines[-1] == "":
                # Two empty lines means finish
                break

            lines.append(line)

        task_text = "\n".join(lines).strip()
        if not task_text:
            print("+-- [!] No task entered. Please type or paste a task.\n")
            continue

        print("\n" + "-" * 76)
        print("[*] Analyzing task & generating step-by-step solution...")
        print("-" * 76 + "\n")

        result = solver.solve(task_text)

        print("\n" + "#" * 76)
        print("#                      GENERATED SOLUTION & STEPS                         #")
        print("#" * 76 + "\n")
        print(result["solution_markdown"])
        print("\n" + "=" * 76)
        print("[OK] Task solution complete! You can copy the code above and run it.")
        print("=" * 76 + "\n")


if __name__ == "__main__":
    interactive_terminal()

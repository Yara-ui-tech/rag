"""
Production WSGI Server Entrypoint for RAG Academy.
Auto-detects the operating system and serves the application using
a production-grade WSGI server (Waitress on Windows/cross-platform, Gunicorn on Linux).
"""

import os
import sys
import platform

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure local directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import app, get_pipeline


def run_production_server():
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5050))
    threads = int(os.environ.get("THREADS", 6))

    # Pre-warm the RAG pipeline and vector store
    print("=" * 76)
    print("      [*] STARTING RAG TRAINING STUDIO (PRODUCTION WSGI SERVER) [*]")
    print("=" * 76)
    print(f"[*] Binding to: http://{host}:{port}")
    print("[*] Pre-warming RAG Vector Store & Ingestion Pipeline...")
    pipe = get_pipeline()
    print(f"[OK] In-memory Vector Store active ({len(pipe.vector_store)} chunks ready)")
    print("=" * 76)

    # Use Waitress for production serving
    try:
        from waitress import serve
        print(f"[+] Server Engine: Waitress WSGI (Threads: {threads})")
        print(f"[+] Access the UI at: http://localhost:{port}")
        print("=" * 76 + "\n")
        serve(app, host=host, port=port, threads=threads)
    except ImportError:
        # Fallback to standard Flask if waitress is not installed
        print("[!] Notice: Waitress not found, falling back to basic runner.")
        app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    run_production_server()

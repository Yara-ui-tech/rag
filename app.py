"""
RAG Training Suite - Interactive Web Application Server
Flask backend connecting the RAG curriculum with a modern web UI.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from dotenv import load_dotenv, set_key

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.chunker import RecursiveCharacterTextSplitter, DocumentChunk, load_document, TokenCounter
from src.embedder import EmbeddingClient, normalize_vector
from src.vector_store import VectorStore
from src.generator import PromptGenerator, LLMGenerator
from src.pipeline import RAGPipeline
from src.task_solver import TaskSolver

load_dotenv()

# Base Directories
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Initialize Flask with explicit absolute paths
app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    static_url_path="/static",
    template_folder=TEMPLATES_DIR
)

# Detect Serverless (e.g. Vercel, AWS Lambda) where root filesystem is read-only
IS_SERVERLESS = bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))
if IS_SERVERLESS:
    STORAGE_DIR = "/tmp/storage"
else:
    STORAGE_DIR = os.path.join(BASE_DIR, "storage")

INDEX_PATH = os.path.join(STORAGE_DIR, "rag_index.json")
ENV_PATH = os.path.join(BASE_DIR, ".env")

try:
    os.makedirs(STORAGE_DIR, exist_ok=True)
except OSError:
    pass

try:
    os.makedirs(DATA_DIR, exist_ok=True)
except OSError:
    pass

# Global Pipeline Instance
pipeline = None


def get_pipeline() -> RAGPipeline:
    global pipeline
    if pipeline is None:
        pipeline = RAGPipeline(
            chunk_size=400,
            chunk_overlap=80,
            top_k=3,
            index_file=INDEX_PATH,
        )
        # If vector store is empty, ingest data directory
        if len(pipeline.vector_store) == 0:
            print("[+] Ingesting initial documents into vector store...")
            pipeline.ingest_directory(DATA_DIR)
    return pipeline


START_TIME = time.time()


@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response


# ----------------------------------------------------------------------
# Health Check Endpoint (For Docker & Kubernetes / Cloud Load Balancers)
# ----------------------------------------------------------------------
@app.route("/health")
@app.route("/api/health")
def healthcheck():
    pipe = get_pipeline()
    return jsonify({
        "status": "healthy",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "total_chunks": len(pipe.vector_store),
        "version": "1.0.0",
    }), 200


# ----------------------------------------------------------------------
# Web UI Page Route
# ----------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ----------------------------------------------------------------------
# System Status & Config Endpoints
# ----------------------------------------------------------------------
@app.route("/api/status", methods=["GET"])
def get_status():
    pipe = get_pipeline()
    api_key = os.getenv("OPENAI_API_KEY", "")
    has_valid_key = bool(api_key and not api_key.startswith("your-openai-api-key") and len(api_key) > 20)

    # List files in data/
    docs = []
    if os.path.exists(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            ext = os.path.splitext(f)[1].lower()
            if ext in [".md", ".txt", ".rst"]:
                fpath = os.path.join(DATA_DIR, f)
                docs.append({
                    "name": f,
                    "size_bytes": os.path.getsize(fpath),
                })

    return jsonify({
        "status": "ready",
        "has_api_key": has_valid_key,
        "chat_model": os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        "total_chunks": len(pipe.vector_store),
        "documents": docs,
        "top_k": pipe.top_k,
    })


@app.route("/api/settings", methods=["POST"])
def update_settings():
    data = request.json or {}
    api_key = data.get("api_key")
    chat_model = data.get("chat_model")
    top_k = data.get("top_k")

    if api_key is not None:
        os.environ["OPENAI_API_KEY"] = api_key.strip()
    if chat_model:
        os.environ["OPENAI_CHAT_MODEL"] = chat_model.strip()

    global pipeline
    # Reinitialize pipeline with new settings
    pipe = get_pipeline()
    if top_k is not None:
        pipe.top_k = int(top_k)

    # Re-init LLM generator and embedder if key changed
    pipe.embedder = EmbeddingClient()
    pipe.generator = LLMGenerator(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))

    return jsonify({"success": True, "message": "Settings updated successfully!"})


# ----------------------------------------------------------------------
# Step 1: Chunking Lab Endpoints
# ----------------------------------------------------------------------
@app.route("/api/chunk-preview", methods=["POST"])
def preview_chunking():
    data = request.json or {}
    text = data.get("text", "")
    chunk_size = int(data.get("chunk_size", 400))
    chunk_overlap = int(data.get("chunk_overlap", 80))

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    counter = TokenCounter()

    results = []
    for idx, c in enumerate(chunks):
        results.append({
            "index": idx + 1,
            "text": c,
            "chars": len(c),
            "tokens": counter.count(c),
        })

    return jsonify({
        "total_chars": len(text),
        "total_tokens": counter.count(text),
        "chunk_count": len(results),
        "chunks": results,
    })


# ----------------------------------------------------------------------
# Step 2: Vector Math Lab Endpoints
# ----------------------------------------------------------------------
@app.route("/api/vector-math", methods=["POST"])
def compute_vector_math():
    data = request.json or {}
    sentences = data.get("sentences", [])
    if len(sentences) < 2:
        return jsonify({"error": "Please provide at least 2 sentences."}), 400

    pipe = get_pipeline()
    embeddings = pipe.embedder.embed_batch(sentences)

    # Compute pairwise cosine similarities
    n = len(sentences)
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            sim = float(np.dot(embeddings[i], embeddings[j]))
            row.append(round(sim, 4))
        matrix.append(row)

    vectors_preview = []
    for i, vec in enumerate(embeddings):
        # Return first 10 dimensions for visualization
        vectors_preview.append({
            "sentence": sentences[i],
            "dimension": len(vec),
            "norm": float(np.linalg.norm(vec)),
            "first_ten": [round(float(x), 4) for x in vec[:10]],
        })

    return jsonify({
        "sentences": sentences,
        "matrix": matrix,
        "vectors_preview": vectors_preview,
    })


# ----------------------------------------------------------------------
# Step 3: Vector Store Inspection & Retrieval
# ----------------------------------------------------------------------
@app.route("/api/chunks", methods=["GET"])
def get_all_chunks():
    pipe = get_pipeline()
    chunks = []
    for c in pipe.vector_store.chunks:
        chunks.append({
            "chunk_id": c.chunk_id,
            "text": c.text,
            "metadata": c.metadata,
        })
    return jsonify({"total": len(chunks), "chunks": chunks})


@app.route("/api/search", methods=["POST"])
def search_chunks():
    data = request.json or {}
    query = data.get("query", "").strip()
    top_k = int(data.get("top_k", 3))

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    pipe = get_pipeline()
    query_vec = pipe.embedder.embed_text(query)
    results = pipe.vector_store.search(query_vec, top_k=top_k)

    return jsonify({
        "query": query,
        "top_k": top_k,
        "results": results,
    })


# ----------------------------------------------------------------------
# Step 4 & 5: Full RAG Query & Prompt Inspection
# ----------------------------------------------------------------------
@app.route("/api/query", methods=["POST"])
def execute_rag_query():
    data = request.json or {}
    question = data.get("question", "").strip()
    include_rag = data.get("include_rag", True)

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    pipe = get_pipeline()

    if not include_rag:
        # LLM Generation WITHOUT RAG (Direct baseline test)
        t0 = time.time()
        naive_prompt = f"Answer this question as accurately as possible:\n\n{question}"
        answer = pipe.generator.generate(naive_prompt)
        gen_time = (time.time() - t0) * 1000
        return jsonify({
            "question": question,
            "answer": answer,
            "retrieved_chunks": [],
            "prompt": naive_prompt,
            "is_rag": False,
            "retrieval_ms": 0.0,
            "generation_ms": round(gen_time, 2),
            "total_ms": round(gen_time, 2),
        })

    # Full RAG Pipeline execution
    result = pipe.query(question)
    return jsonify({
        "question": result["question"],
        "answer": result["answer"],
        "retrieved_chunks": result["retrieved_chunks"],
        "prompt": result["prompt"],
        "is_rag": True,
        "retrieval_ms": round(result["retrieval_ms"], 2),
        "generation_ms": round(result["generation_ms"], 2),
        "total_ms": round(result["total_ms"], 2),
    })


# ----------------------------------------------------------------------
# Task Solver Endpoint (Solves arbitrary tasks with steps & code)
# ----------------------------------------------------------------------
@app.route("/api/solve-task", methods=["POST"])
def solve_task():
    data = request.json or {}
    task_text = data.get("task", "").strip()

    if not task_text:
        return jsonify({"error": "Task text cannot be empty."}), 400

    solver = TaskSolver()
    result = solver.solve(task_text)
    return jsonify(result)


# ----------------------------------------------------------------------
# Document Re-indexing Endpoint
# ----------------------------------------------------------------------
@app.route("/api/reindex", methods=["POST"])
def reindex_documents():
    pipe = get_pipeline()
    pipe.vector_store = VectorStore()  # Reset in-memory store
    if os.path.exists(INDEX_PATH):
        try:
            os.remove(INDEX_PATH)
        except Exception:
            pass

    count = pipe.ingest_directory(DATA_DIR)
    return jsonify({
        "success": True,
        "message": f"Successfully re-indexed {count} chunks into VectorStore.",
        "total_chunks": len(pipe.vector_store),
    })


@app.route("/api/document-content", methods=["GET"])
def get_doc_content():
    filename = request.args.get("name")
    if not filename:
        return jsonify({"error": "Filename required"}), 400
    safe_path = os.path.join(DATA_DIR, os.path.basename(filename))
    if not os.path.exists(safe_path):
        return jsonify({"error": "File not found"}), 404
    with open(safe_path, "r", encoding="utf-8") as f:
        content = f.read()
    return jsonify({"name": filename, "content": content})


if __name__ == "__main__":
    # Start the Flask app on 127.0.0.1:5050
    port = int(os.environ.get("PORT", 5050))
    print(f"\n[+] RAG Training UI starting at: http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False)

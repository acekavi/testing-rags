KN Assessment

    Build a Production-Style RAG Q&A API (Core)

Time: 90–120 mins
Stack: Python, FastAPI, LangChain, any LLM

Requirements

    Ingest documents from ./data/ (txt/pdf optional)

    Chunking with overlap + metadata (doc_name, page, chunk_id)

    Create embeddings and store in a vector DB:

    local: FAISS / Chroma

    Retriever with configurable top_k

    Add RAG chain with:

    “answer only from context; if not found say I don’t know”

    return citations (doc + chunk ids)

    Expose APIs:

    POST /ingest → builds index

    POST /ask → { question, top_k }

Output JSON

{

"answer": "...",

"sources": [

    {"doc":"a.txt","chunk_id":2,"score":0.81,"snippet":"..."}

]

}

Must-have quality checks (what you grade)

    Clean structure: services/, routes/, config.py

    Deterministic chunking + metadata

    Handles “no relevant docs”

    Citations match retrieved chunks (no fake sources)

    Config via env vars (model name, chunk size, top_k)





    RAG Improvement Task

Time: 30–45 mins
Pick ONE to implement:

Option A: Hybrid Search

    Combine BM25 (keyword) + vector similarity

    Merge and rerank results

Option B: Reranking (Cross-Encoder)

    Use a cross-encoder reranker (sentence-transformers cross-encoder)

    Apply rerank after initial vector top_k=20 → rerank to final top_k=5

Option C: MMR (Diversity)

    Implement MMR retrieval to reduce duplicate chunks

Evaluation

    Candidate explains trade-offs: speed vs accuracy, cost, latency

    Shows measurable improvement (even simple logging is fine)





    Guardrailed Structured Extraction (LLM + Validation)

Time: 45–60 mins
Goal: Given messy text, output validated JSON.

Requirements

    Define Pydantic schema (invoice/receipt schema)

    LangChain prompt enforces JSON only

    Validate output; if fails → auto-retry with corrective prompt

    Return:

{

"data": {...},

"validation_passed": true,

"retries": 1

}

Evaluation

    Schema-first approach

    Retry loop + safe parsing

    No crashing on missing fields

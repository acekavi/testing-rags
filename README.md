To switch models:

```bash
# Pull the model
docker exec -it rag-ollama ollama pull llama3.2

# Update .env
LLM_MODEL_NAME=llama3.2
```

## Testing the API

```bash
# 1. Ingest documents
curl -X POST http://localhost:8080/ingest

# 2. Ask a question
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the return policy?"}'
```

## Key Requirements Met

1. **Chunking**: Deterministic chunking with overlap and metadata (doc_name, page, chunk_id)
2. **RAG Chain**: Answers only from context; says "I don't know" if not found
3. **Citations**: Returns citations matching retrieved chunks (no fake sources)
4. **Error Handling**: Handles "no relevant docs" gracefully
5. **Config via env vars**: All settings configurable
6. **100% Local**: No API keys, no cloud dependencies, data stays private

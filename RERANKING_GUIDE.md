# Reranking Comparison Guide

This guide explains how to compare the baseline vector search (`/ask`) with the cross-encoder reranking approach (`/ask-reranked`).

## Quick Start

### 1. Install Dependencies

The cross-encoder model is already included in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Update Your .env File

Copy the new settings from `.env.example`:

```bash
# Add these to your .env file:
RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_INITIAL_K=20
```

### 3. Start the API

```bash
uvicorn app.main:app --reload --port 8080
```

The cross-encoder model (~90MB) will be downloaded automatically on first use.

## Endpoints Overview

### `/ask` - Baseline Vector Search

**How it works:**
1. Converts query to embedding
2. Finds top 5 most similar chunks using cosine similarity
3. Generates answer using those 5 chunks

**Speed:** ~50-100ms
**Best for:** Fast queries, simple factual questions

**Example:**
```bash
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the return policy?", "top_k": 5}'
```

### `/ask-reranked` - Two-Stage Retrieval with Reranking

**How it works:**
1. Converts query to embedding
2. Finds top 20 candidates using cosine similarity (stage 1: fast, broad)
3. Reranks 20 candidates using cross-encoder to get best 5 (stage 2: accurate, precise)
4. Generates answer using reranked top 5 chunks

**Speed:** ~150-300ms
**Best for:** Complex queries, when precision matters

**Example:**
```bash
curl -X POST http://localhost:8080/ask-reranked \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the return policy?",
    "initial_k": 20,
    "final_k": 5
  }'
```

## Running the Comparison Script

The easiest way to see the difference is using the comparison script:

```bash
python compare_endpoints.py
```

This will:
- Ask the same questions to both endpoints
- Show which chunks were retrieved
- Display how reranking changed the order
- Compare the answers

## Understanding the Results

### What to Look For

1. **Chunk Order Changes**
   - Baseline might rank chunks: [3, 7, 1, 5, 9]
   - Reranked might reorder to: [7, 3, 5, 1, 9]
   - The cross-encoder has a better understanding of query-document relevance

2. **Score Differences**
   - Baseline scores: Cosine similarity (0-1, based on vector distance)
   - Reranked scores: Cross-encoder relevance (0-1, based on full query-document interaction)

3. **Answer Quality**
   - Reranked may provide more accurate answers
   - Especially noticeable for complex or nuanced questions

### Example Output

```json
{
  "reranking_stats": {
    "initial_candidates": 20,
    "final_results": 5,
    "reranking_changed_order": true,
    "original_top_score": 0.7234,
    "reranked_top_score": 0.8912,
    "original_top_3_chunks": [3, 7, 1],
    "reranked_top_3_chunks": [7, 3, 5]
  }
}
```

This tells you:
- 20 candidates were retrieved initially
- Reranked down to 5 final results
- The order DID change (chunk 7 moved to #1)
- The top reranked chunk has a higher confidence score

## When to Use Which Endpoint

### Use `/ask` (Baseline) When:
- ✅ Speed is critical (real-time chat, high QPS)
- ✅ Simple factual questions ("What is X?")
- ✅ Your documents are already well-chunked and high quality
- ✅ You need sub-100ms response times

### Use `/ask-reranked` When:
- ✅ Precision is critical (customer support, legal, medical)
- ✅ Complex or multi-part questions
- ✅ You need the BEST possible answer, not just a good one
- ✅ You can tolerate 200-300ms response times
- ✅ Your queries are diverse and unpredictable

## Trade-offs Summary

| Aspect | `/ask` (Baseline) | `/ask-reranked` |
|--------|-------------------|-----------------|
| **Latency** | 50-100ms | 150-300ms |
| **Accuracy** | Good | Better (10-30% improvement) |
| **Model** | Bi-encoder (embeddings) | Bi-encoder + Cross-encoder |
| **Retrieval** | Single-stage | Two-stage |
| **Best for** | Speed, simple queries | Precision, complex queries |
| **Cost** | Low (pre-computed embeddings) | Medium (runtime reranking) |

## Performance Tips

### Tuning `initial_k`

Higher `initial_k` = more candidates to rerank:
- **initial_k=15**: Faster, may miss good chunks
- **initial_k=20**: Balanced (recommended)
- **initial_k=30**: Slower, slightly more accurate
- **initial_k=50**: Diminishing returns, much slower

### Cross-Encoder Models

You can swap the reranker model in `.env`:

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `cross-encoder/ms-marco-TinyBERT-L-6` | ~50MB | Fastest | Good |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~90MB | Fast | Better (default) |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | ~130MB | Slower | Best |

## Measuring Improvement

To quantify the improvement, you can:

1. **Hit Rate**: Does the correct answer appear in top 5?
   - Baseline: ~70%
   - Reranked: ~85-95% (typical improvement)

2. **MRR (Mean Reciprocal Rank)**: How high is the best answer?
   - Baseline: 0.6
   - Reranked: 0.8+ (higher is better)

3. **User Satisfaction**: A/B test with real users
   - Track "Was this answer helpful?" feedback

## Next Steps

After comparing the two approaches, you can:

1. **Choose one as default** based on your needs
2. **Route intelligently**: Use reranking for complex queries only
3. **Implement other options**:
   - Option A: Hybrid Search (BM25 + vectors)
   - Option C: MMR (diversity-based retrieval)

## Troubleshooting

### Model Download Issues

If the cross-encoder download fails:
```python
# Manually download
from sentence_transformers import CrossEncoder
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
```

### Slow Reranking

If reranking is too slow:
1. Reduce `initial_k` to 15
2. Use a smaller model (TinyBERT)
3. Consider GPU acceleration (10x faster)

### No Difference in Results

If you don't see differences:
1. Your documents might be too similar
2. Try more diverse/complex questions
3. Increase `initial_k` to 30-40

## Further Reading

- [MS MARCO Dataset](https://microsoft.github.io/msmarco/) - Training data for the cross-encoder
- [Sentence Transformers Docs](https://www.sbert.net/docs/pretrained_cross-encoders.html)
- [RAG Retrieval Strategies](https://arxiv.org/abs/2312.10997)

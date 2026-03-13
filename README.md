# DaveRAG

Production-oriented RAG pipeline for the `dave_data.json` dataset. The repository treats each Q/A pair as one retrieval unit, normalizes it into a stable internal schema, builds embeddings, runs hybrid retrieval plus lightweight reranking, and serves grounded answers with citations through FastAPI.

## Dataset format

`dave_data.json` is a JSON array with 197 records. Each record currently follows this shape:

```json
{
  "source_id": "restaurant-history-114",
  "title": "What vegan hacks do some customers attempt?",
  "text": "Question: What vegan hacks do some customers attempt at Dave's Hot Chicken?\nAnswer: Some customers try ordering cauliflower without breading or serving it on plain bread, though cross-contact risks remain.",
  "metadata": {
    "topic": "restaurant-history",
    "type": "qa"
  }
}
```

The ingestion pipeline parses each `text` field into a normalized document with:

- `question`
- `answer`
- `text_for_embedding`
- `text_for_generation`
- `keywords`
- stable metadata fields such as `topic`, `document_type`, `source_id`, `version`, and `is_active`

## Architecture

- `src/daverag/data.py`: dataset loading and validation
- `src/daverag/normalization.py`: Q/A parsing, normalization, keyword extraction
- `src/daverag/embeddings.py`: pluggable embedding backends
- `src/daverag/index.py`: persisted document and embedding index
- `src/daverag/retrieval.py`: hybrid vector plus BM25 retrieval and reranking
- `src/daverag/generation.py`: grounded answer generation backends
- `src/daverag/service.py`: orchestration layer
- `src/daverag/api.py`: FastAPI app
- `src/daverag/cli.py`: local indexing and query CLI

## Retrieval flow

1. Load and validate the raw JSON.
2. Parse every Q/A pair into a normalized document.
3. Generate `text_for_embedding` and `keywords`.
4. Embed each document.
5. Persist normalized documents and vectors.
6. On query, embed the question.
7. Run hybrid retrieval:
   - vector similarity over the embedding matrix
   - BM25 keyword scoring over normalized text
8. Rerank top candidates with lexical overlap and title/keyword boosts.
9. Generate a grounded answer from the reranked context.
10. Return answer, confidence, insufficiency flag, sources, and raw retrieved chunks.

The service also adds:

- lightweight query classification for topic-aware filtering
- in-memory response caching for repeated queries
- offline evaluation with retrieval and answer hit-rate metrics

## Running locally

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Build the index:

```bash
python3 -m daverag.cli index
```

Query the pipeline:

```bash
python3 -m daverag.cli ask "Who founded Dave's Hot Chicken?"
```

Run the starter evaluation set:

```bash
python3 -m daverag.cli eval --file eval/sample_eval.json
```

Run the API:

```bash
uvicorn daverag.api:app --reload
```

Open the local frontend:

```text
http://127.0.0.1:8000/
```

The frontend shows:

- animated answer rendering
- confidence and insufficient-context flags
- citations used in the final answer
- retrieved chunks with vector, keyword, rerank, and final scores

## Configuration

Defaults are in `.env.example`.

- `EMBEDDING_BACKEND=local` uses a deterministic local hash embedding fallback.
- `EMBEDDING_BACKEND=openai` uses `text-embedding-3-small` or another configured OpenAI embedding model.
- `GENERATION_BACKEND=extractive` returns a grounded extractive answer without an LLM call.
- `GENERATION_BACKEND=openai` uses the OpenAI Responses API for answer generation.

## Production notes

- The local embedding backend makes the repo runnable without credentials, but production should use a real embedding model.
- The current implementation persists vectors to the filesystem. In production, replace this with Pinecone, pgvector, Qdrant, or OpenAI vector stores.
- The answer backend is intentionally strict and supports insufficient-context fallback.
- The data model uses stable `source_id` values to support versioned updates and deletes.

## Tests

```bash
pytest
```

# Semantic Search Studio

LLM-powered **semantic search** over your own documents. Upload PDFs, Markdown,
or text; ask questions in plain English; get a **grounded, cited answer**
synthesized by Claude from the most relevant passages in your corpus.

> A portfolio project demonstrating a modern Retrieval-Augmented Generation
> (RAG) pipeline end to end: ingestion → chunking → local embeddings → vector
> search → LLM answer synthesis with citations.

![architecture](https://img.shields.io/badge/stack-FastAPI%20%2B%20React%20%2B%20Claude-6ee7b7)

## How it works

```
            ┌─────────────┐   embed    ┌───────────┐
 Upload ──▶ │  Ingestion  │ ─────────▶ │  ChromaDB │
 (PDF/MD)   │ parse+chunk │  (local)   │  vectors  │
            └─────────────┘            └─────┬─────┘
                                             │ top-k by cosine
 Question ─────────── embed query ───────────┤
                                             ▼
                                      ┌──────────────┐
                                      │   Claude     │  grounded,
                                      │ claude-opus  │  cited answer
                                      │    -4-8      │  ───────────▶  UI
                                      └──────────────┘
```

- **Embeddings run locally** via `sentence-transformers` (`all-MiniLM-L6-v2`) —
  no API key needed to index, no per-call cost.
- **Only the final answer** calls the Claude API, so quality is visible exactly
  where it matters.
- Every answer cites `[n]` markers tied to the retrieved passages, which the UI
  displays so answers are verifiable rather than black-box.

See [`constitution/`](./constitution) for the project **mission**,
**tech-stack**, and **roadmap**.

## Project structure

```
llm-semantic-search/
├── constitution/        # mission.md, tech-stack.md, roadmap.md
├── backend/             # FastAPI + embeddings + ChromaDB + Claude
│   ├── app/             # application modules
│   ├── tests/           # pytest unit + integration suite
│   └── Dockerfile
├── frontend/            # React + Vite UI
│   ├── src/
│   ├── nginx.conf       # serves the SPA + proxies /api
│   └── Dockerfile
├── scripts/
│   └── smoke_test.sh    # live end-to-end check for a running stack
└── docker-compose.yml   # one-command startup
```

## Prerequisites

- Python 3.11+
- Node.js 18+
- An Anthropic API key (only used for answer synthesis)
- (Optional) Docker + Docker Compose for one-command startup

## Quick start with Docker (recommended)

One command builds and runs the whole stack (backend + frontend + nginx proxy):

```bash
cp .env.example .env          # then put your ANTHROPIC_API_KEY in .env
docker compose up --build
```

Then open <http://localhost:8080>. The frontend serves the UI and proxies
`/api` to the backend, so everything is on one origin. The backend API and its
docs are also exposed directly at <http://localhost:8000/docs>.

Data persists across restarts via named volumes (`chroma-data` for the index,
`model-cache` for the downloaded embedding model). Tear down with
`docker compose down` (add `-v` to also wipe the index and model cache).

## Manual setup (without Docker)

### 1. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # then add your ANTHROPIC_API_KEY
uvicorn app.main:app --reload    # serves on http://localhost:8000
```

Interactive API docs: <http://localhost:8000/docs>

> The embedding model (~90 MB) downloads on first ingest/search and is cached.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev                      # serves on http://localhost:5173
```

Open <http://localhost:5173>, drop in a document, and ask a question.

## API

| Method | Path          | Description                                  |
|--------|---------------|----------------------------------------------|
| POST   | `/api/ingest` | Upload + index a document (multipart `file`) |
| POST   | `/api/search` | `{ "query": "...", "top_k": 5 }` → cited answer |
| GET    | `/api/stats`  | Indexed documents and chunk counts           |
| POST   | `/api/reset`  | Clear the index                              |
| GET    | `/api/health` | Liveness probe                               |

## Performance metrics

Every `/api/search` response carries a `metrics` object so the system's speed
and cost are visible, not implicit. The pipeline times each stage and captures
Claude's token usage:

```jsonc
"metrics": {
  "embed_ms": 8.4,         // local query embedding
  "retrieve_ms": 12.1,     // ChromaDB cosine search
  "synthesize_ms": 2140.7, // Claude answer synthesis (dominates)
  "total_ms": 2161.2,
  "input_tokens": 1240,
  "output_tokens": 180,
  "estimated_cost_usd": 0.0107,
  "top_score": 0.83        // similarity of the best-matching passage
}
```

The UI renders this as a footer under each answer. It makes the cost/latency
profile of RAG concrete: **the Claude call dominates** end-to-end time, while
local embedding and vector retrieval are sub-second and free.

Cost is estimated from per-million-token prices in
[`backend/app/config.py`](./backend/app/config.py) (`input_price_per_mtok` /
`output_price_per_mtok`, defaulting to `claude-opus-4-8` at \$5 in / \$25 out) —
update them if you switch models.

## Testing

Two layers of verification: a fast offline **unit/integration suite** and a
live **smoke test** against a running stack.

### Unit & integration tests (pytest)

Runs fully offline — no API key, no network, no model download. The embedding
model and the Claude call are stubbed; ChromaDB runs against a fresh temp
directory per test, so the real ingest → chunk → index → retrieve → answer
pipeline and every HTTP route are exercised.

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

Expected: **23 passed**. Coverage:

| File | What it checks |
|---|---|
| `tests/test_ingestion.py` | Text extraction (txt/md/unsupported) and overlap-aware chunking |
| `tests/test_vectorstore.py` | ChromaDB add/query/stats/reset, similarity ordering, metadata |
| `tests/test_api.py` | All endpoints + the full search flow, validation, and error paths |

### Smoke test (live stack)

After `docker compose up`, verify the running system end-to-end:

```bash
./scripts/smoke_test.sh                      # tests http://localhost:8080 (via nginx)
./scripts/smoke_test.sh http://localhost:8000  # or hit the backend directly
```

It checks health → ingest → stats → search. The search step is reported as
SKIPPED (not failed) if `ANTHROPIC_API_KEY` isn't configured, so the script is
useful even before you add a key.

## Design notes

- **Grounding first.** The synthesis prompt instructs Claude to answer *only*
  from the retrieved passages and to admit when the context is insufficient —
  preventing confident hallucinations.
- **Separation of concerns.** Embedding, retrieval, and generation are
  independent modules (`embeddings.py`, `vectorstore.py`, `llm.py`), so any
  layer can be swapped or tested in isolation.
- **Adaptive thinking.** Synthesis uses Claude's adaptive thinking at medium
  effort — a balance of reasoning quality and latency for a search UX.

## Roadmap highlights

Hybrid (dense + keyword) retrieval, reranking, streaming answers, and a
Dockerized one-command deploy are tracked in
[`constitution/roadmap.md`](./constitution/roadmap.md).

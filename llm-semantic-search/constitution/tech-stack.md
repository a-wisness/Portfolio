# Tech Stack

The stack is chosen to showcase applied ML/AI engineering while staying easy to
run locally with a single API key.

## Backend — Python 3.11+

| Concern | Choice | Why |
|---|---|---|
| Web framework | **FastAPI** | Async, typed request/response via Pydantic, auto OpenAPI docs at `/docs`. The standard for Python ML services. |
| ASGI server | **Uvicorn** | Lightweight dev/prod server for FastAPI. |
| Embeddings | **sentence-transformers** (`all-MiniLM-L6-v2`) | Open-source, runs locally with no API key or per-call cost. 384-dim vectors, fast on CPU, strong quality/speed tradeoff. |
| Vector store | **ChromaDB** (persistent, on-disk) | Zero-infra embedded vector DB. Cosine similarity, metadata filtering, survives restarts. No external service to stand up. |
| Answer synthesis | **Anthropic SDK** + `claude-opus-4-8` | Claude reads the retrieved passages and writes a grounded, cited answer. Adaptive thinking for reasoning quality. |
| PDF parsing | **pypdf** | Pure-Python text extraction from PDFs. |
| Config | **pydantic-settings** + `.env` | Typed settings; secrets stay out of code. |

### Why Claude + open embeddings (not one provider for both)

Embeddings are a high-volume, low-creativity operation — running them locally
with `sentence-transformers` means the demo costs nothing to index and needs no
key to try. Answer **synthesis** is where model quality is visible to the user,
so that single step uses Claude (`claude-opus-4-8`). This split is also good
engineering: it decouples the retrieval layer from any one vendor.

### Anthropic API conventions (locked)

- Model ID: **`claude-opus-4-8`** (exact string, no date suffix).
- **Adaptive thinking**: `thinking={"type": "adaptive"}` with
  `output_config={"effort": "medium"}` for the synthesis call.
- Client reads `ANTHROPIC_API_KEY` from the environment — never hard-code keys.
- `max_tokens` sized for a full answer (~1500); raise + stream if answers grow.

## Frontend — React 18 + Vite

| Concern | Choice | Why |
|---|---|---|
| Build tool | **Vite** | Instant dev server, fast HMR, minimal config. |
| UI library | **React 18** | Component model recruiters recognize; no framework lock-in. |
| HTTP | **fetch** (native) | No dependency needed for a handful of endpoints. |
| Styling | Hand-written CSS (`styles.css`) | Custom, distinctive look — avoids generic component-library aesthetics. |

## Project layout

```
llm-semantic-search/
├── constitution/        # mission, roadmap, tech-stack (this folder)
├── backend/
│   ├── app/
│   │   ├── main.py        # FastAPI app + routes
│   │   ├── config.py      # typed settings
│   │   ├── schemas.py     # Pydantic request/response models
│   │   ├── ingestion.py   # parse + chunk documents
│   │   ├── embeddings.py  # sentence-transformers wrapper
│   │   ├── vectorstore.py # ChromaDB wrapper
│   │   ├── llm.py         # Claude answer synthesis
│   │   └── search.py      # orchestrates retrieve -> synthesize
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── api.js
    │   └── components/
    └── package.json
```

## Local development

- Backend: `uvicorn app.main:app --reload` on `:8000`.
- Frontend: `npm run dev` on `:5173`, proxying `/api` to the backend.
- Only required secret: `ANTHROPIC_API_KEY` in `backend/.env`.

## Deployment notes (future)

- Backend containerizes cleanly (single Python image + persistent volume for
  Chroma). Frontend builds to static assets for any CDN/host.
- Embedding model (~90 MB) downloads on first run and caches locally.

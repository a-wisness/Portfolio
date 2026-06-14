# Roadmap

A phased build. Phase 1 is the portfolio-ready MVP and the scope of the initial
implementation in this repo; later phases are stretch goals that show product
thinking.

## Phase 0 — Foundation ✅ (this commit)

- Repository scaffold and the constitution (`mission`, `tech-stack`, `roadmap`).
- Backend skeleton: FastAPI app, typed settings, Pydantic schemas.
- Frontend skeleton: Vite + React app shell.

## Phase 1 — MVP: end-to-end RAG (core deliverable)

The minimum that makes the demo real and clickable.

- **Ingestion**
  - Upload endpoint accepting PDF, Markdown, and plain text.
  - Text extraction (pypdf for PDFs) and overlap-aware chunking.
- **Indexing**
  - Embed chunks locally with `sentence-transformers`.
  - Persist vectors + metadata (source filename, chunk index, text) in ChromaDB.
- **Retrieval**
  - Embed the query, fetch top-k chunks by cosine similarity.
- **Generation**
  - Claude (`claude-opus-4-8`) synthesizes a grounded answer with inline
    `[n]` citations referencing the retrieved passages.
  - Graceful "I couldn't find this in the documents" when context is weak.
- **Frontend**
  - Upload panel with indexing feedback.
  - Search bar → answer card (with citations) + expandable source passages.
  - Document list / count.

**Exit criteria:** a fresh visitor uploads a doc, asks a question, and gets a
cited answer with visible sources — no config beyond the API key.

## Phase 2 — Search quality & UX polish

- **Hybrid retrieval:** combine dense (vector) with sparse (BM25/keyword) for
  better recall on names, codes, and rare terms.
- **Reranking:** a Claude- or cross-encoder-based rerank pass over top-k for
  sharper ordering before synthesis.
- **Streaming answers:** stream the synthesis token-by-token to the UI.
- **Highlight citations:** clicking `[2]` scrolls to / highlights that passage.
- **Per-document scoping:** search within a chosen document or the whole corpus.

## Phase 3 — Robustness & scale

- Larger corpora: batched embedding, ingestion progress, async background jobs.
- More formats: `.docx`, HTML, CSV.
- Evaluation harness: a small Q/A set measuring retrieval hit-rate and answer
  groundedness to quantify changes.
- Caching of query embeddings and repeated answers.

## Phase 4 — Productionization (portfolio "above and beyond")

- Dockerfile + docker-compose for one-command spin-up.
- Multi-user sessions / namespaced collections.
- Auth and rate limiting.
- Deploy: static frontend + containerized backend with a persistent volume.
- Observability: request logging, latency and token-usage metrics.

## Tracking

Phase 1 items are the definition of done for the first release. Anything in
Phase 2+ is explicitly optional and should be pulled in only after Phase 1 is
solid and demoable.

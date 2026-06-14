# Tech Stack

Chosen to showcase PyTorch deep-learning engineering end to end, while staying
runnable locally with no GPU and no API keys.

## ML / training — Python 3.11+

| Concern | Choice | Why |
|---|---|---|
| DL framework | **PyTorch** (CPU wheel) | The core showcase. NCF model, training loop, and evaluation are hand-written in idiomatic PyTorch. |
| Data handling | **pandas** + **numpy** | Load and preprocess MovieLens; build ID mappings and interaction sets. |
| Dataset | **MovieLens-100k** (default), 1M optional | Public benchmark from GroupLens. 100k trains in ~1–2 min on CPU for a fast, reproducible demo; 1M is a config switch. Downloaded by a script. |
| Metrics | hand-implemented **HR@K / NDCG@K** | Leave-one-out ranking metrics — implementing them (not importing) is part of the demonstration. |
| Retrieval (ANN) | **brute-force cosine** (default) / optional **FAISS** | Item-embedding lookups (cold-start, "similar movies") go through a pluggable index ([`ann.py`](../backend/app/ann.py)). Exact brute force needs no extra dependency; FAISS (`USE_FAISS=true`, `requirements-faiss.txt`) is there for catalog scale. |
| Config | **pydantic-settings** + dataclass/`.env` | Typed, reproducible hyperparameters. |

### Models & evaluation (Phase 2–3)

- **Comparable baselines:** Popularity, Matrix Factorization, GMF, MLP, and
  NeuMF all implement one `forward(user, item)` interface and run through a
  shared, model-agnostic ranking harness ([`evaluation.py`](../backend/app/evaluation.py)).
- **Objectives:** point-wise **BCE** and pairwise **BPR** (`objective="bpr"`).
- **Sequential model:** **SASRec** ([`sasrec.py`](../backend/app/sasrec.py)) —
  self-attention over user interaction histories, evaluated with the same
  leave-one-out next-item protocol (experiment; served model stays NeuMF).
- **Negative sampling:** uniform or **popularity-aware (hard)** negatives
  (`negative_sampling`), compared in the benchmark.
- **Cold-start blend:** genre priors mixed with embedding similarity
  (`genre_weight`) in the cold-start recommender.
- **Benchmark & sweep:** [`benchmark.py`](../backend/app/benchmark.py) tabulates all
  models (committed to `reports/`); [`sweep.py`](../backend/app/sweep.py) grids the
  NeuMF hyperparameters. `matplotlib` (dev-only) renders training curves.

### Serving & ops (Phase 4)

- **Model registry** ([`registry.py`](../backend/app/registry.py)): versioned
  artifacts under `artifacts/versions/` with JSON metadata sidecars and an
  `active.txt` pointer; legacy single-file fallback preserved.
- **Hot reload:** the active recommender is a hot-swappable module slot —
  list / activate / reload versions over HTTP with no restart.
- **Observability** ([`observability.py`](../backend/app/observability.py)):
  dependency-free in-process metrics (request volume, latency percentiles,
  status buckets, recommendation coverage) + request-logging middleware.
- **Bounded LRU caches** for per-user / per-similar / per-like-set results.

### Model & training design (locked)

- **Architecture:** NeuMF = GMF tower (element-wise user⊗item embedding) + MLP
  tower (concatenated embeddings → hidden layers) → concatenated → linear →
  sigmoid. Separate embedding tables per tower (per the NCF paper).
- **Objective:** implicit feedback. Positives = observed interactions (rating ≥
  4 treated as a positive "like"; configurable). Negatives = uniformly sampled
  unobserved items, `num_negatives` per positive. Loss = `BCEWithLogitsLoss`.
- **Split & eval:** leave-one-out per user (hold out the latest interaction).
  Evaluate by ranking the held-out item against 99 sampled negatives → HR@10,
  NDCG@10, logged each epoch.
- **Reproducibility:** global seed; deterministic data prep; device-aware
  (`cuda` if available, else `cpu`).
- **Artifact:** training writes a single versioned bundle to `artifacts/` —
  model `state_dict`, user/item index mappings, movie metadata, and the config
  used. The API loads exactly this bundle.

## Backend — FastAPI

| Concern | Choice | Why |
|---|---|---|
| Web framework | **FastAPI** + Uvicorn | Async, typed (Pydantic), auto OpenAPI docs. Loads the trained artifact at startup. |
| Inference | **PyTorch** (eval mode, `no_grad`) | Scores candidate items for a user / set of liked movies. |

### Recommendation endpoints (planned)

- `GET  /api/movies?search=&limit=` — browse/search the catalog (title, genres).
- `POST /api/recommend` — body: `{ "liked_movie_ids": [...], "top_k": N }`.
  Cold-start path for an ad-hoc visitor: aggregate the learned **item
  embeddings** of the liked movies and rank the catalog by similarity (excludes
  the liked set). Lets any visitor get recommendations without being a known
  user.
- `GET  /api/users/{user_id}/recommendations` — true NCF scoring for a known
  user: score all unseen items, return top-K. Demonstrates the model as trained.
- `GET  /api/movies/{movie_id}/similar` — nearest movies in item-embedding space
  (cosine). Shows the embeddings learned something meaningful.
- `GET  /api/health` — liveness; reports whether a model artifact is loaded.

## Frontend — React 18 + Vite

| Concern | Choice | Why |
|---|---|---|
| Build tool | **Vite** | Fast dev server, minimal config. |
| UI | **React 18**, native `fetch` | Search movies, select a few "I like these", get a ranked recommendation grid with scores + genres. Dark, hand-written CSS (no component library — distinctive look). |

## Containerization & tooling

| Concern | Choice | Why |
|---|---|---|
| Orchestration | **docker-compose** | One command to serve. Frontend nginx serves the SPA and reverse-proxies `/api` to the backend (same-origin, no CORS). |
| Model in Docker | shared **`artifacts/` volume** | A `trainer` step (`docker compose run --rm backend python -m app.train`) downloads data + trains into the volume; `backend` serves from it. The API degrades gracefully (`/api/health` reports `model_loaded: false`) until trained. CPU-only torch keeps the image lean. |
| Tests | **pytest** | Runs fully offline on tiny synthetic data — no MovieLens download, no real training. |

> **No secrets required.** Unlike the previous project, this one calls no
> external API, so there's no key to manage.

## Project layout (planned)

```
pytorch-recommender/
├── constitution/          # mission, tech-stack, roadmap
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI app + routes
│   │   ├── config.py        # typed settings / hyperparameters
│   │   ├── schemas.py       # Pydantic request/response models
│   │   ├── data.py          # MovieLens download + preprocessing + LOO split
│   │   ├── dataset.py       # torch Dataset + negative sampling
│   │   ├── model.py         # NeuMF (GMF + MLP) in PyTorch
│   │   ├── metrics.py       # HR@K, NDCG@K
│   │   ├── train.py         # training/eval loop -> writes artifact
│   │   └── recommender.py   # loads artifact, scores/ranks for the API
│   ├── scripts/download_data.py
│   ├── tests/
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── Dockerfile
├── frontend/                # React + Vite UI (+ nginx.conf, Dockerfile)
├── scripts/smoke_test.sh
└── docker-compose.yml
```

## Local development

- Train: `python -m app.train` (downloads data on first run, writes `artifacts/`).
- Backend: `uvicorn app.main:app --reload` on `:8000`.
- Frontend: `npm run dev` on `:5173`, proxying `/api` to the backend.

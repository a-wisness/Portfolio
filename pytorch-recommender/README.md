# CineMatch — PyTorch Neural Collaborative Filtering Recommender

A deep-learning movie recommender built **from scratch in PyTorch**. It learns
user and movie embeddings from MovieLens interactions with **Neural
Collaborative Filtering (NeuMF)**, trained on **implicit feedback** with
negative sampling, and serves personalized recommendations through a FastAPI
backend and a React UI.

> A portfolio project demonstrating end-to-end applied deep learning: a correct
> training/evaluation loop (leave-one-out, HR@K / NDCG@K), a serialized model
> artifact, a typed inference API, and a test suite covering the data, the
> model, and the metrics.

See [`constitution/`](./constitution) for the **mission**, **tech-stack**, and
**roadmap**.

## How it works

```
 MovieLens ratings ──▶ binarize to "likes" (rating ≥ 4) ──▶ leave-one-out split
                                                                    │
        negative sampling (per epoch)                               │
                    │                                               ▼
                    ▼                                    ┌────────────────────┐
        ┌───────────────────────┐   train (BCE)         │ leave-one-out eval │
        │  NeuMF                 │ ────────────────────▶ │  HR@10 / NDCG@10    │
        │  GMF tower (u ⊗ i)     │                       └────────────────────┘
        │  + MLP tower (u ‖ i)   │
        │  → fused → logit       │ ──▶ artifacts/model.pt ──▶ FastAPI ──▶ React
        └───────────────────────┘
```

- **NeuMF** (He et al., 2017): a GMF tower (element-wise user⊗item embeddings)
  and an MLP tower (concatenated embeddings → hidden layers), fused into a final
  prediction layer. Hand-written in PyTorch ([`backend/app/model.py`](./backend/app/model.py)).
- **Implicit feedback:** observed interactions are positives; unobserved items
  are sampled as negatives each epoch; trained with `BCEWithLogitsLoss`.
- **Honest evaluation:** the standard leave-one-out protocol — rank each user's
  held-out item against 99 sampled negatives — reporting **HR@10** and
  **NDCG@10** (metrics hand-implemented in [`metrics.py`](./backend/app/metrics.py)).

## Recommendation strategies (API)

| Endpoint | Strategy |
|---|---|
| `POST /api/recommend` | **Cold-start** for any visitor: average the learned item embeddings of the movies they like, rank the catalog by cosine similarity |
| `GET /api/users/{id}/recommendations` | **NeuMF scoring** for a known training user (excludes already-seen movies) |
| `GET /api/movies/{id}/similar` | **Nearest items** in the learned embedding space |
| `GET /api/movies?search=` | Browse / search the catalog |
| `GET /api/health` | Liveness + active model version (+ its metrics) |

### Operational endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/models` | List trained versions (with metrics; active one flagged) |
| `POST /api/models/{version}/activate` | Hot-swap the served version (no restart) |
| `POST /api/models/reload` | Reload the active version from disk (after retraining) |
| `GET /api/metrics` | Serving metrics: request volume, latency p50/p95, coverage |

## Project structure

```
pytorch-recommender/
├── constitution/          # mission.md, tech-stack.md, roadmap.md
├── backend/
│   ├── app/
│   │   ├── model.py         # NeuMF + MF / GMF / MLP baselines + factory
│   │   ├── sasrec.py        # SASRec self-attention sequential model (experiment)
│   │   ├── baselines.py     # non-neural popularity baseline
│   │   ├── data.py          # MovieLens download + preprocessing + LOO split
│   │   ├── dataset.py       # torch Datasets: point-wise (BCE) + pairwise (BPR)
│   │   ├── metrics.py       # HR@K, NDCG@K
│   │   ├── evaluation.py    # model-agnostic leave-one-out ranking harness
│   │   ├── ann.py           # ANN retrieval: brute-force / optional FAISS
│   │   ├── training.py      # train any model under BCE or BPR
│   │   ├── train.py         # CLI: train NeuMF -> artifacts/model.pt
│   │   ├── benchmark.py     # train all models -> reports/benchmark.{md,png}
│   │   ├── sweep.py         # hyperparameter sweep -> reports/sweep.md
│   │   ├── recommender.py   # active-version load, ANN retrieval, hot reload, caching
│   │   ├── registry.py      # versioned artifacts + active-version pointer
│   │   ├── observability.py # in-process serving metrics (latency, coverage)
│   │   ├── main.py          # FastAPI routes (recommend + operational)
│   │   ├── config.py        # typed settings / hyperparameters
│   │   └── schemas.py       # Pydantic request/response models
│   ├── tests/               # offline pytest suite (74 tests)
│   ├── artifacts/           # versioned model registry (versions/ + active.txt)
│   └── Dockerfile
├── reports/                 # committed evaluation reports
│   ├── benchmark.md         # model comparison table
│   ├── benchmark.png        # training curves + final-metric bars
│   └── sweep.md             # hyperparameter sweep table
├── frontend/                # React + Vite UI (+ nginx.conf, Dockerfile)
├── scripts/smoke_test.sh    # live end-to-end check
├── docker-compose.yml       # dev stack (+ trainer profile)
├── docker-compose.prod.yml  # single-host production stack
└── DEPLOY.md                # deployment guide
```

## Quick start with Docker (recommended)

```bash
docker compose up --build      # → open http://localhost:8080
```

If a checkpoint is committed in `backend/artifacts/`, it's baked into the image
and served immediately. Otherwise the API reports `model_loaded: false` until
you train (below). API docs: <http://localhost:8000/docs>.

**Train / retrain** into the persistent artifacts volume:

```bash
# One-shot train job (writes a new version into the artifacts volume), then serve:
docker compose --profile train up --build trainer
docker compose up --build

# …or retrain against an already-running stack:
docker compose run --rm trainer
docker compose restart backend          # or: curl -X POST localhost:8000/api/models/reload
```

Each run produces a new version under `artifacts/versions/`; switch which one is
served at runtime via `POST /api/models/{version}/activate` (see below).

**Production:** a single-host deployment (frontend-only public port, restart
policies, healthchecks, resource limits) is in
[`docker-compose.prod.yml`](./docker-compose.prod.yml):

```bash
docker compose -f docker-compose.prod.yml up -d --build   # → http://<host>/
```

See [`DEPLOY.md`](./DEPLOY.md) for the full guide — no-downtime model updates via
the registry, TLS guidance, monitoring, and scaling caveats.

## Manual setup (without Docker)

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m app.train                 # downloads MovieLens-100k, trains, writes artifacts/model.pt
uvicorn app.main:app --reload       # serves on http://localhost:8000
```

Training on MovieLens-100k takes ~2–4 minutes on CPU. Hyperparameters
(embedding dim, MLP layers, negatives, epochs, learning rate, dataset) are in
[`app/config.py`](./backend/app/config.py) and overridable via env vars.

### Frontend

```bash
cd frontend
npm install
npm run dev                         # http://localhost:5173 (proxies /api → :8000)
```

## Testing

### Unit & integration tests (pytest)

Runs fully **offline** — no MovieLens download, no real training. The data
helpers, model, dataset, and metrics are tested directly; the API tests load a
tiny serialized NeuMF artifact so the real model-load + scoring paths run.

```bash
cd backend
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

| File | Covers |
|---|---|
| `tests/test_data.py` | Index maps, leave-one-out split, negative sampling, end-to-end prepare |
| `tests/test_model.py` | NeuMF forward/shapes, dataset negative sampling, loss decreases |
| `tests/test_models.py` | MF / GMF / MLP forward + shapes, the model factory, MF bias terms |
| `tests/test_metrics.py` | HR@K and NDCG@K correctness |
| `tests/test_evaluation.py` | Model-agnostic ranking harness + popularity baseline |
| `tests/test_training.py` | Training harness (BCE/BPR), BPR dataset |
| `tests/test_ann.py` | Brute-force / FAISS index parity, k-capping, fallback |
| `tests/test_recommender.py` | ANN-backed retrieval + result caching |
| `tests/test_registry.py` | Versioning, active pointer, activate, legacy fallback |
| `tests/test_observability.py` | Metrics collector: latency percentiles, coverage |
| `tests/test_sasrec.py` | SASRec model, causal masking, sequence dataset, training |
| `tests/test_api.py` | All endpoints (recommend + operational), validation & errors |

Currently **85 tests**, all offline (the FAISS parity test self-skips if
`faiss` isn't installed). `test_training.py` also covers the negative-sampling
strategies; `test_recommender.py` covers the genre-blend cold-start.

### Smoke test (live stack)

After `docker compose up`:

```bash
./scripts/smoke_test.sh                        # via nginx proxy (port 8080)
./scripts/smoke_test.sh http://localhost:8000  # backend directly
```

Recommendation steps are reported SKIPPED (not failed) if no model is trained.

## Model comparison (benchmark)

All models are evaluated through the **same** leave-one-out protocol, so the
numbers are directly comparable. On MovieLens-100k (10 epochs each, embedding
dim 32):

| Model | Objective | HR@10 | NDCG@10 |
|---|---|---:|---:|
| **NeuMF** ★ | BCE | **0.733** | **0.458** |
| NeuMF (pop-neg) | BCE | 0.668 | 0.436 |
| SASRec † | seq | 0.696 | 0.436 |
| GMF | BCE | 0.687 | 0.417 |
| MF | BCE | 0.669 | 0.407 |
| NeuMF | BPR | 0.666 | 0.391 |
| MLP | BCE | 0.618 | 0.361 |
| Popularity | — | 0.469 | 0.265 |

The fused NeuMF tops every baseline and beats the non-personalized popularity
floor by a wide margin. **NeuMF (pop-neg)** swaps uniform negatives for
popularity-aware (hard) ones — interestingly, no win here. **SASRec** (†) is the
self-attention *sequential* model; it converges slowly (trained 40 epochs vs 10)
but reaches the learned-model range. Reproduce with:

```bash
cd backend && python -m app.benchmark          # writes reports/benchmark.{md,png}
```

Full table and training-curve / bar plots: [`reports/benchmark.md`](./reports/benchmark.md).

> The **BPR** (pairwise) objective is available for any model via
> `train_model(..., objective="bpr")`; here it underperforms BCE at 10 epochs,
> which is a fair, reported result rather than a hidden one.

## Scale & retrieval

- **MovieLens-1M:** switch datasets with `DATASET=ml-1m python -m app.train`
  (or set `dataset` in `config.py`). The pipeline preprocesses all 1M ratings
  (6,040 users × 3,706 items, ~569k positives) in ~22s.
- **ANN retrieval:** cold-start and "similar movies" look up item embeddings
  through a pluggable index ([`ann.py`](./backend/app/ann.py)) — exact
  brute-force cosine by default, or **FAISS** for catalog scale:

  ```bash
  pip install -r backend/requirements-faiss.txt   # optional
  USE_FAISS=true uvicorn app.main:app             # else brute force
  ```

- **Caching:** per-user, per-similar, and per-like-set recommendations are
  memoized in the recommender (deterministic for a loaded model).
- **Hard negatives:** train with popularity-aware negatives via
  `NEGATIVE_SAMPLING=popularity` (vs. the default uniform).
- **Richer cold-start:** `POST /api/recommend` blends genre priors with
  embedding similarity (`GENRE_WEIGHT`, default 0.2).
- **Sequential model:** SASRec is implemented as an experiment
  ([`sasrec.py`](./backend/app/sasrec.py)) and benchmarked with the same
  protocol; the served model remains NeuMF.

## Hyperparameter sweep

```bash
cd backend && python -m app.sweep                 # writes reports/sweep.md
```

Grids embedding dim × negatives × learning rate, evaluated with the same
leave-one-out protocol. On ml-100k, more negatives clearly help — best config
`gmf_dim=32, num_negatives=8` (NDCG@10 0.469). Table: [`reports/sweep.md`](./reports/sweep.md).

## Productionization (MLOps)

- **Model versioning & registry** ([`registry.py`](./backend/app/registry.py)) —
  every `python -m app.train` writes a timestamped snapshot to
  `artifacts/versions/<version>.pt` with a JSON metadata sidecar; `active.txt`
  names the served version.
- **Hot reload, no downtime** — the active model is a hot-swappable slot:

  ```bash
  curl localhost:8000/api/models                          # list versions
  curl -X POST localhost:8000/api/models/<version>/activate  # switch live
  curl -X POST localhost:8000/api/models/reload             # after retraining
  ```

- **Observability** — request-logging middleware plus `GET /api/metrics`:

  ```jsonc
  {
    "total_requests": 7,
    "latency_ms": { "p50": 1.0, "p95": 68.5, "max": 80.8 },
    "recommendations": { "coverage": 0.0048, "catalog_size": 1682, ... }
  }
  ```

  Coverage (distinct items ever recommended ÷ catalog) is a real health signal —
  a recommender that only surfaces a few popular titles has a problem.
- **Bounded caches** — per-user / per-similar / per-like-set LRU caches, capped
  by `cache_size`.

## Design notes

- **Correct ML first.** Negative sampling, the leave-one-out split, and the
  ranking metrics are the substance — each is implemented explicitly and
  unit-tested.
- **Separation of concerns.** Data, model, dataset, metrics, training, and
  serving are independent modules; the API depends only on the serialized
  artifact, not on the training code path.
- **Reproducibility.** Seeded runs, config-driven hyperparameters, and a single
  self-contained artifact (weights + ID maps + movie metadata + metrics).
- **No external services / keys.** Everything runs locally on CPU.

## Roadmap highlights

Baseline comparison (popularity / MF / GMF / MLP vs NeuMF), a BPR pairwise
objective, MovieLens-1M, FAISS retrieval, and a committed evaluation report are
tracked in [`constitution/roadmap.md`](./constitution/roadmap.md).

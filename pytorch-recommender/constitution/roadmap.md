# Roadmap

A phased build. Phase 1 is the portfolio-ready MVP; later phases are
robustness/scale and productionization.

**Status:** Phases 0–4 feature-complete. **85 tests passing** (offline; +1
FAISS test skipped unless faiss installed). Committed: a versioned NeuMF
checkpoint (`artifacts/versions/`), evaluation reports (`reports/benchmark.md`,
`reports/sweep.md`), and deploy manifests (`docker-compose.prod.yml`,
`DEPLOY.md`).

## Phase 0 — Foundation ✅

- Repository scaffold and the constitution (`mission`, `tech-stack`, `roadmap`).
- Backend + frontend skeletons; `requirements`, Docker, compose stubs.

## Phase 1 — MVP: train an NCF recommender and serve it ✅ (core deliverable)

**Data pipeline**
- Download MovieLens-100k (script); map users/items to contiguous indices.
- Binarize to implicit positives (rating ≥ 4 configurable); build per-user
  interaction sets.
- Leave-one-out split: hold out each user's latest interaction for test.
- Pre-sample 99 negatives per user for evaluation candidates.

**Model (PyTorch)**
- `NeuMF`: GMF tower + MLP tower with separate embeddings, fused head.
- Clean `forward` returning logits; `BCEWithLogitsLoss`.

**Training & evaluation**
- Training loop with per-epoch negative resampling, Adam, configurable
  hyperparameters, global seed, device selection.
- Leave-one-out evaluation computing **HR@10** and **NDCG@10** each epoch.
- Serialize the best checkpoint + ID mappings + movie metadata + config to
  `artifacts/`.

**Serving (FastAPI)**
- Load the artifact at startup; `/api/health` reports `model_loaded`.
- `GET /api/movies` (search/browse), `POST /api/recommend` (cold-start from
  liked movies via item embeddings), `GET /api/users/{id}/recommendations`
  (NCF scoring), `GET /api/movies/{id}/similar` (embedding nearest neighbors).
- Exclude already-seen items; return scores + genres.

**Frontend (React)**
- Search the catalog, select liked movies to build a profile, request
  recommendations, view a ranked grid with scores and genres.
- "Similar movies" view from item embeddings.

**Exit criteria:** `python -m app.train` produces an artifact with reported
HR@10/NDCG@10; a visitor selects movies in the UI and receives sensible,
deduplicated recommendations.

**Result:** met. NeuMF trained on MovieLens-100k (20 epochs, CPU) →
**HR@10 0.72 / NDCG@10 0.47**; 1.2 MB checkpoint committed and served by the
API; FastAPI + React UI working; full offline test suite green.

## Phase 2 — Model quality & evaluation depth ✅

- ✅ **Baselines for comparison:** popularity baseline + Matrix Factorization +
  GMF + MLP, tabulated against NeuMF (HR@10, NDCG@10) through one shared
  evaluation harness. See [`reports/benchmark.md`](../reports/benchmark.md) and
  run `python -m app.benchmark`.
- ✅ **Pairwise objective:** BPR loss variant alongside BCE
  (`train_model(..., objective="bpr")`), included as a row in the benchmark.
- ✅ **Training visibility:** NDCG@K training curves + final-metric bar chart
  saved to `reports/benchmark.png`.
- ✅ **Hyperparameter sweep:** delivered in Phase 3 ([`sweep.py`](../backend/app/sweep.py),
  [`reports/sweep.md`](../reports/sweep.md)).
- ⏳ **Better negatives:** popularity-aware / hard negative sampling — deferred
  to Phase 4.

**Phase 2 result (MovieLens-100k, 10 epochs):** NeuMF leads at HR@10 0.733 /
NDCG@10 0.458, ahead of GMF, MF, MLP, and well above the popularity floor
(NDCG@10 0.265) — confirming the fused NCF architecture's lift.

## Phase 3 — Robustness & scale

- ✅ **MovieLens-1M** supported via the `dataset` config switch; the data
  pipeline preprocesses all 1M ratings (6,040 users × 3,706 items, ~569k
  positives) in ~22s. (25M is a further step.)
- ✅ **ANN retrieval** ([`ann.py`](../backend/app/ann.py)): item-embedding lookups
  for cold-start and "similar movies" go through a pluggable index — exact
  brute-force by default, optional FAISS (`USE_FAISS=true`) for catalog scale.
- ✅ **Caching:** per-user, per-similar, and per-like-set recommendations are
  memoized in the recommender (deterministic for a loaded model).
- ✅ **Hyperparameter sweep** ([`sweep.py`](../backend/app/sweep.py)): grid over
  embedding dim × negatives × lr, evaluated with the same protocol; see
  [`reports/sweep.md`](../reports/sweep.md). (Resolves the Phase 2 ⏳ sweep item.)
- ✅ **Committed evaluation reports:** [`reports/benchmark.md`](../reports/benchmark.md)
  + curves and [`reports/sweep.md`](../reports/sweep.md).
- ⏳ **Richer cold-start** (genre priors blended with embedding similarity) —
  deferred to Phase 4.

**Phase 3 findings (ml-100k, 8-epoch sweep):** more negatives clearly help
(`num_negatives` 8 > 4); the best config is `gmf_dim=32, num_negatives=8`
(NDCG@10 0.469). The ANN refactor reproduces the brute-force results exactly.

## Phase 4 — Productionization (portfolio "above and beyond") — in progress

**MLOps core (done):**
- ✅ **Model versioning / registry** ([`registry.py`](../backend/app/registry.py)):
  each training run writes a snapshot to `artifacts/versions/<version>.pt` with a
  JSON sidecar; `active.txt` names the served version. Legacy single-file
  `model.pt` still honored as a fallback.
- ✅ **Reload without downtime:** the active recommender is a hot-swappable
  module slot. `GET /api/models`, `POST /api/models/{version}/activate`, and
  `POST /api/models/reload` switch/reload the served model with no restart.
- ✅ **Observability** ([`observability.py`](../backend/app/observability.py)):
  request-logging middleware + `GET /api/metrics` exposing request volume,
  latency p50/p95/max, status buckets, and recommendation **coverage** (distinct
  items recommended / catalog size).
- ✅ **Bounded caches:** the recommender's per-user / per-similar / per-like-set
  LRU caches are size-capped (`cache_size`).
- ✅ **Docker seed** updated to seed the whole versioned registry into the
  artifacts volume.

**Models & data science (done):**
- ✅ **Sequential recommender — SASRec** ([`sasrec.py`](../backend/app/sasrec.py)):
  self-attention over each user's interaction history, evaluated with the same
  leave-one-out next-item protocol and added to the benchmark. (Experiment; the
  served model stays NeuMF.) It converges slowly — competitive (~NDCG@10 0.44)
  once given a larger epoch budget.
- ✅ **Hard / popularity-aware negatives** ([`dataset.py`](../backend/app/dataset.py)
  `NegativeSampler`): `negative_sampling="popularity"`, compared against uniform
  in the benchmark ("NeuMF (pop-neg)").
- ✅ **Richer cold-start:** the cold-start recommender blends genre priors with
  embedding similarity (`genre_weight`).

**One-command demo + deploy (done, unverified here):**
- ✅ Compose `trainer` profile: `docker compose --profile train up trainer`
  trains a fresh version into the artifacts volume, then `docker compose up`
  serves it.
- ✅ Production manifests + guide: [`docker-compose.prod.yml`](../docker-compose.prod.yml)
  (frontend-only public port, restart policies, healthchecks, resource limits)
  and [`DEPLOY.md`](../DEPLOY.md) (host deploy, no-downtime model updates via the
  registry, TLS guidance, scaling caveats). Builds to the verified dev pattern;
  not run — no Docker in the dev environment.

## Tracking

Phases 0–4 are feature-complete. Phases 0–3 and Phase 4's MLOps core (versioning,
hot-reload, observability, bounded caches), the SASRec experiment, hard
negatives, and genre-blend cold-start are all verified offline.

**Not yet verified** (no Docker in the dev environment): the Docker image build,
the compose `trainer`/prod flows, the live UI click-through, and full
MovieLens-1M *training* (the 1M data pipeline is verified; only training time
was skipped).

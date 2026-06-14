# Mission

## Product

**CineMatch** is a deep-learning movie recommender built in **PyTorch**. It
learns latent representations of users and movies from the MovieLens
interaction data and serves personalized top-N recommendations through a
FastAPI backend and a React UI. A visitor picks a few movies they like, and the
system returns movies it predicts they'll enjoy — with a transparent score and
the movie's genres.

## The problem it solves

Catalogs are too large to browse. Recommenders surface the small set of items a
given user is most likely to engage with. This project demonstrates the modern
**collaborative filtering** approach: rather than hand-engineered rules, the
model *learns* what "similar taste" means by training embeddings on millions of
implicit signals (which users interacted with which movies).

## Modeling approach (what makes this a *deep learning* project)

- **Neural Collaborative Filtering (NCF / NeuMF)** — He et al., 2017. The model
  fuses two towers:
  - **GMF** (Generalized Matrix Factorization): element-wise product of user
    and item embeddings — a learnable generalization of classic matrix
    factorization.
  - **MLP**: concatenated user/item embeddings passed through a multi-layer
    perceptron to capture non-linear interactions.
  - **NeuMF** concatenates both towers' outputs into a final prediction layer.
- **Implicit feedback + ranking.** Observed interactions are positives;
  unobserved (user, item) pairs are sampled as negatives. The model trains with
  binary cross-entropy over positive/negative pairs (point-wise learning to
  rank). This mirrors real-world recommenders, which see *what users did*, not
  star ratings they didn't give.
- **Honest evaluation.** The standard **leave-one-out** protocol: hold out each
  user's most recent interaction, rank it against 99 sampled negatives, and
  report **Hit Ratio@K (Recall@K)** and **NDCG@K**. These ranking metrics are
  what the field actually uses — not accuracy on a regression target.

## Who it's for

- **Primary (portfolio context):** ML/AI recruiters and engineers evaluating
  applied deep-learning ability — specifically PyTorch modeling, a correct
  training/eval loop, and the ability to ship a model behind a real API + UI.
- **Illustrative end user:** a movie watcher who wants suggestions based on a
  handful of films they already love.

## What success looks like

1. A clean, from-scratch **PyTorch NCF implementation** that trains on MovieLens
   and reports HR@10 / NDCG@10 competitive with the published baselines.
2. A visitor can open the app, select a few liked movies, and get sensible
   recommendations in seconds — no ML knowledge required.
3. The codebase reads as production-shaped ML: a reproducible training pipeline,
   a serialized model artifact, a typed inference API, and a test suite that
   validates the data, the model, and the metrics.

## Guiding principles

- **Correctness of the ML, first.** The negative sampling, leave-one-out split,
  and ranking metrics must be implemented correctly — that's the substance.
- **Reproducibility.** Seeded runs, config-driven hyperparameters, a single
  serialized artifact (weights + ID mappings + metadata) that the API loads.
- **Honest recommendations.** Recommendations exclude already-seen items and
  show a score; "similar movies" come from the learned embedding space.
- **Portfolio-legible.** Readable, well-separated modules over cleverness.

## Explicit non-goals

- Not a session-based / sequential recommender (no RNN/transformer over event
  sequences) in v1 — that's a stretch goal.
- Not multi-tenant or auth-gated; single-user demo.
- Not a distributed / GPU-cluster training project — it trains on a single
  machine (CPU is sufficient for MovieLens-100k).

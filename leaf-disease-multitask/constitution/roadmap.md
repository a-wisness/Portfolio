# Roadmap

A phased build. Phase 1 is the portfolio-ready MVP; later phases add model
quality, robustness, and productionization.

**Status:** Phases 0–4 ✅ (feature-complete). **50/50 tests pass.** A **real
bounded training run on the full dataset succeeded** (TF 2.17 / Keras 3, CPU) —
seg IoU 0.582 / Dice 0.664, classification accuracy 0.369 on held-out data; see
the Phase 2 result table and committed `reports/`. The model registry,
hot-reload endpoints, inference cache (Phase 3), and observability/metrics
(Phase 4) are **verified live** against the real model. The Docker build, the
compose flows, the `npm` frontend build, the live browser UI, and a
*full-length* (un-bounded) training run remain **not verified in this
environment** (no Docker / Node here).

## Phase 0 — Foundation

- Repository scaffold and the constitution (`mission`, `tech-stack`, `roadmap`).
- Backend + frontend skeletons; `requirements`, Docker, compose stubs.

## Phase 1 — MVP: one multi-task model, trained and served (core deliverable)

**Data pipeline (`data.py`)**
- `tf.data` loader for the **classification** set (28 classes, from
  `train/`), one-hot labels, a dummy zero mask, weights `seg=0, cls=1`.
- `tf.data` loader for the **segmentation** set (image+mask pairs from `data/`
  and `aug_data/`), a dummy zero class, weights `seg=1, cls=0`.
- A **unified loader** that interleaves both, shuffles, batches, and yields
  `(image, {segmentation, classification}, {segmentation, classification}
  sample weights)`.
- A real **held-out split** for each task (no train/test leakage).

**Model (`model.py`) + losses (`losses.py`)**
- `build_multitask_model()`: shared MobileNetV2 encoder → U-Net decoder head
  (`segmentation`, sigmoid) + classification head (`classification`, softmax).
- Dice+BCE segmentation loss; IoU and Dice metrics; categorical cross-entropy
  and accuracy for classification (all as `weighted_metrics`).

**Training & evaluation (`train.py`, `evaluate.py`)**
- Two-phase fit: (A) frozen encoder, (B) fine-tune top blocks at low LR, with
  `EarlyStopping` + `ModelCheckpoint`; global seed properly set.
- Serialize the best model + class label map + metadata to `artifacts/`.
- `evaluate.py`: report seg IoU/Dice and a classification report on held-out data.

**Serving (`main.py`, `inference.py`, `schemas.py`)**
- Load the artifact at startup; `/api/health` reports `model_loaded`.
- `POST /api/predict` (image → class + confidence + top-k + mask PNG),
  `GET /api/classes`.
- `inference.py` + a `predict_cli.py` for command-line predictions.

**Frontend (React)**
- Upload a leaf image, see the original with the predicted **mask overlaid**,
  the disease label, confidence, and a top-3 list. Dark, hand-written CSS.

**Tooling**
- `docker-compose` (nginx serves SPA + proxies `/api`), `trainer` step writing
  into a shared `artifacts/` volume.
- **Offline pytest suite** on tiny synthetic data: pipeline shapes & sample
  weights, model wiring (two outputs, correct shapes), losses/metrics math, and
  the API with a stubbed model.

**Exit criteria:** `python -m app.train` produces an artifact with reported
IoU/Dice + classification accuracy; a visitor uploads a leaf in the UI and
receives a mask overlay + disease prediction with confidence; the offline test
suite is green.

**Result (this environment):**
- ✅ Offline test suite green — **20/20** (`pytest`), covering the data pipeline
  (shapes + per-head sample weights + mask binarization), the model wiring (two
  named outputs, sigmoid mask / softmax classes, encoder freeze/unfreeze), the
  loss/metric math, and the API (stubbed predictor, 7 endpoint cases).
- ✅ End-to-end **train → `model.save` (.keras) → reload with custom losses →
  `predict`** verified on tiny synthetic data with `encoder_weights=None`
  (TF 2.17 / Keras 3). Both heads train and report **separate** metrics
  (`classification_accuracy` vs `segmentation_iou`/`dice`), confirming the
  sample-weight routing across the two disjoint datasets.
- ⏳ **Not yet verified here:** a full real training run (needs the MobileNetV2
  ImageNet weights download + meaningful compute) and therefore real
  IoU/Dice/accuracy numbers; the Docker image builds and compose flow; the
  `npm install` / Vite build (no Node in this environment); the live UI
  click-through. The code is built to the patterns used elsewhere; only
  execution was skipped.

## Phase 2 — Model quality & evaluation depth ✅ (code-complete)

- ✅ **Task-correct augmentation** ([`data.py`](../backend/app/data.py)):
  segmentation geometric transforms are applied to image **and** mask together
  (channel-concat so flips stay aligned); color jitter on images only.
  Classification gets flips + color jitter. Tests assert the mask stays binary.
- ✅ **Class-imbalance handling**: balanced inverse-frequency class weights
  (capped at 10×, mean-normalized) folded into the per-sample classification
  weight. The real run computed `min=0.31, max=7.00`, so the 2-image
  `Tomato two spotted spider mites leaf` class is no longer ignored.
- ✅ **Loss weighting** between the two heads is configurable
  (`seg_loss_weight` / `cls_loss_weight`); the ablation arm zeros the unused
  head's loss.
- ✅ **Single-task vs multi-task ablation** harness
  ([`ablation.py`](../backend/app/ablation.py)) — trains a classification-only
  baseline against the multi-task model and tabulates val accuracy to
  `reports/ablation.md`. Built + import-tested; **not run on real data here**
  (it trains twice ≈ 1.5 h on CPU).
- ✅ Committed **training curves** + markdown **report**
  ([`reporting.py`](../backend/app/reporting.py)) → `reports/training_curves.png`
  and `reports/report.md` (per-class precision/recall/f1).
- ⏳ **Grad-CAM** over the classification head — deferred to a later pass.

**Result — real bounded training run (this environment, CPU, TF 2.17/Keras 3):**
a deliberately bounded run (`max_steps_per_epoch=120`, 5 frozen + 3 fine-tune
epochs ≈ 2.6 passes over the data; the defaults of 12+8 full-pass epochs were
too long for an interactive session). Validation loss improved monotonically
**1.997 → 0.913**, with fine-tuning giving a clear additional drop
(1.146 → 0.913):

| task | metric | value | held-out set |
|---|---|---|---|
| Segmentation | IoU  | **0.582** | 705 val images |
| Segmentation | Dice | **0.664** | 705 val images |
| Classification | accuracy | **0.369** | 236-image `test/` split |

Per-class results vary widely (e.g. `Corn rust leaf` f1 0.82 vs several classes
at 0.00) — expected for a short run; the numbers are an honest floor, not a
tuned result. A longer run (raise `max_steps_per_epoch` / epoch counts) is the
obvious lever. The committed `reports/report.md` + `training_curves.png` capture
this run; the 76 MB `.keras` artifact is git-ignored (re-trainable via
`python -m app.train`).

## Phase 3 — Robustness & scale ✅ (code-complete, verified end-to-end)

- ✅ **Model versioning / registry** ([`registry.py`](../backend/app/registry.py)):
  each training run writes a snapshot to `artifacts/versions/<id>/`
  (`model.keras` + `labels.json` + `metadata.json`); `artifacts/active.txt`
  names the served version. Legacy flat Phase 1–2 artifacts are auto-migrated
  into `versions/legacy/` on first use (verified: the Phase 2 model migrated and
  kept serving).
- ✅ **Hot-reload over HTTP**: `GET /api/models` (list + active),
  `POST /api/models/{version}/activate`, `POST /api/models/reload` — switch the
  served model with no restart. `GET /api/health` reports the active version +
  cache stats (from the registry, without paying the TF model-load cost).
- ✅ **Inference caching** ([`inference.py`](../backend/app/inference.py)):
  bounded LRU keyed by image content hash + model version, cleared on reload.
  Verified live — repeated uploads hit the cache (2 hits / 1 miss over 3 calls).
- ✅ **Richer held-out evaluation**: a **confusion matrix** (`reports/
  confusion_matrix.png`) alongside the per-class report. (Per-class IoU isn't
  meaningful here — segmentation is a single binary leaf/background class — so
  it's intentionally omitted.)
- ⏳ Optional `download_data.py` for the full upstream datasets — **deferred**
  (Phase 1 decision was to use the local data as-is; low value for the demo).

**Verification (this environment):** **43/43 tests pass** (added registry,
caching, and model-management endpoint suites — all TF-free). The full path was
exercised live against the real trained model via `TestClient`: legacy
migration → `GET /api/health` (active_version `legacy`, 28 classes, no TF load)
→ `GET /api/models` → `POST /api/predict` (Corn rust leaf @ 0.817, cache hits on
repeat) → `POST /api/models/reload`. Committed `reports/` now include the
confusion matrix.

## Phase 4 — Productionization (portfolio "above and beyond") ✅ (code-complete)

- ✅ **Observability** ([`observability.py`](../backend/app/observability.py)):
  a dependency-free, thread-safe in-process metrics store + a request-logging
  middleware. `GET /api/metrics` exposes request volume, status-code buckets,
  latency **p50/p95/p99/max/avg** (bounded ring buffer), and the **distribution
  of predicted classes**. Every request is logged `method path -> status (ms)`.
- ✅ **Production manifest** ([`docker-compose.prod.yml`](../docker-compose.prod.yml)):
  standalone — backend internal-only, frontend the single public entrypoint on
  `:80`, `restart: always`, healthchecks on both services, CPU/memory limits,
  read-only data mount, and a `trainer` profile.
- ✅ **Healthcheck** baked into the backend image (`HEALTHCHECK` hitting
  `/api/health`).
- ✅ **Deploy guide** ([`DEPLOY.md`](../DEPLOY.md)): train-then-serve, **no-downtime
  model updates** via the registry (`/api/models/reload` + `/activate`),
  observability, TLS guidance, and scaling caveats (single-replica in-process
  cache/metrics).

**Verification (this environment):** **50/50 tests pass** (added an observability
suite + metrics-endpoint API tests). The metrics + logging path was exercised
live via `TestClient`. The compose manifests and Dockerfile `HEALTHCHECK` are
**not run here** (no Docker) — same caveat as earlier phases.

## Tracking

Update this file as phases complete, marking items ✅ and recording **real**
measured metrics (IoU/Dice, accuracy) and explicitly flagging anything not
verified in the dev environment (Docker build, full training time, live UI).

# LeafLens — multi-task leaf segmentation + disease classification

One TensorFlow/Keras model that, in a **single forward pass**, both **segments**
a crop leaf (pixel mask) and **classifies** its disease (28 classes) — served by
a FastAPI backend and a React UI.

This is a rewrite of the original `LeafDiseaseDetection-DL` notebook, which
trained two *separate* models (a U-Net for segmentation, a MobileNetV2 classifier
for disease). LeafLens **unifies them**: a single MobileNetV2 encoder feeds a
segmentation decoder head and a classification head, trained jointly across the
two datasets.

## Why this is interesting

The two source datasets are **disjoint** — segmentation images have masks but no
class label; classification images have a class but no mask. LeafLens trains one
model on both by giving every sample **per-head sample weights**: a
classification image trains only the classification head (`seg_weight=0`), a
segmentation image trains only the segmentation head (`cls_weight=0`), while the
**shared encoder learns from every image**. See
[`backend/app/data.py`](backend/app/data.py) and
[`backend/app/train.py`](backend/app/train.py).

It also fixes real bugs from the original notebook: softmax + categorical
cross-entropy for the 28-class head (was sigmoid + binary), a genuine held-out
test split (the notebook's "test set" was a reshuffled copy of train), IoU/Dice
metrics for masks (not just misleading pixel accuracy), a proper `seed`, and a
streaming `tf.data` pipeline instead of one giant in-RAM array.

## Architecture

```
        leaf image (224×224×3, raw [0,255])
                     │  (rescaled to [-1,1] inside the model)
        ┌──── shared MobileNetV2 encoder ────┐   (ImageNet weights; skip taps:
        │  block_1/3/6/13_expand + 16_project │    112,56,28,14,7 px)
        └─────────────────┬───────────────────┘
              ┌───────────┴────────────┐
     U-Net decoder (+skips)      GlobalAvgPool → Dropout
              │                         → Dense(28, softmax)
   segmentation: 224×224×1        classification: 28-way
        (sigmoid mask)               (disease label)
```

## Project layout

```
leaf-disease-multitask/
├── constitution/         mission.md · tech-stack.md · roadmap.md
├── backend/
│   ├── app/              config · data · model · losses · train · evaluate · inference
│   │                     registry · reporting · ablation · main (FastAPI)
│   ├── scripts/          predict_cli.py
│   ├── tests/            offline pytest suite (synthetic data)
│   └── Dockerfile · requirements*.txt
├── frontend/             React 18 + Vite UI (+ nginx.conf, Dockerfile)
├── scripts/smoke_test.sh
├── docker-compose.yml
└── README.md
```

## Quickstart (Docker)

Data is read in place from the sibling `../LeafDiseaseDetection-DL` folder
(mounted read-only).

```bash
# 1. Train once into the shared artifacts volume
docker compose --profile train run --rm trainer

# 2. Serve API + UI
docker compose up --build
# UI:  http://localhost:8080
# API: proxied at http://localhost:8080/api  (backend not exposed directly)
```

`/api/health` reports `model_loaded: false` until the trainer has run, so the
stack comes up gracefully either way.

## Quickstart (local, no Docker)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train (reads ../LeafDiseaseDetection-DL data, writes ../artifacts/)
python -m app.train

# Evaluate on held-out data (seg IoU/Dice + classification report)
python -m app.evaluate

# Ablation: does the auxiliary segmentation task help the classifier?
# (trains a classification-only baseline vs the multi-task model)
python -m app.ablation

# Predict from the CLI
python -m scripts.predict_cli "../../LeafDiseaseDetection-DL/Classification Data/test/Apple rust leaf/2011-011.jpg" \
    --save-overlay /tmp/overlay.png

# Serve the API
uvicorn app.main:app --reload   # http://localhost:8000
```

Frontend dev server (proxies `/api` to `:8000`):

```bash
cd frontend
npm install
npm run dev                      # http://localhost:5173
```

## Train on Colab, infer locally

No GPU? Train on a free Colab GPU and drop the result into a local install for
serving — training and serving are decoupled by the model registry, so a run
produces a self-contained version folder you just copy in and activate. See
[`docs/COLAB.md`](docs/COLAB.md) and the ready-to-run
[`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb).

## API

| Method | Path            | Description |
|--------|-----------------|-------------|
| GET    | `/api/health`   | Liveness; `model_loaded`, `num_classes`, `active_version`, `cache` stats. |
| GET    | `/api/classes`  | The 28 class labels. |
| POST   | `/api/predict`  | multipart image → `predicted_class`, `confidence`, `top_k`, `leaf_coverage`, and base64 PNG `mask`/`overlay` (results are LRU-cached). |
| GET    | `/api/models`   | List model versions + which is active. |
| POST   | `/api/models/{version}/activate` | Switch the served model version (hot-reload, no restart). |
| POST   | `/api/models/reload` | Re-resolve + reload the active version. |
| GET    | `/api/metrics`  | Request volume, latency p50/p95/p99/max, status buckets, predicted-class distribution. |

## Configuration

All settings are env vars prefixed `LEAFLENS_` (see [`.env.example`](.env.example)
and [`backend/app/config.py`](backend/app/config.py)) — data paths, image size,
batch size, epochs per phase, learning rates, the seg/cls batch mix ratio, and
loss weights.

## Testing

The suite is **fully offline**: it builds tiny synthetic images/masks in a temp
dir and exercises the data pipeline (shapes + per-head sample weights), the model
wiring (two named outputs, correct shapes), the loss/metric math, and the API
(with a stubbed predictor — no TensorFlow needed for the API tests).

```bash
cd backend
pip install -r requirements-dev.txt
pytest -q
```

## Model quality (Phase 2)

- **Task-correct augmentation** ([`data.py`](backend/app/data.py)): for
  segmentation, geometric transforms are applied to the image **and mask
  together** (concatenated on the channel axis) so they stay aligned; color
  jitter touches the image only. Classification images get flips + color jitter.
- **Class-imbalance handling**: balanced inverse-frequency **class weights**
  (capped, mean-normalized) are folded into the per-sample classification weight
  — without this the `Tomato two spotted spider mites leaf` class (2 training
  images) is effectively ignored.
- **Ablation** ([`ablation.py`](backend/app/ablation.py)): trains a
  classification-only baseline against the multi-task model to quantify whether
  the shared encoder benefits from the auxiliary segmentation task.
- **Reports** ([`reporting.py`](backend/app/reporting.py)): training runs emit
  `reports/training_curves.png` and `reports/report.md` (validation metrics +
  per-class precision/recall/f1).

## Robustness & serving (Phase 3)

- **Model registry** ([`registry.py`](backend/app/registry.py)): every training
  run is a versioned snapshot under `artifacts/versions/<id>/` with an
  `active.txt` pointer; legacy flat artifacts are auto-migrated. List, activate,
  and **hot-reload** model versions over HTTP — no restart.
- **Inference cache**: a bounded LRU keyed by image content + model version,
  cleared on reload; `/api/health` exposes hit/miss stats.
- **Confusion matrix** added to the held-out evaluation (`reports/
  confusion_matrix.png`).

## Productionization (Phase 4)

- **Observability** ([`observability.py`](backend/app/observability.py)): a
  dependency-free in-process metrics store + request-logging middleware;
  `GET /api/metrics` exposes volume, latency percentiles, status buckets, and the
  predicted-class distribution.
- **Production manifest** ([`docker-compose.prod.yml`](docker-compose.prod.yml))
  + **deploy guide** ([`DEPLOY.md`](DEPLOY.md)): internal-only backend, public
  nginx frontend, healthchecks, restart policies, resource limits, and
  **no-downtime model updates** through the registry.

## Status

Phases 1–4 ✅ (MVP · model quality · robustness/serving · productionization) —
see [`constitution/roadmap.md`](constitution/roadmap.md) for the phased plan,
real measured metrics, and what is / isn't verified in this environment.

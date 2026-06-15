# Tech Stack

Chosen to showcase a multi-task TensorFlow/Keras model end to end, while staying
runnable locally with no GPU and no API keys. Reuses the existing local data in
place (no re-download).

## ML / training — Python 3.11+

| Concern | Choice | Why |
|---|---|---|
| DL framework | **TensorFlow / Keras** (CPU wheel) | The core showcase, and the framework the original notebook used. Multi-task model, `tf.data` pipeline, training loop, and metrics are hand-written. |
| Backbone | **MobileNetV2** (ImageNet weights) | Shared encoder for both heads. Lightweight enough for CPU; the same skip-tap layers the notebook used for its best U-Net. |
| Data handling | **`tf.data`** + **numpy** + **Pillow** | Streaming pipelines for both datasets (no giant in-RAM array like the notebook). Per-sample task weights are baked into the pipeline. |
| Segmentation metrics | hand-implemented **IoU** + **Dice** | Pixel accuracy is misleading for masks; IoU/Dice are what the field reports. |
| Classification metrics | **accuracy** + scikit-learn **classification report** | Per-class precision/recall on a real held-out split. |
| Config | **pydantic-settings** + `.env` | Typed, reproducible hyperparameters and data paths. |

### Data (used as-is, referenced in place)

The improved project does **not** copy the ~10k images. `config.py` points at the
existing sibling folders by default:

- **Classification:** `../LeafDiseaseDetection-DL/Classification Data/`
  — `train/` (2,336 imgs) and `test/` (236 imgs) across **28 classes**.
- **Segmentation:** `../LeafDiseaseDetection-DL/Image Segmentation Data/`
  — `data/` (588 image+mask pairs) and `aug_data/` (2,940 augmented pairs);
  masks are PNGs sharing the image stem.

Paths are overridable via env vars so the data can live anywhere.

### Model & training design (locked)

- **Architecture:** `build_multitask_model()` returns one `tf.keras.Model` with
  two named outputs:
  - **`segmentation`** — MobileNetV2 encoder skip taps
    (`block_1/3/6/13_expand_relu`, `block_16_project`) → pix2pix-style upsample
    decoder with skip concatenations → `Conv2D(1, sigmoid)` at 224×224.
  - **`classification`** — encoder bottleneck → `GlobalAveragePooling2D` →
    `Dropout` → `Dense(28, softmax)`.
- **Disjoint-data training (the key technique):** a unified `tf.data` pipeline
  concatenates both datasets, each sample carrying `(image, {seg_mask,
  class_onehot}, {seg_weight, cls_weight})`. Classification images set
  `seg_weight=0`; segmentation images set `cls_weight=0`. Keras applies the
  per-output `sample_weight`, so each head trains only on its labeled samples
  while the encoder sees everything.
- **Losses:** `segmentation` = Dice + binary cross-entropy; `classification` =
  categorical cross-entropy. `weighted_metrics` so reported metrics ignore the
  dummy-labeled samples.
- **Two-phase schedule:** (A) encoder frozen, train heads; (B) unfreeze top-N
  encoder blocks, fine-tune at low LR. `EarlyStopping` + `ModelCheckpoint`.
- **Reproducibility:** global seed (properly *called*), deterministic splits,
  device-aware.
- **Artifact:** training writes a single bundle to `artifacts/` — a Keras
  **SavedModel**, the class label map (JSON), and a metadata sidecar (input
  size, metrics, config). The API loads exactly this bundle.

## Backend — FastAPI

| Concern | Choice | Why |
|---|---|---|
| Web framework | **FastAPI** + Uvicorn | Async, typed (Pydantic), auto OpenAPI docs. Loads the artifact at startup. |
| Inference | **TensorFlow** (predict) | One forward pass returns mask + class probabilities. |
| Heavy deps | **lazy-imported** | TF is imported inside functions so the app and tests load light. |

### Endpoints (planned)

- `POST /api/predict` — multipart image upload → `{ predicted_class, confidence,
  top_k: [...], mask_png_base64 }` (mask returned as a PNG overlay-ready image).
- `GET  /api/classes` — the 28 class labels.
- `GET  /api/health` — liveness; reports whether a model artifact is loaded.

## Frontend — React 18 + Vite

| Concern | Choice | Why |
|---|---|---|
| Build tool | **Vite** | Fast dev server, minimal config. |
| UI | **React 18**, native `fetch` | Drag/drop or pick a leaf image → shows the original with the **predicted mask overlaid** + the disease label, confidence, and a top-3 list. Dark, hand-written CSS (no component library). |

## Containerization & tooling

| Concern | Choice | Why |
|---|---|---|
| Orchestration | **docker-compose** | One command to serve. Frontend nginx serves the SPA and reverse-proxies `/api` to the backend (same-origin, no CORS). |
| Model in Docker | shared **`artifacts/` volume** | A `trainer` step trains into the volume; `backend` serves from it. `/api/health` reports `model_loaded: false` until trained. CPU-only TF keeps the image lean. |
| Tests | **pytest** | Runs fully offline on tiny **synthetic** images/masks — no real dataset, no real training. Validates the data pipeline shapes/weights, model wiring (two outputs, correct shapes), losses/metrics, and the API with a stubbed model. |

> **No secrets required.** This project calls no external API, so there's no key
> to manage.

## Project layout (planned)

```
leaf-disease-multitask/
├── constitution/            # mission, tech-stack, roadmap
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI app + routes
│   │   ├── config.py        # typed settings / hyperparameters / data paths
│   │   ├── schemas.py       # Pydantic response models
│   │   ├── data.py          # tf.data pipelines (both datasets) + unified weighted loader
│   │   ├── model.py         # build_multitask_model() shared encoder + 2 heads
│   │   ├── losses.py        # Dice/BCE seg loss; IoU & Dice metrics
│   │   ├── train.py         # two-phase training -> writes artifact
│   │   ├── evaluate.py      # held-out eval: seg IoU/Dice + classification report
│   │   └── inference.py     # load artifact, predict mask + class for one image
│   ├── scripts/predict_cli.py
│   ├── tests/
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── Dockerfile
├── frontend/                # React + Vite UI (+ nginx.conf, Dockerfile)
├── scripts/smoke_test.sh
├── docker-compose.yml
├── .env.example
└── README.md
```

## Local development

- Train: `python -m app.train` (reads the sibling data dirs, writes `artifacts/`).
- Backend: `uvicorn app.main:app --reload` on `:8000`.
- Frontend: `npm run dev` on `:5173`, proxying `/api` to the backend.
- Predict from CLI: `python -m scripts.predict_cli path/to/leaf.jpg`.

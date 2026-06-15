# Mission

## Product

**LeafLens** is a deep-learning plant-leaf analyzer built in **TensorFlow /
Keras**. A user uploads a single photo of a crop leaf and gets two answers from
**one model in one forward pass**:

1. **Where the leaf is** — a pixel mask separating leaf from background
   (segmentation), shown as an overlay.
2. **What's wrong with it** — the predicted disease / leaf type out of 28
   classes (classification), with a confidence score.

It ships as a Python training package, a FastAPI inference service, and a small
React UI (upload → mask overlay + disease label).

## The problem it solves

The original project (`LeafDiseaseDetection-DL`, a Jupyter notebook) treated
these as **two unrelated models**: a U-Net that segments leaves, and a separate
MobileNetV2 classifier for disease. They shared nothing — not weights, not a
pipeline, not an inference path. That is wasteful (two backbones to train and
serve) and misses the obvious synergy: knowing *where the leaf is* should help
*diagnose it*, and a shared visual backbone should learn richer features from
both signals.

**LeafLens unifies them into a single multi-task network.** This is the
substance of the rewrite.

## Modeling approach (what makes this a *deep learning* project)

- **Multi-task learning with a shared encoder.** A single **MobileNetV2**
  backbone (ImageNet-pretrained) feeds two heads:
  - a **U-Net-style decoder** with skip connections → a 224×224 binary leaf mask
    (`sigmoid`);
  - a **classification head** (global average pooling → dropout → `Dense(28,
    softmax)`) → the disease label.
  One `model(image)` returns both `{mask, class}`.
- **Training across two disjoint datasets.** The segmentation images carry masks
  but no class label; the classification images carry a class but no mask. We
  train on **both** using **per-head sample weights**: each image contributes
  loss only to the head it actually has a label for, while the shared encoder
  learns from every image. This is the core technique and the reason the model
  is genuinely unified rather than two networks bolted together.
- **Two-phase transfer learning.** Phase A trains the heads with the encoder
  frozen; Phase B fine-tunes the top encoder blocks at a low learning rate.
- **Honest, task-appropriate metrics.** Segmentation reported with **IoU** and
  **Dice** (not just pixel accuracy, which is misleading for masks);
  classification reported with **accuracy + a per-class report** on a real
  held-out split — fixing the original notebook's train/test leakage.

## Who it's for

- **Primary (portfolio context):** ML/AI recruiters and engineers evaluating
  applied deep-learning ability — specifically multi-task model design in
  TensorFlow, a correct training/eval pipeline over heterogeneous data, and the
  ability to serve the model behind a real API + UI.
- **Illustrative end user:** a grower or gardener who photographs a leaf and
  wants it localized and diagnosed.

## What success looks like

1. A clean, from-functions **multi-task Keras model** that trains on both
   datasets at once and reports IoU/Dice (segmentation) and accuracy
   (classification) on a held-out set.
2. A visitor opens the app, uploads a leaf, and within seconds sees the leaf
   masked and a disease prediction with confidence — no ML knowledge required.
3. The codebase reads as production-shaped ML: a reproducible `tf.data`
   pipeline, a single serialized artifact (SavedModel + label map + metadata), a
   typed inference API, and an offline test suite.

## Guiding principles

- **One model, two tasks.** The shared encoder and dual heads are the whole
  point — keep them genuinely shared.
- **Correctness first.** Real held-out split (no leakage), softmax + categorical
  cross-entropy for the 28-class head, task-masked losses, IoU/Dice for masks.
- **Reproducibility.** Seeded runs, config-driven hyperparameters, one
  serialized artifact the API loads.
- **Portfolio-legible.** Readable, well-separated modules and named functions
  over the notebook's copy-pasted blocks.

## Explicit non-goals

- Not a from-scratch backbone — transfer learning (MobileNetV2) is deliberate
  and appropriate for the dataset size.
- Not multi-tenant or auth-gated; single-user demo.
- Not a distributed / multi-GPU training project — it trains on a single machine.
- v1 does not require the two heads to agree on the *same* image during training
  (the datasets are disjoint); joint-labeled data is a stretch goal.

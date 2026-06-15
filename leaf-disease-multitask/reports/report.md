# LeafLens — training report

- **Model version:** 0.1.0
- **Created (UTC):** 2026-06-15T16:53:56.529137+00:00
- **Classes:** 28
- **Config:** batch_size=16, head_epochs=5, finetune_epochs=3, head_lr=0.001, finetune_lr=1e-05, seg_sample_ratio=0.5, augment=True, use_class_weights=True

## Validation metrics (mixed multi-task stream)

| metric | value |
|---|---|
| classification_accuracy | 0.4036 |
| classification_loss | 0.5951 |
| loss | 0.9129 |
| segmentation_dice | 0.6642 |
| segmentation_iou | 0.5823 |
| segmentation_loss | 0.3239 |

## Held-out evaluation

**Segmentation (val split):** IoU 0.5823, Dice 0.6642 over 705 images.
**Classification (held-out test):** accuracy 0.3686 over 236 images.

### Per-class (precision / recall / f1)

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| Apple Scab Leaf | 0.38 | 0.50 | 0.43 | 10 |
| Apple leaf | 0.30 | 0.78 | 0.44 | 9 |
| Apple rust leaf | 0.43 | 0.60 | 0.50 | 10 |
| Bell_pepper leaf | 0.00 | 0.00 | 0.00 | 8 |
| Bell_pepper leaf spot | 0.50 | 0.22 | 0.31 | 9 |
| Blueberry leaf | 0.22 | 0.18 | 0.20 | 11 |
| Cherry leaf | 0.15 | 0.20 | 0.17 | 10 |
| Corn Gray leaf spot | 0.27 | 0.75 | 0.40 | 4 |
| Corn leaf blight | 0.67 | 0.17 | 0.27 | 12 |
| Corn rust leaf | 0.75 | 0.90 | 0.82 | 10 |
| Peach leaf | 0.40 | 0.44 | 0.42 | 9 |
| Potato leaf early blight | 0.33 | 0.38 | 0.35 | 8 |
| Potato leaf late blight | 0.00 | 0.00 | 0.00 | 8 |
| Raspberry leaf | 0.50 | 0.29 | 0.36 | 7 |
| Soyabean leaf | 0.33 | 0.50 | 0.40 | 8 |
| Squash Powdery mildew leaf | 0.71 | 0.83 | 0.77 | 6 |
| Strawberry leaf | 0.57 | 1.00 | 0.73 | 8 |
| Tomato Early blight leaf | 0.10 | 0.11 | 0.11 | 9 |
| Tomato Septoria leaf spot | 0.25 | 0.27 | 0.26 | 11 |
| Tomato leaf | 0.50 | 0.12 | 0.20 | 8 |
| Tomato leaf bacterial spot | 0.11 | 0.11 | 0.11 | 9 |
| Tomato leaf late blight | 0.33 | 0.40 | 0.36 | 10 |
| Tomato leaf mosaic virus | 0.00 | 0.00 | 0.00 | 10 |
| Tomato leaf yellow virus | 0.50 | 0.50 | 0.50 | 6 |
| Tomato mold leaf | 0.25 | 0.17 | 0.20 | 6 |
| grape leaf | 0.70 | 0.58 | 0.64 | 12 |
| grape leaf black rot | 0.50 | 0.25 | 0.33 | 8 |
| macro avg | 0.36 | 0.38 | 0.34 | 236 |
| weighted avg | 0.36 | 0.37 | 0.34 | 236 |

## Training curves

![training curves](training_curves.png)

## Confusion matrix

![confusion matrix](confusion_matrix.png)

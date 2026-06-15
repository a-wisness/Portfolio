"""The multi-task model: one shared encoder, two heads.

A single MobileNetV2 encoder (ImageNet-pretrained) feeds:
  * a U-Net-style decoder -> ``segmentation`` output (224x224x1, sigmoid mask)
  * a global-pooling head    -> ``classification`` output (num_classes, softmax)

``model(image)`` returns ``{"segmentation": mask, "classification": probs}`` in
one forward pass. Input is raw RGB in [0, 255]; the model rescales internally to
MobileNetV2's expected [-1, 1] range, so callers never need to preprocess.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers

ENCODER_NAME = "mobilenetv2_encoder"

# Skip-connection taps, coarse->fine spatial size for a 224x224 input:
#   block_1_expand_relu  112x112
#   block_3_expand_relu   56x56
#   block_6_expand_relu   28x28
#   block_13_expand_relu  14x14
#   block_16_project       7x7  (bottleneck)
_SKIP_LAYERS = [
    "block_1_expand_relu",
    "block_3_expand_relu",
    "block_6_expand_relu",
    "block_13_expand_relu",
    "block_16_project",
]


def _upsample(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """Transposed-conv upsample block (x2 spatial), BN + ReLU."""
    x = layers.Conv2DTranspose(
        filters, 3, strides=2, padding="same", use_bias=False, name=f"{name}_deconv"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.ReLU(name=f"{name}_relu")(x)
    return x


def build_encoder(
    input_shape: tuple[int, int, int], weights: str | None = "imagenet"
) -> tf.keras.Model:
    """MobileNetV2 backbone exposing the skip-tap feature maps.

    ``weights=None`` builds an un-pretrained backbone (used by the offline tests
    so they never need to download ImageNet weights)."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights=weights
    )
    outputs = [base.get_layer(name).output for name in _SKIP_LAYERS]
    return tf.keras.Model(inputs=base.input, outputs=outputs, name=ENCODER_NAME)


def build_multitask_model(
    num_classes: int,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    dropout: float = 0.2,
    encoder_weights: str | None = "imagenet",
) -> tf.keras.Model:
    """Build the shared-encoder, dual-head model."""
    inputs = layers.Input(shape=input_shape, name="image")
    # MobileNetV2 expects inputs in [-1, 1]; do it inside the graph.
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="preprocess")(inputs)

    encoder = build_encoder(input_shape, weights=encoder_weights)
    skips = encoder(x)              # [112, 56, 28, 14, 7]
    bottleneck = skips[-1]          # 7x7

    # --- Classification head (off the bottleneck) ------------------------
    c = layers.GlobalAveragePooling2D(name="cls_gap")(bottleneck)
    c = layers.Dropout(dropout, name="cls_dropout")(c)
    classification = layers.Dense(
        num_classes, activation="softmax", name="classification"
    )(c)

    # --- Segmentation decoder head (U-Net, with skips) -------------------
    decoder_skips = list(reversed(skips[:-1]))   # [14, 28, 56, 112]
    up_filters = [512, 256, 128, 64]
    s = bottleneck
    for i, (filters, skip) in enumerate(zip(up_filters, decoder_skips)):
        s = _upsample(s, filters, name=f"dec{i}")
        s = layers.Concatenate(name=f"dec{i}_concat")([s, skip])
    s = _upsample(s, 32, name="dec_final")        # 112 -> 224
    segmentation = layers.Conv2D(
        1, 3, padding="same", activation="sigmoid", name="segmentation"
    )(s)

    return tf.keras.Model(
        inputs=inputs,
        outputs={"segmentation": segmentation, "classification": classification},
        name="leaflens_multitask",
    )


def set_encoder_trainable(
    model: tf.keras.Model, trainable: bool, fine_tune_at: int = 0
) -> None:
    """Freeze/unfreeze the shared encoder for two-phase transfer learning.

    When ``trainable`` is True and ``fine_tune_at`` > 0, only encoder layers from
    that index upward are unfrozen (the deeper, more task-specific blocks).
    BatchNormalization layers are kept frozen to preserve ImageNet statistics.
    """
    encoder = model.get_layer(ENCODER_NAME)
    encoder.trainable = trainable
    if not trainable:
        return
    for i, layer in enumerate(encoder.layers):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = i >= fine_tune_at

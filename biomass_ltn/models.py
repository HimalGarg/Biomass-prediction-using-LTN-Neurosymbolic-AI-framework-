from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers

from .config import TARGET_NAMES


def build_multimodal_regressor(
    tabular_dim: int,
    image_size: int = 128,
    dropout: float = 0.20,
    backbone: str = "small_cnn",
    mobilenet_weights: str | None = None,
    output_bias: list[float] | None = None,
) -> tf.keras.Model:
    """Build a shared image+tabular regressor for the five biomass targets."""
    image_input = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    tabular_input = tf.keras.Input(shape=(tabular_dim,), name="tabular")

    image_features = _image_encoder(
        image_input,
        dropout=dropout,
        backbone=backbone,
        mobilenet_weights=mobilenet_weights,
    )
    tabular_features = _tabular_encoder(tabular_input, dropout=dropout)

    fused = layers.Concatenate(name="fusion")([image_features, tabular_features])
    fused = layers.Dense(256, activation="elu")(fused)
    fused = layers.BatchNormalization()(fused)
    fused = layers.Dropout(dropout)(fused)
    shared = layers.Dense(128, activation="elu", name="shared_representation")(fused)
    shared = layers.Dropout(dropout / 2.0)(shared)

    outputs = layers.Dense(
        len(TARGET_NAMES),
        activation="softplus",
        bias_initializer=_output_bias_initializer(output_bias),
        name="biomass_normalized",
    )(shared)

    return tf.keras.Model(
        inputs=[image_input, tabular_input],
        outputs=outputs,
        name="biomass_multimodal_regressor",
    )


def build_feature_extractor(model: tf.keras.Model) -> tf.keras.Model:
    return tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer("shared_representation").output,
        name=f"{model.name}_features",
    )


def _output_bias_initializer(output_bias: list[float] | None):
    if output_bias is None:
        return "zeros"
    return tf.keras.initializers.Constant(output_bias)


def _image_encoder(
    image_input: tf.Tensor,
    dropout: float,
    backbone: str,
    mobilenet_weights: str | None,
) -> tf.Tensor:
    if backbone == "mobilenetv2":
        weights = None if mobilenet_weights in (None, "none") else mobilenet_weights
        base = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights=weights,
            pooling="avg",
        )
        base.trainable = False if weights else True
        x = layers.Lambda(
            lambda tensor: tf.keras.applications.mobilenet_v2.preprocess_input(tensor * 255.0),
            name="mobilenetv2_preprocess",
        )(image_input)
        x = base(x)
        return layers.Dense(128, activation="elu", name="image_embedding")(x)

    if backbone != "small_cnn":
        raise ValueError(f"Unknown backbone: {backbone}")

    x = image_input
    for filters in (24, 48, 96, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("elu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    return layers.Dense(128, activation="elu", name="image_embedding")(x)


def _tabular_encoder(tabular_input: tf.Tensor, dropout: float) -> tf.Tensor:
    x = layers.Dense(96, activation="elu")(tabular_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="elu", name="tabular_embedding")(x)
    return x

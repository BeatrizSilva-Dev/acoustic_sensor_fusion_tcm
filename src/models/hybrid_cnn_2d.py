import os
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    SeparableConv2D,
    BatchNormalization,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    concatenate, ReLU
)
from tensorflow.keras.initializers import HeNormal

from src.models.cnn_branch import build_cnn_branch


os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.experimental.enable_op_determinism()
except:
    pass

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    SeparableConv2D,
    BatchNormalization,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense,
    Dropout
)

from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import l2


def build_robust_cnn_architecture(input_shape):

    inputs = Input(
        shape=input_shape,
        name="input_spectrogram"
    )

    x = SeparableConv2D(
        filters=16,
        kernel_size=(3,3),
        padding="same",
        activation="relu",
        depthwise_initializer=HeNormal(),
        pointwise_initializer=HeNormal(),
        depthwise_regularizer=l2(1e-4),
        pointwise_regularizer=l2(1e-4)
    )(inputs)

    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2,2))(x)


    x = SeparableConv2D(
        filters=32,
        kernel_size=(3,3),
        padding="same",
        activation="relu",
        depthwise_initializer=HeNormal(),
        pointwise_initializer=HeNormal(),
        depthwise_regularizer=l2(1e-4),
        pointwise_regularizer=l2(1e-4)
    )(x)

    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2,2))(x)


    x = SeparableConv2D(
        filters=64,
        kernel_size=(3,3),
        padding="same",
        activation=None,
        name="last_conv",
        depthwise_initializer=HeNormal(),
        pointwise_initializer=HeNormal(),
        depthwise_regularizer=l2(1e-4),
        pointwise_regularizer=l2(1e-4)
    )(x)

    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(
        64,
        activation="relu",
        kernel_initializer=HeNormal(),
        kernel_regularizer=l2(1e-4)
    )(x)

    x = Dropout(0.30)(x)

    outputs = Dense(
        1,
        activation="sigmoid"
    )(x)

    model = Model(
        inputs,
        outputs,
        name="cnn_sensor"
    )

    return model


def build_late_fusion_architecture(input_shape):

    input_internal = Input(shape=input_shape, name="internal_input")
    input_external = Input(shape=input_shape, name="external_input")

    internal_branch = build_cnn_branch(
        input_shape,
        "internal_branch"
    )

    external_branch = build_cnn_branch(
        input_shape,
        "external_branch"
    )

    feature_internal = internal_branch(input_internal)
    feature_external = external_branch(input_external)

    fused = concatenate([feature_internal, feature_external])

    x = Dense(
        64,
        activation="relu",
        kernel_initializer=HeNormal(),
        kernel_regularizer=l2(1e-4)
    )(fused)

    x = Dropout(0.30)(x)

    output = Dense(
        1,
        activation="sigmoid"
    )(x)

    return Model(
        [input_internal, input_external],
        output
    )
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    SeparableConv2D,
    BatchNormalization,
    MaxPooling2D,
    ReLU,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    concatenate
)
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import l2

def build_cnn_branch(input_shape, branch_name):

    inp = Input(shape=input_shape,
                name=f"{branch_name}_input")

    x = SeparableConv2D(
        16,
        (3,3),
        padding="same",
        activation="relu",
        depthwise_initializer=HeNormal(),
        pointwise_initializer=HeNormal(),
        depthwise_regularizer=l2(1e-4),
        pointwise_regularizer=l2(1e-4)
    )(inp)

    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    x = SeparableConv2D(
        32,
        (3,3),
        padding="same",
        activation="relu",
        depthwise_initializer=HeNormal(),
        pointwise_initializer=HeNormal(),
        depthwise_regularizer=l2(1e-4),
        pointwise_regularizer=l2(1e-4)
    )(x)

    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    x = SeparableConv2D(
        64,
        (3,3),
        padding="same",
        activation=None,
        name=f"{branch_name}_last_conv",
        depthwise_initializer=HeNormal(),
        pointwise_initializer=HeNormal(),
        depthwise_regularizer=l2(1e-4),
        pointwise_regularizer=l2(1e-4)
    )(x)

    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = GlobalAveragePooling2D()(x)

    return Model(inp,
                 x,
                 name=branch_name)
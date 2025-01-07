import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate


def wide_deep_model(input_shape):
    # Wide component (linear model)
    wide_input = Input(shape=(input_shape,), name="wide_input")
    wide_output = Dense(1, activation="linear", name="wide_output")(wide_input)

    # Deep component (DNN)
    deep_input = Input(shape=(input_shape,), name="deep_input")
    deep_output = Dense(64, activation="relu")(deep_input)
    deep_output = Dense(32, activation="relu")(deep_output)
    deep_output = Dense(16, activation="relu")(deep_output)

    # Concatenate wide and deep components
    combine = Concatenate()([wide_output, deep_output])
    output = Dense(1, activation="linear", name="output")(combine)

    # Build model
    model = Model(inputs=[wide_input, deep_input], outputs=output)

    return model

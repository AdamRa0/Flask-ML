from .constants import (
    FILTER_SIZE_FOUR,
    FILTER_SIZE_THREE,
    FILTER_SIZE_TWO,
    FILTER_SIZE_ONE,
    FINAL_LAYER_ACTIVATION_FUNCTION,
    NUM_UNITS_IN_DENSE_LAYERS,
    NUM_UNITS_IN_FINAL_DENSE_LAYER,
    OUTPUT_NODES,
    INPUT_SHAPE,
)

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    MaxPool2D,
    Dropout,
    BatchNormalization,
)


def create_model():
    model = Sequential(
        [
            Conv2D(FILTER_SIZE_ONE, (3, 3), activation="relu"),
            BatchNormalization(),
            Conv2D(FILTER_SIZE_ONE, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPool2D(pool_size=(2, 2)),
            Conv2D(FILTER_SIZE_TWO, (3, 3), input_shape=INPUT_SHAPE, activation="relu"),
            BatchNormalization(),
            Conv2D(FILTER_SIZE_TWO, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPool2D(pool_size=(2, 2)),
            Conv2D(FILTER_SIZE_THREE, (3, 3), activation="relu"),
            BatchNormalization(),
            Conv2D(FILTER_SIZE_THREE, (3, 3), activation="relu"),
            BatchNormalization(),
            Conv2D(FILTER_SIZE_THREE, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPool2D(pool_size=(2, 2)),
            Conv2D(FILTER_SIZE_FOUR, (3, 3), activation="relu"),
            BatchNormalization(),
            Conv2D(FILTER_SIZE_FOUR, (3, 3), activation="relu"),
            BatchNormalization(),
            Conv2D(FILTER_SIZE_FOUR, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPool2D(pool_size=(2, 2)),
            Conv2D(FILTER_SIZE_FOUR, (3, 3), activation="relu"),
            BatchNormalization(),
            Conv2D(FILTER_SIZE_FOUR, (3, 3), activation="relu"),
            BatchNormalization(),
            Conv2D(FILTER_SIZE_FOUR, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPool2D(pool_size=(2, 2)),
            Flatten(),
            Dense(units=NUM_UNITS_IN_DENSE_LAYERS, activation="relu"),
            Dropout(0.5),
            Dense(units=NUM_UNITS_IN_DENSE_LAYERS, activation="relu"),
            Dropout(0.5),
            Dense(units=NUM_UNITS_IN_FINAL_DENSE_LAYER, activation="relu"),
            Dropout(0.5),
            Dense(units=OUTPUT_NODES, activation=FINAL_LAYER_ACTIVATION_FUNCTION),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    model.build(INPUT_SHAPE)

    model.summary()

    return model

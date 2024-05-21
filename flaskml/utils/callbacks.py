import tensorflow as tf


full_model_early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="accuracy",
    patience=3
)
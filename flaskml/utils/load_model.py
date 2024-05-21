import tensorflow as tf


def load_model(model_path):
  print(f"Loading saved model from: {model_path}")
  model = tf.keras.models.load_model(
      model_path
  )
  return model
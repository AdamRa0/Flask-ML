import tensorflow as tf
from .constants import IMG_SIZE


def preprocess_image(file_path: str):
  """
  file_path: File to image
  Returns preprocessed image
  """

  image = tf.io.read_file(file_path)
  image = tf.io.decode_image(image, channels=3, expand_animations=False)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
  image = tf.reshape(image, [IMG_SIZE, IMG_SIZE, 3])

  return image
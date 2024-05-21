from .preprocess_image import preprocess_image
from .constants import BATCH_SIZE
from .return_image_label import return_image_label

import tensorflow as tf


def create_data_batches(
    X,
    y=None,
    test_data=False,
    valid_data=False
  ):
  """
  X: Feature values(Images)
  y: Label values
  test_data: Boolean value determining if to create data batches for test data
  valid_data: Boolean value determining if to create data batches for validation data

  Creates and returns data batches from X and/or y values
  """

  if test_data:
    print("Creating test data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))
    data_batch = data.map(preprocess_image).batch(BATCH_SIZE)
    return data_batch

  elif valid_data:
    print("Creating validation data batches...")
    data = tf.data.Dataset.from_tensor_slices(
        (
            tf.constant(X),
            tf.constant(y)
        )
    )
    data_batch = data.map(return_image_label).batch(BATCH_SIZE)
    return data_batch

  else:
    print("Creating training data batches...")
    data = tf.data.Dataset.from_tensor_slices(
        (
            tf.constant(X),
            tf.constant(y)
        )
    )
    data = data.shuffle(buffer_size=len(X))
    data = data.map(return_image_label)
    data_batch = data.batch(BATCH_SIZE)
    return data_batch
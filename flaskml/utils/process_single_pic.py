from .constants import BATCH_SIZE
from .preprocess_image import preprocess_image

import tensorflow as tf


def process_single_image(image):
    data = tf.data.Dataset.from_tensor_slices((tf.constant(image)))
    processed_image = data.map(preprocess_image).batch(BATCH_SIZE)
    return processed_image

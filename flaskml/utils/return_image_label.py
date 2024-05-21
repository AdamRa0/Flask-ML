from .preprocess_image import preprocess_image


def return_image_label(file_path: str, label: str):
  """
  file_path: Path to image
  label: Image label

  returns preprocessed image and label
  """
  image = preprocess_image(file_path)

  return image, label
from utils.append_data import append_data
from utils.create_data_batches import create_data_batches
from utils.create_data_list import create_data_list
from utils.shuffle_data_list import shuffle_data_list
from utils.create_train_test_dataset import create_train_test_datasets
from utils.create_model import create_model
from utils.save_model import save_model
from utils.constants import NUM_EPOCHS
from utils.callbacks import full_model_early_stopping


import matplotlib.pyplot as plt
import numpy as np


some_data: list[str] = append_data("<Replace with path to dataset>")
jpeg_extensions = ('.jpg', '.jpeg', '.JPG', '.JPEG')
some_data = [path for path in some_data if path.endswith(jpeg_extensions)]

general_data_list: list[tuple[str, str]] = create_data_list(some_data)

shuffled_data_list: list[tuple[str, str]] = shuffle_data_list(general_data_list)

labels: list[str] = [i[1] for i in shuffled_data_list]

unique_labels: list[str] = np.unique(labels)

boolean_labels: list[bool] = [label == unique_labels for label in labels] # will form our y values

  
def get_prediction_label(prediction_probabilities):
  """
  prediction_probabilities: Array of predictions from model

  returns predicted label
  """

  return unique_labels[np.argmax(prediction_probabilities)]


def plot_image_prediction_true_value(prediction_probabilities, X, y, index):
  """
  prediction_probabilities: Array of predictions from model
  X: image to be plotted
  y: true label
  index: Position of image, prediction and true label you wish to show

  plot actual image with the predicted value and true values as title to image plot
  """
  image, true_label, predicted_probabilities = X[index], y[index], prediction_probabilities[index]
  predicted_label = get_prediction_label(predicted_probabilities)

  probability_index: int = np.argmax(predicted_probabilities)

  plt.imshow(image)
  plt.yticks([])
  plt.xticks([])
  plt.title(
      f"Predicted Label: {predicted_label}  True Label: {true_label}\n"
      f"Probability image is predicted label: {predicted_probabilities[probability_index] * 100:2.0f}%")


def unbatchify(batched_dataset):
  """
  returns list of images and lables from unbatched dataset
  """
  images_ = []
  labels_ = []

  for image, label in batched_dataset.unbatch().as_numpy_iterator():
    images_.append(image)
    labels_.append(unique_labels[np.argmax(label)])

  return images_, labels_

X = [i[0] for i in shuffled_data_list]

full_model_X = X[:5000]

full_model_y = boolean_labels[:5000]

train_X, test_X, train_y, _ = create_train_test_datasets(full_model_X, full_model_y)

train_data = create_data_batches(
  train_X,
  train_y,
)

test_data = create_data_batches(
  test_X,
  test_data=True,
)

full_model = create_model()

full_model.fit(
  x=train_data,
  epochs=NUM_EPOCHS,
  callbacks=[full_model_early_stopping],
)

save_model(full_model, "full-data-VGG-16-Adam-Optimizer-Model")
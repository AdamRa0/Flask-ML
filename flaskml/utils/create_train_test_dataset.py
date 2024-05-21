from sklearn.model_selection import train_test_split


def create_train_test_datasets(X, y, TEST_SIZE=0.2, RANDOM_STATE=42):
  """
  X: Feature values
  y: Label values
  TEST_SIZE: Percentage of test dataset from original. Default is 20%
  RANDOM_STATE: Random integer. Default is 42

  Creates train and test datasets from X and y values.
  Can also be used to create train and validation datasets
  """
  X_train, X_test, y_train, y_test = train_test_split(
      X,
      y,
      test_size=TEST_SIZE,
      random_state=RANDOM_STATE
    )

  return X_train, X_test, y_train, y_test
from sklearn.utils import shuffle


def shuffle_data_list(data_list: list[tuple[str, str]]) -> list[tuple[str, str]]:
  return shuffle(data_list)
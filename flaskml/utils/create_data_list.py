def create_data_list(general_data: list[str]) -> list[tuple[str, str]]:
  """
  data_path: path to folder containing desired files
  label: string argument for the desired label

  Creates and returns a list of tuples containing filepath and desired label
  """
  #List that will store our files and their labels
  dataset: list[tuple[str, str]] = []

  for path in general_data:
    if "/Cat/" in path:
      dataset.append((path, "CAT"))
    if "/Dog/" in path:
      dataset.append((path, "DOG"))
  return dataset
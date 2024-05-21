import os
import tqdm


def append_data(FILE_PATH: str) -> list[str]:
  data_list: list[str] = []
  for root, dirs, files in tqdm(os.walk(FILE_PATH)):
    for f in files:
      data_list.append(f"{root}/{f}")

  return data_list
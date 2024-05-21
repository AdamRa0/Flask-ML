import os
import datetime


def save_model(model, suffix=None) -> str:
  modeldir = os.path.join(
      'flaskml/models',
      datetime.datetime.now().strftime("%Y%m%d-%H%M%s")
  )
  model_path = modeldir + "-" + suffix + ".keras"
  print(f"Saving model to: {model_path}...")
  model.save(model_path)
  return model_path
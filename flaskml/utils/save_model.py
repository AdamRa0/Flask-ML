from flaskml import create_app

import os
import datetime
from flask import Flask


def save_model(model, suffix=None) -> str:
  app: Flask = create_app()

  modeldir = os.path.join(
      app.config["MODEL_FOLDER"],
      datetime.datetime.now().strftime("%Y%m%d-%H%M%s")
  )
  model_path = modeldir + "-" + suffix + ".keras"
  print(f"Saving model to: {model_path}...")
  model.save(model_path)
  return model_path
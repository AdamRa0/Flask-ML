from flask import Flask, render_template, request, flash, send_from_directory
from werkzeug.utils import secure_filename

from .utils.process_single_pic import process_single_image
from .utils.load_model import load_model

import numpy as np
import os

UPLOAD_FOLDER: str = "uploads"
ALLOWED_EXTENSIONS: set[str] = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def create_app():
    app = Flask(__name__)
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

    if os.path.exists(app.config['UPLOAD_FOLDER']):
        pass
    else:
        os.mkdir(app.config['UPLOAD_FOLDER'])

    @app.route("/", methods=["GET", "POST"])
    def home():
        chance: str = ""
        file_label: str = ""


        if request.method == "POST":
            if "file" not in request.files:
                flash("No file part")

            file = request.files["file"]
            file_label = request.form["label"]

            if file.filename == "":
                flash("No selected file")

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

                processed_data = process_single_image(file)
                model = load_model("path to saved model")
                predictions = model.predict(processed_data)
                chance = f"{predictions[np.argmax(predictions)] * 100:2.0f}"

        return render_template("home.html", p=chance, label=file_label)
    
    # For some reason url not found
    @app.route("/uploads/<filename>")
    def inflate_images(filename=""):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    return app

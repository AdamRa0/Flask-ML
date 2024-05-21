from flask import Flask

def create_app():
    app = Flask(__name__)

    @app.route("/")
    def home():
        return "Home"
    

    @app.route("/predict")
    def predict():
        pass

    return app
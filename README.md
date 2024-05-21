# Flask-ML
Flask API making use of a neural network to classify Cat and Dog images.

Dataset: [Kaggle Cats vs Dogs](https://www.microsoft.com/en-us/download/details.aspx?id=54765)

Requirements:
- Python 3.10

To run the app, clone the project, cd into it and run the following commands:
```bash
python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

flask --app flaskml run
```

The model is created inside the model.py file. To create your model, run this file seperately.
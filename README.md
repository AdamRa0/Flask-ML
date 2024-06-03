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

You however first, need to create the model, train it and save it.
To load it, replace the text, path to saved model, with the path to your saved model. This is done inside flaskml/__init__.py

The model is created inside the model.py file. To create your model, run this file seperately.

Replace the text, replace with path to dataset, with the path to your dataset

## How to run the application on kubernetes
Run the following command
```bash
docker build . -t your-dockerhub-username/flaskml-image
```

Deploy image to dockerhub

Replace the image value in flaskml-deployment.yaml with the name of the image you deployed to dockerhub

If on minikube, start it up and run the following commands

```bash
kubectl apply -f flaskml-deployment.yaml

# To run the service responsible for our application
kubectl service flaskml-service

# Should that fail
kubectl port-forward service/flaskml-service 2500:8000
# Visit localhost:2500 and your application should be available
```
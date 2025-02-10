import io
import os
import pickle
from dataclasses import Field
from urllib import request
import mlflow.pyfunc
import numpy as np

import pandas as pd
from PIL import Image
from charset_normalizer import models
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn

tags = [
    {
        'name': 'Maths',
        'description': 'Operations related to mathematics',
    },
    {
        'name': 'Models',
        'description': 'Operations related to models',
    },
    {
        'name': 'Form',
        'description': 'Operations related to forms',
    }
]

app = FastAPI(
    title="My FastAPI API",
    description="This is a simple app",
    version="0.1.0",
    #openap_tags
)


@app.get("/", tags=['Models'])
def default_root():
    return {"Hello": "World"}


@app.get("/square", tags=['Maths'])
def square(n: int = 1) -> str:
    n = n * n
    return f"Le résultat au carré est {n}"


from pydantic import BaseModel


class Data(BaseModel):
    name: str
    city: str


@app.post("/formulaire", tags=["Form"])
def formulaire(data: Data):
    data = dict(data)
    # NAME
    name = data['name']
    # CITY
    city = data['city']
    return f"Hello {name} from {city}."


@app.post("/upload", tags=["Form"])
def upload(file: UploadFile = File(...)):
    return file.filename


class InputData(BaseModel):
    Gender: str
    Age: float
    Graduated: str
    Profession: str
    Work_Experience: float
    Spending_Score: str
    Family_Size: float
    Segmentation: str


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.post("/predict", tags=['Model'])
def predict(data: InputData):
    input_data = pd.DataFrame([data.dict()])
    prediction = model.predict(input_data)[0]
    if prediction == 0:
        return "Not married"
    else:
        return "Married"


@app.post("/predict_file", tags=['Model'])
def predict_file(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    if 'Gender' not in df.columns or 'Graduated' not in df.columns:
        return False
    else:
        x = df.drop(["Ever_Married"], axis=1).dropna()
        y_pref = model.predict(x)
        print(y_pref)
        return [int(n) for n in model.predict(x)]


try:
    os.environ['AWS_ACCESS_KEY_ID'] = "AKIA3R62MVALLDMAF37Q"
    os.environ['AWS_SECRET_ACCESS_KEY'] = "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"
    mlflow.set_tracking_uri("https://quera-server-mlflow-cda209265623.herokuapp.com/")

    path = mlflow.MlflowClient().get_registered_model('Ever_Married').latest_versions[0].source
    model = mlflow.pyfunc.load_model(path)
    print("Modèle chargé avec succès depuis MLflow.")
except Exception as e:
    model = None
    print(f"Erreur lors du chargement du modèle : {e}")


@app.post("/predict-model", tags=["Model"])
def predict(data: InputData):
    """
    Point de terminaison pour effectuer une prédiction.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Le modèle n'est pas chargé.")

    try:
        # Convertir les données d'entrée en DataFrame Pandas
        input_data = pd.DataFrame([data.dict()])

        # Effectuer la prédiction avec le modèle chargé depuis MLflow
        prediction = model.predict(input_data)

        if prediction == 0:
            return "Not married"
        else:
            return "Married"

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction : {str(e)}")



# Endpoint pour prédire un chiffre manuscrit à partir d'une image
@app.post("/predict_digit")
async def predict_digit(file: UploadFile = File(...)):

    """
    Point de terminaison pour effectuer une prédiction à partir d'une image.
    """
    try:
        os.environ['AWS_ACCESS_KEY_ID'] = "AKIA3R62MVALLDMAF37Q"
        os.environ['AWS_SECRET_ACCESS_KEY'] = "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"
        mlflow.set_tracking_uri("https://quera-server-mlflow-cda209265623.herokuapp.com/")

        path = mlflow.MlflowClient().get_registered_model('jade_model_mnist').latest_versions[0].source
        model = mlflow.pyfunc.load_model(path)
        print("Modèle chargé avec succès depuis MLflow.")
    except Exception as e:
        model = None
        print(f"Erreur lors du chargement du modèle : {e}")
    if model is None:
        raise HTTPException(status_code=500, detail="Le modèle n'est pas chargé.")

    try:
        # Charger l'image envoyée par l'utilisateur
        image = Image.open(file.file).convert("L")  # Convertir en niveaux de gris

        # Redimensionner l'image à 28x28 pixels (taille attendue par MNIST)
        image_resized = image.resize((28, 28))

        # Normaliser les pixels entre 0 et 1 et aplatir l'image en un vecteur 1D
        image_array = np.array(image_resized).reshape(1, -1) / 255.0

        # Effectuer la prédiction avec le modèle chargé depuis MLflow
        prediction = model.predict(image_array)
        predicted_class = int(prediction[0])

        return {"predicted_class": predicted_class}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction : {str(e)}")

@app.post("/predict_digit_cnn")
async def predict_digit_cnn(file: UploadFile = File(...)):

    """
    Point de terminaison pour effectuer une prédiction à partir d'une image.
    """
    try:
        os.environ['AWS_ACCESS_KEY_ID'] = "AKIA3R62MVALLDMAF37Q"
        os.environ['AWS_SECRET_ACCESS_KEY'] = "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"
        mlflow.set_tracking_uri("https://quera-server-mlflow-cda209265623.herokuapp.com/")

        path = mlflow.MlflowClient().get_registered_model('jade_domasvasserot_cnn ').latest_versions[0].source
        model = mlflow.pyfunc.load_model(path)
        print("Modèle chargé avec succès depuis MLflow.")
    except Exception as e:
        model = None
        print(f"Erreur lors du chargement du modèle : {e}")
    if model is None:
        raise HTTPException(status_code=500, detail="Le modèle n'est pas chargé.")

    try:
        # Charger l'image envoyée par l'utilisateur
        image = Image.open(file.file).convert("L")  # Convertir en niveaux de gris
        print(image)
        # Redimensionner l'image à 28x28 pixels (taille attendue par MNIST)
        image_resized = image.resize((28, 28))

        # Normaliser les pixels entre 0 et 1 et aplatir l'image en un vecteur 1D
        image_array = np.array(image_resized).reshape(1, -1) / 255.0

        # Effectuer la prédiction avec le modèle chargé depuis MLflow
        prediction = model.predict(image_array)
        print(prediction)
        predicted_class = int(prediction[0])

        return {"predicted_class": predicted_class}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction : {str(e)}")
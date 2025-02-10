import os
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np

# Configuration de MLflow
os.environ['AWS_ACCESS_KEY_ID'] = "AKIA3R62MVALLDMAF37Q"
os.environ['AWS_SECRET_ACCESS_KEY'] = "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"
mlflow.set_tracking_uri("https://quera-server-mlflow-cda209265623.herokuapp.com/")
experiment = mlflow.set_experiment("MNIST Classification - Jade DOMAS-VASSEROT")
print(experiment.experiment_id)

# Chargement des données MNIST
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data
y = mnist.target.astype(int)

# Prétraitement des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Séparation en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# Infer signature : obtention des informations sur les colonnes en entrée
signature = infer_signature(X_train, y_train)
# Entraînement du modèle avec journalisation dans MLflow
with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="MNIST Classification - Jade DOMAS-VASSEROT"):
    # Modèle Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Évaluation du modèle
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy score : {accuracy}")
    print(f"train score : {model.score(X_test, y_test)}")

    # Journalisation des métriques et hyperparamètres dans MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("n_estimators", 100)
    # Sauvegarder les données d'entraînement dans un fichier CSV
    train_data = pd.concat([X, y.rename("label")], axis=1)
    train_data.to_csv("mnist_train_data.csv", index=False)
    print("Données d'entraînement sauvegardées dans 'mnist_train_data.csv'.")
    mlflow.log_artifact("mnist_train_data.csv")
    X_train_df = pd.DataFrame(X_train)
    # Fournir un exemple d'entrée pour le modèle
    input_example = X_train_df.head(1)
    # Enregistrement du modèle
    mlflow.sklearn.log_model(
        model,
        "Jade Model Minist",
        signature=signature,
        input_example=input_example,
        registered_model_name='jade_model_mnist'
    )
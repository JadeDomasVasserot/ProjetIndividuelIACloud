import os
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from mlflow.models.signature import infer_signature

# Configuration de MLflow
# CONFIGURER DANS HEROKU OU .ENV
#os.environ['AWS_ACCESS_KEY_ID'] = "AWS_ACCESS_KEY_ID"
#os.environ['AWS_SECRET_ACCESS_KEY'] = "AWS_SECRET_ACCESS_KEY"
mlflow.set_tracking_uri("https://quera-server-mlflow-cda209265623.herokuapp.com/")
experiment = mlflow.set_experiment("MNIST CNN - Jade DOMAS-VASSEROT")
print(f"Experiment ID: {experiment.experiment_id}")

# Chargement des données MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Prétraitement des données
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0  # Normalisation entre 0 et 1
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Encodage one-hot des labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Définir le modèle séquentiel convolutionnel
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # Couche de sortie pour la classification multiclasse
])

# Compiler le modèle
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Résumé du modèle
model.summary()

# Entraînement et journalisation avec MLflow
with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="CNN MNIST"):
    # Activer l'autologging pour TensorFlow
    mlflow.tensorflow.autolog()

    # Entraîner le modèle
    history = model.fit(
        X_train,
        y_train,
        batch_size=128,
        epochs=10,
        validation_split=0.2,
        verbose=1
    )

    # Évaluer le modèle sur l'ensemble de test
    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    # Journalisation manuelle des métriques et du modèle dans MLflow (si nécessaire)
    input_example = X_train[:1]  # Exemple d'entrée pour la signature du modèle
    signature = infer_signature(input_example, model.predict(input_example))

    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_accuracy", test_accuracy)

    # Enregistrer le modèle dans MLflow avec signature et exemple d'entrée
    mlflow.tensorflow.log_model(
        tf_meta_graph_tags=None,
        tf_signature_def_key=None,
        artifact_path="mnist_cnn_model",
        registered_model_name="jade_mnist_cnn",
        signature=signature,
        input_example=input_example,
    )

FROM continuumio/miniconda3

# Définir le répertoire de travail
WORKDIR /root/airflow

# Copier et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier les dossiers et fichiers nécessaires
COPY dags/ dags/
COPY plugins/ plugins/
COPY train.py .
COPY airflow.db .

# Définir des arguments pour passer les clés AWS
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

# Définir les variables d'environnement à partir des arguments
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY



# Commande pour démarrer Airflow
CMD airflow standalone


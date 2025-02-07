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

# Définir les variables d'environnement
ENV AWS_ACCESS_KEY_ID=AKIA3R62MVALLDMAF37Q
ENV AWS_SECRET_ACCESS_KEY=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


# Commande pour démarrer Airflow
CMD airflow standalone


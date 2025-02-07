# Utiliser l'image Miniconda comme base
FROM continuumio/miniconda3

# Définir le répertoire de travail dans le conteneur
WORKDIR /home/app

# Copier les fichiers nécessaires dans le conteneur
COPY api.py .
COPY model.pkl .
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port utilisé par Uvicorn (par défaut 8000)
EXPOSE 8000
ENV PORT 8000
# Commande pour démarrer l'application
CMD [ "sh", "-c", "uvicorn api:app --host=0.0.0.0 --port=${PORT:-8000}"]
# Définir la variable d'environnement PORT pour Heroku



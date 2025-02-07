
# Nom de l'image et du conteneur
IMAGE_NAME="mlflow-server"
CONTAINER_NAME="mlflow-server"

# Chemin vers le Dockerfile
DOCKERFILE_PATH="./Dockerfile"

# Étape 1 : Construire l'image Docker
echo "Construction de l'image Docker..."
docker build -t $IMAGE_NAME -f $DOCKERFILE_PATH .
if [ $? -ne 0 ]; then
    echo "Erreur lors de la construction de l'image Docker."
    exit 1
fi

# Étape 2 : Vérifier si un conteneur avec le même nom existe déjà
EXISTING_CONTAINER=$(docker ps -a -q -f name=$CONTAINER_NAME)
if [ ! -z "$EXISTING_CONTAINER" ]; then
    echo "Un conteneur avec le nom $CONTAINER_NAME existe déjà. Suppression..."
    docker rm -f $CONTAINER_NAME
fi

# Étape 3 : Lancer un conteneur à partir de l'image
echo "Lancement du conteneur..."
docker run --name $CONTAINER_NAME -d $IMAGE_NAME
if [ $? -ne 0 ]; then
    echo "Erreur lors du lancement du conteneur."
    exit 1
fi

# Étape 4 : Afficher les logs du conteneur (optionnel)
echo "Affichage des logs du conteneur :"
docker logs $CONTAINER_NAME

echo "Conteneur $CONTAINER_NAME lancé avec succès."

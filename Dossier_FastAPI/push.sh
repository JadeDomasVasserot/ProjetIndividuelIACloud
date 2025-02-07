
# Nom de l'image Docker
IMAGE_NAME="ynov-fastapi"

# Construire l'image Docker
docker build . -t $IMAGE_NAME

# Exécuter l'image Docker en local avec un volume
docker run -p 8000:8000 -e PORT=8000 -v "$(pwd):/home/app"-it $IMAGE_NAME

# Vérifier si le run a réussi
if [ $? -eq 0 ]; then
    echo "L'image Docker a été exécutée avec succès en local."
else
    echo "L'exécution de l'image Docker en local a échoué."
    exit 1
fi

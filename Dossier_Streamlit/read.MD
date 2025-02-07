## Commandes utiles

- Démarrer l'application : streamlit run nomFichier.py

https://share.streamlit.io/signup

créer une clef API dans setting 
API_KEY = "123"

docker build . -t ynov-streamlit    
docker run -p 8001:8001 -e PORT=8001 -v "$(pwd):/home/app"-it ynov-streamlit

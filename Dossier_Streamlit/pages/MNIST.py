import streamlit as st
import requests

# Titre de l'application
st.title("Prédiction MNIST avec FastAPI")

# Téléversement d'un fichier
uploaded_file = st.file_uploader("Téléchargez une image MNIST au format PNG", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Afficher l'image téléversée
    st.image(uploaded_file, caption="Image téléversée")
    col1, col2 = st.columns(2)
    with col1:
        # Bouton pour envoyer le fichier à l'API
        if st.button("Prédire Randomforest"):
            # Envoyer le fichier à l'API FastAPI
            try:
                files = {"file": uploaded_file.getvalue()}
                response = requests.post("https://ynov-api-jade-c06a85ee7a38.herokuapp.com/predict_digit", files=files)

                # Vérifier la réponse de l'API
                if response.status_code == 200:
                    prediction = response.json()
                    st.success(f"Prédiction : {prediction['predicted_class']}")
                else:
                    st.error(f"Erreur API : {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Erreur lors de la requête : {e}")
    with col2:
    # Bouton pour envoyer le fichier à l'API
        if st.button("Prédire CNN"):
            # Envoyer le fichier à l'API FastAPI
            try:
                files = {"file": uploaded_file.getvalue()}
                response = requests.post("https://ynov-api-jade-c06a85ee7a38.herokuapp.com/predict_digit_cnn", files=files)

                # Vérifier la réponse de l'API
                if response.status_code == 200:
                    prediction = response.json()
                    st.success(f"Prédiction : {prediction['predicted_class']}")
                else:
                    st.error(f"Erreur API : {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Erreur lors de la requête : {e}")

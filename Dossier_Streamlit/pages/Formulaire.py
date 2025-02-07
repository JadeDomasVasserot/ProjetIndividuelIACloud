import streamlit as st
import requests

st.title("Formulaire de Prédiction")
professions = ['Healthcare', 'Engineer', 'Lawyer', 'Entertainment', 'Artist', 'Executive',
               'Doctor', 'Homemaker', 'Marketing']
# Champs du formulaire
gender = st.selectbox("Genre", ["Male", "Female"])
age = st.number_input("Âge", min_value=18, max_value=89)
graduated = st.selectbox("Diplômé ?", ["Yes", "No"])
profession = st.selectbox("Profession ?", professions)
work_experience = st.number_input("Expérience professionnelle (en années)", min_value=0.0)
spending_score = st.selectbox("Score de dépenses", ["Low", "Average", "High"])
family_size = st.number_input("Taille de la famille", min_value=1, max_value=100)
segmentation = st.selectbox("Segmentation", ["A", "B", "C", "D"])

# Soumettre les données au modèle via l'API
if st.button("Prédire"):
    input_data = {
        "Gender": gender,
        "Age": age,
        "Graduated": graduated,
        "Profession": profession,
        "Work_Experience": work_experience,
        "Spending_Score": spending_score,
        "Family_Size": family_size,
        "Segmentation": segmentation,
    }
    response = requests.post("https://ynov-api-jade-c06a85ee7a38.herokuapp.com/predict", json=input_data)
    if response.status_code == 200:
        st.success(f"Résultat : {response.text}")
    else:
        st.error(f"Erreur : {response.status_code}")

import os

import streamlit as st
import pandas as pd

# https://streamlit.io



# Configurer la page
st.set_page_config(
    page_title="My Dashbord",
    page_icon="💶",
    layout="wide",
)
path = 'https://raw.githubusercontent.com/Quera-fr/Python-Programming/refs/heads/main/data.csv'


@st.cache_data
def load_data():
    return pd.read_csv(path)


df = load_data()

# Ajout API Key
try:
    st.sidebar().write(st.secrets['API_KEY'])
except:
    st.error("No API key found.")
# clef secret sur heroku
try:
    os.environ['API_KEY']
except KeyError:
    st.error("No API key found.")


# Ajouter un titre
st.title('My Dashboard')

# Ajouter un sous-titre
st.subheader('Présentation des données')

# Ajouter du texte en markdown
st.write("Présentation de données avec Streamlit")
# Checkbox
bool_form = st.checkbox("Afficher le formulaire")
if bool_form:
    # Créer un champ de saisie
    name = st.text_input("Entrer votre nom")

    # Les variables sont dynamiques
    st.write(f'Bonjour {name}')
    # Afficher DF
    st.write(df)

# Image Vidéo
st.sidebar.image('https://logodownload.org/wp-content/uploads/2019/09/lyon-logo-0.png', width=300)
st.sidebar.video('https://youtu.be/8gV3v8lEmJU')

# SelectBox


import seaborn as sns

with st.form(key="my_form"):
    col1, col2 = st.columns(2)
    with col1:
        profession = st.selectbox("Sélectionner une option ", options=df.Profession.unique())
        # SLider
        age_slider = st.slider("Sélectionner un âge", min_value=df.Age.min(), max_value=df.Age.max(),
                               value=(df.Age.min(), df.Age.max()))
        # filtre age selon profession
        st.write(str(age_slider))
        button = st.form_submit_button("Envoyer")
    with col2:
        data_age = df[(df.Profession == profession) & (df.Age > age_slider[0]) & (df.Age < age_slider[1])].Age
        if button:
            plot = sns.histplot(data=data_age, bins=age_slider[1] - age_slider[0])
            # Afficher un plot
            st.pyplot(plot.figure)
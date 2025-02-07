from io import BytesIO

import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import seaborn as sns

# Ajouter un titre
st.title('Interface - Projet Jade DOMAS-VASSEROT')

st.sidebar.header("Chargement du CSV")
uploaded_file = st.sidebar.file_uploader("Choisissez votre fichier .csv", type="csv")
if uploaded_file:
    # read csv
    df = pd.read_csv(uploaded_file)
    colums = df.columns
    if df is not None:
        # select columns
        st.sidebar.header("Sélection des colonnes")
        selected_columns = st.sidebar.multiselect("Sélectionner les colonnes à télécharger/modifier ou visualiser",
                                                  colums)
        st.write("Vous avez la possibilité d'éditer à la volée le dataframe ")
        if selected_columns:
            # data editor
            edited_df = st.data_editor(df[selected_columns])
            st.download_button(label="Télécharger votre nouveau fichier modifié", data=edited_df.to_csv(),
                               file_name="data_modifie.csv", mime='text/csv')
            st.subheader("Visualiasation des graphiques")
            type_graphique = ["Aucun", "Histogramme", "Graphique à bar", "Boîte à moustache", "Nuage de point", "Camenbert"]
            char_type = st.selectbox("Choisissez le type de graphique :", type_graphique)
            if char_type != "Aucun":
                st.write("### Graphique")
                if char_type == "Histogramme":
                    columns = st.selectbox("Sélectionner les X ", options=selected_columns)
                    if columns:
                        plt.figure()
                        sns.histplot(data=df, x=columns)
                        plt.title(f"Histogramme de {columns}")
                        st.pyplot(plt)
                elif char_type == "Graphique à bar":
                    columns_x = st.selectbox("Sélectionner les X ", options=selected_columns)
                    column_y = st.selectbox("Sélectionner les Y ", options=selected_columns)
                    if column_y and columns_x:
                        plt.figure()
                        sns.barplot(data=df, x=columns_x, y=column_y)
                        plt.title(f"Graphique à barre des {columns_x} et {column_y}")
                        st.pyplot(plt)
                elif char_type == "Boîte à moustache":
                    columns = st.selectbox("Sélectionner les X ", options=selected_columns)
                    if columns:
                        plt.figure()
                        sns.boxplot(data=df, x=columns)
                        st.pyplot(plt)
                elif char_type == "Camenbert":
                    columns = st.selectbox("Sélectionner les X ", options=selected_columns)
                    plt.figure()
                    values = df[columns].value_counts()
                    colors = sns.color_palette("pastel")
                    plt.pie(x=values, labels=values.index, colors=colors)
                    st.pyplot(plt)
                elif char_type == "Nuage de point":
                    columns_x = st.selectbox("Sélectionner les X ", options=selected_columns)
                    column_y = st.selectbox("Sélectionner les Y ", options=selected_columns)
                    if column_y and columns_x:
                        plt.figure()
                        sns.scatterplot(data=df, x=columns_x, y=column_y)
                        st.pyplot(plt)
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                st.download_button(label="Télécharger image", data=buffer, file_name="image.png", mime='image/png')

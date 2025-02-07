import streamlit as st
import pandas as pd

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # read csv
    df = pd.read_csv(uploaded_file, delimiter=",")

    colums = df.columns

    selected_columns = st.multiselect("Select columns", colums)

    # data editor
    edited_df = st.data_editor(df[selected_columns])

    # download button
    st.download_button(label="Download as CSV", data=edited_df.to_csv(), file_name='data.csv', mime='text/csv')

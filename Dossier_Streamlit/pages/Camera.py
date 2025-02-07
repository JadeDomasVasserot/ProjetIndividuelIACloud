
import streamlit as st

# Configurer la page
st.set_page_config(
    page_title="Camera",
    page_icon="📹ss",
    layout="wide",
)
col_cam, cam = st.columns(2)
with col_cam:
    enable = st.checkbox("Enable camera")
    picture = st.camera_input("Take a picture", disabled=not enable)

with cam:
    if picture:
        st.image(picture)
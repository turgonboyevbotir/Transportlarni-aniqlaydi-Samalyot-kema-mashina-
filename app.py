import streamlit as st
from fastai.vision.all import *
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.title('Transport klassifikatsiya qiluvchi app')

file = st.file_uploader('Rasmni yuklang', type=['jpeg', 'svg', 'gif', 'png', 'jpg'])


if file:
    img = PILImage.create(file)

    model = load_learner('transport_model.pkl')

    pred, pred_id, probs = model.predict(img)

    st.image(file)

    st.success(f"Bashorat : {pred}")
    st.info(f"Ehtimollik : {probs[pred_id]*100:.1f}%")


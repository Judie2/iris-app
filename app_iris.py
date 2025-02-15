import streamlit as st
import numpy as np
import pandas as pd ;
import joblib


#Titre de la page
st.title("IRIS PREDICTION APP")


st.write("Veuillez signifier les caracterisques de la fleur")

# Creation du menu
st.sidebar.header("MENU")

# Créer des sections 
st.sidebar.button("Prédiction")
st.sidebar.button("Visualisation")
st.sidebar.button("Contact")
st.sidebar.button("Help")

PetalLength =st.slider("PetalLength", 0.0, 10.0, value=2.0, step=0.1)
SepalLength = st.slider("SepalLength", 0.0, 10.0, value=0.2, step=0.1)
PetalWidth =st.slider("PetalWidth ", 0.0, 10.0, value=2.0, step=0.1)
SepalWidth = st.slider("SepalWidth", 0.0, 10.0, value=0.3, step=0.1)
    
    
    
if st.button("Prediction sur la fleur", help="Cliquez pour envoyer les donnees", type="primary"): 
    model= joblib.load("mon_model2.pkl")
    scaler= joblib.load("mon_scaler.pkl")
    
    caracteristiques = np.array([[SepalLength, SepalWidth, PetalLength, PetalWidth]])
    X = pd.DataFrame(caracteristiques,columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'])
    
    X_transform = scaler.transform(X)
    prediction = model.predict(X_transform)
    
    response = prediction[0]

 
    st.write(
        "<p style='font-size: 20px; font-weight: bold;'>Votre fleur est : "
        "<span style='color: #ff4b4b;'>" + response + "</span></p>",
        unsafe_allow_html=True
    )


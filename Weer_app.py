import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Charger le modèle entraîné
@st.cache_data
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Charger les encoders
@st.cache_data
def load_encoders():
    with open("label_encoders.pkl", "rb") as file:
        encoders = pickle.load(file)
    return encoders

model = load_model()
encoders = load_encoders()

st.title("Système d'Orientation Médicale")
st.write("Entrez vos informations pour obtenir un conseil médical.")

# Entrées utilisateur
sexe = st.selectbox("Sexe", encoders["Sexe"].classes_)
âge = st.number_input("Âge", min_value=0, max_value=120, value=30)
symptôme = st.selectbox("Symptôme", encoders["Symptômes"].classes_)
maladie_diagnostiquee = st.selectbox("Maladie Diagnostiquée", encoders["Maladie Diagnostiquée"].classes_)
score_gravite = st.slider("Score de Gravité", 1, 5, 3)
triage_urgence = st.slider("Triage Urgence", 1, 5, 3)

# Convertir les entrées utilisateur en format numérique
sexe_encoded = encoders["Sexe"].transform([sexe])[0]
symptôme_encoded = encoders["Symptômes"].transform([symptôme])[0]
maladie_encoded = encoders["Maladie Diagnostiquée"].transform([maladie_diagnostiquee])[0]

# Préparer les données pour la prédiction
features = [[âge, sexe_encoded, symptôme_encoded, maladie_encoded, score_gravite, triage_urgence]]

if st.button("Obtenir un conseil"):
    prediction = model.predict(features)[0]
    conseil = encoders["Conseil"].inverse_transform([prediction])[0]
    st.write(f"### Conseil Médical : {conseil}")

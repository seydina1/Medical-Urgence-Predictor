# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:36:17 2025

@author: seydi
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:37:52 2025

@author: seydi
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pprint
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle


# fichier Excel
file_path = "BDD_Patients_Urgences_1000.xlsx"
xls = pd.ExcelFile(file_path)
# Vérifier les feuilles disponibles
print(xls.sheet_names)
# Charger la feuille de données
df = pd.read_excel(xls, sheet_name="Sheet1")
# Afficher les premières lignes
print(df.head())

# Colonnes catégoriques à encoder
categorical_columns = ["Sexe", "Symptômes", "Maladie Diagnostiquée", "Conseil"]

# Appliquer l'encodage LabelEncoder sur ces colonnes
"""
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
"""
# Colonnes à encoder
categorical_columns = ["Sexe", "Symptômes", "Maladie Diagnostiquée", "Conseil"]

# Appliquer l'encodage LabelEncoder et stocker les correspondances
label_encoders = {}
column_mappings = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    column_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

# Vérifier le résultat après encodage
print(df.head())


# Définir les features et la cible
X = df.drop(columns=["ID Patient", "Conseil"])  # Variables explicatives
y = df["Conseil"]  # Variable cible

# Séparer en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser et entraîner le modèle RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Prédire sur les données de test
y_pred = model.predict(X_test)
# Évaluer la performance
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
import pickle

# Sauvegarder le modèle
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

# Sauvegarder les encodeurs
with open("label_encoders.pkl", "wb") as file:
    pickle.dump(label_encoders, file)

print("Modèle et encodeurs sauvegardés avec succès !")


import streamlit as st
from datetime import datetime
import pandas as pd
import random

# Titre principal
st.title("Rapport de Topographie")

# Informations générales
st.header("Informations Générales")
col1, col2 = st.columns(2)
with col1:
    date_rapport = st.date_input("Date du rapport", datetime.today())
    heure_rapport = st.time_input("Heure du rapport", datetime.now().time())
    numero_rapport = st.text_input("Numéro du rapport", "RPT-001")
with col2:
    lieu = st.text_input("Lieu du relevé", "Site Alpha")
    titre_rapport = st.text_input("Titre du rapport", "Étude topographique du terrain X")

# Affichage du rapport
st.header("Aperçu du Rapport")
st.subheader("Informations Générales")
st.write(f"**Date du rapport :** {date_rapport}")
st.write(f"**Heure du rapport :** {heure_rapport}")
st.write(f"**Numéro du rapport :** {numero_rapport}")
st.write(f"**Lieu :** {lieu}")
st.write(f"**Titre :** {titre_rapport}")

st.subheader("Cartes et Plans")
if uploaded_maps:
    for uploaded_map in uploaded_maps:
        st.image(uploaded_map, caption=uploaded_map.name, use_column_width=True)
else:
    st.write("[Espace réservé pour les cartes et plans]")

st.subheader("Profils Topographiques")
if uploaded_profiles:
    for uploaded_profile in uploaded_profiles:
        st.image(uploaded_profile, caption=uploaded_profile.name, use_column_width=True)
else:
    st.write("[Espace réservé pour les profils topographiques]")

st.subheader("Légendes et Explications")
st.write(legendes if legendes else "[Aucune légende fournie]")

st.subheader("Informations Techniques")
st.write(f"**Nombre de points relevés :** {nb_points}")
st.write(f"**Précision des relevés :** {precision} cm")
st.write(f"**Altitude moyenne :** {altitude_moyenne} m")

st.subheader("Données relevées")
st.dataframe(df)

# Bouton d'export
st.header("Export du rapport")
if st.button("Générer le rapport PDF"):
    st.success("Le rapport a été généré avec succès !")

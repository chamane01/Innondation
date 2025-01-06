import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd

# Configuration de la page
st.set_page_config(page_title="Projet Inondation", layout="wide")

# Affichage du titre
st.title("Carte des Défauts Routiers")

# Exemple : Charger des données fictives depuis un DataFrame (remplacer par tes données réelles)
data = pd.DataFrame([
    {"id": 1, "type": "Nid de poule", "latitude": 5.354, "longitude": -4.008},
    {"id": 2, "type": "Fissure", "latitude": 5.356, "longitude": -4.010},
])

# Création de la carte avec Folium
map_center = [5.354, -4.008]  # Coordonnées initiales (ex. Abidjan)
m = folium.Map(location=map_center, zoom_start=12)

# Ajouter les marqueurs sur la carte
for _, row in data.iterrows():
    folium.Marker(
        location=[row["latitude"], row["longitude"]],
        popup=f"Type : {row['type']}",
    ).add_to(m)

# Intégration de la carte dans Streamlit
st_map = st_folium(m, width=800, height=600)

# Afficher une légende ou des infos supplémentaires
st.sidebar.title("Filtres")
st.sidebar.write("Ajoutez ici des options pour filtrer ou explorer les défauts.")

# Importer les bibliothèques nécessaires
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import Polygon

# Ajouter un logo en haut
st.image("POPOPO.jpg", width=200)

# Streamlit - Titre de l'application
st.title("Carte des zones inondées avec niveaux d'eau et surface")

# Espace de carte interactif affiché même avant téléversement de fichier
st.subheader("Carte interactive avec différentes couches")

# Création d'une carte interactive avec les couches de base (monde)
map_center = [0, 0]  # Centré sur le monde
m = folium.Map(location=map_center, zoom_start=2)

# Ajouter différentes couches de fond (OpenStreetMap, satellite, etc.)
folium.TileLayer('OpenStreetMap', name="OpenStreetMap").add_to(m)
folium.TileLayer('Stamen Terrain', name="Topographique", 
                 attr="Stamen Terrain").add_to(m)
folium.TileLayer('Stamen Toner', name="Toner", 
                 attr="Stamen Toner").add_to(m)
folium.TileLayer('Stamen Watercolor', name="Watercolor", 
                 attr="Stamen Watercolor").add_to(m)

# Ajouter une couche satellite avec attribution
folium.TileLayer(
    tiles="https://{s}.sat.owm.io/sql/base/{z}/{x}/{y}.png?appid=your_openweathermap_api_key",
    attr="Satellite tiles by OpenWeatherMap",
    name="Satellite"
).add_to(m)

# Ajout du contrôle de couches
folium.LayerControl().add_to(m)

# Afficher la carte dans Streamlit (carte monde sans relation avec le fichier CSV pour l'instant)
st_folium(m, width=700, height=500)

# Étape 1 : Téléverser le fichier Excel ou TXT
uploaded_file = st.file_uploader("Téléversez un fichier Excel ou TXT", type=["xlsx", "txt"])

# Affichage de la carte de base d'inondation et autres traitements après téléversement du fichier
if uploaded_file is not None:
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.txt'):
        df = pd.read_csv(uploaded_file, sep=",", header=None, names=["X", "Y", "Z"])

    # Vérification des colonnes
    if 'X' not in df.columns or 'Y' not in df.columns or 'Z' not in df.columns:
        st.error("Erreur : colonnes 'X', 'Y' et 'Z' manquantes.")
    else:
        # Traitement des données comme auparavant...
        pass

else:
    st.warning("Veuillez téléverser un fichier pour démarrer.")

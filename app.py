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

# Espace de carte interactif (monde) toujours visible
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

# Laisser un espace après la carte
st.markdown("---")  # Ligne séparatrice pour plus de clarté

# Section pour téléversement et traitement des données
st.subheader("Téléverser et traiter les données d'inondation")

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
        # Étape 5 : Paramètres du niveau d'inondation
        niveau_inondation = st.number_input("Entrez le niveau d'eau (mètres)", min_value=0.0, step=0.1)
        interpolation_method = st.selectbox("Méthode d'interpolation", ['linear', 'nearest'])

        # Étape 6 : Création de la grille
        X_min, X_max = df['X'].min(), df['X'].max()
        Y_min, Y_max = df['Y'].min(), df['Y'].max()

        resolution = st.number_input("Résolution de la grille", value=300, min_value=100, max_value=1000)
        grid_X, grid_Y = np.mgrid[X_min:X_max:resolution*1j, Y_min:Y_max:resolution*1j]
        grid_Z = griddata((df['X'], df['Y']), df['Z'], (grid_X, grid_Y), method=interpolation_method)

        # Étape 7 : Calcul de la surface inondée
        def calculer_surface(niveau_inondation):
            contour = plt.contour(grid_X, grid_Y, grid_Z, levels=[niveau_inondation], colors='none')
            paths = contour.collections[0].get_paths()
            surfaces = [Polygon(path.vertices).area for path in paths]
            return sum(surfaces) / 10000  # Retourne en hectares

        # Étape 8 : Calcul du volume d'eau
        def calculer_volume(niveau_inondation, surface_inondee):
            volume = surface_inondee * niveau_inondation * 10000  # Conversion en m³ (1 hectare = 10,000 m²)
            return volume

        surface_inondee = calculer_surface(niveau_inondation)
        volume_eau = calculer_volume(niveau_inondation, surface_inondee)

        # Étape 9 : Affichage de la carte des inondations
        st.write(f"Surface inondée : {surface_inondee:.2f} hectares")
        st.write(f"Volume d'eau : {volume_eau:.2f} m³")
        
        # Tracer la carte d'inondation (utilisez matplotlib ou une autre méthode pour créer votre carte ici)
        fig, ax = plt.subplots()
        ax.contourf(grid_X, grid_Y, grid_Z, cmap="Blues")
        st.pyplot(fig)

else:
    st.warning("Veuillez téléverser un fichier pour démarrer.")

# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import Polygon
import folium
from streamlit_folium import st_folium
from folium.plugins import MousePosition

# Titre de l'application et logo
st.set_page_config(page_title="Inondation et Cartographie", layout="wide")
st.title("Analyse d'Inondation et Carte Dynamique")
st.image("logo.png", width=200)  # Remplacez 'logo.png' par le chemin vers votre logo

# Téléversement du fichier
uploaded_file = st.file_uploader("Téléverser un fichier Excel ou TXT", type=["xlsx", "txt"])

# Fonction de création de la carte
def create_map(points):
    # Création de la carte centrée sur un point moyen des données
    center = [np.mean(points['Y']), np.mean(points['X'])]
    m = folium.Map(location=center, zoom_start=10)

    # Ajout des différents fonds de carte
    folium.TileLayer('OpenStreetMap').add_to(m)
    folium.TileLayer('Stamen Terrain').add_to(m)
    folium.TileLayer('Stamen Toner').add_to(m)
    folium.TileLayer('Satellite').add_to(m)

    # Ajout des points à la carte
    for _, row in points.iterrows():
        folium.Marker([row['Y'], row['X']], popup=f"Z: {row['Z']}").add_to(m)

    # Ajout de la position de la souris
    MousePosition().add_to(m)

    return m

# Vérification du fichier téléversé
if uploaded_file is not None:
    # Chargement des données
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.txt'):
        df = pd.read_csv(uploaded_file, sep=",", header=None, names=["X", "Y", "Z"])

    # Vérification des colonnes
    if 'X' in df.columns and 'Y' in df.columns and 'Z' in df.columns:
        st.success("Données chargées avec succès !")

        # Entrée du niveau d'inondation
        niveau_inondation = st.number_input("Entrez le niveau d'inondation (m)", value=1.0)

        # Méthode d'interpolation
        interpolation_method = st.selectbox("Méthode d'interpolation", ['linear', 'nearest'])

        # Création de la grille
        X_min, X_max = df['X'].min(), df['X'].max()
        Y_min, Y_max = df['Y'].min(), df['Y'].max()
        resolution = st.slider("Résolution de la grille", min_value=100, max_value=500, value=300)

        grid_X, grid_Y = np.mgrid[X_min:X_max:resolution*1j, Y_min:Y_max:resolution*1j]
        grid_Z = griddata((df['X'], df['Y']), df['Z'], (grid_X, grid_Y), method=interpolation_method)

        # Calcul de la surface inondée
        def calculer_surface(niveau_inondation):
            contour = plt.contour(grid_X, grid_Y, grid_Z, levels=[niveau_inondation], colors='none')
            paths = contour.collections[0].get_paths()
            surfaces = [Polygon(path.vertices).area for path in paths]
            return sum(surfaces) / 10000  # Retourne en hectares

        # Calcul du volume d'eau
        def calculer_volume(niveau_inondation, surface_inondee):
            return surface_inondee * niveau_inondation * 10000  # Conversion en m³

        surface_inondee = calculer_surface(niveau_inondation)
        volume_eau = calculer_volume(niveau_inondation, surface_inondee)

        # Affichage du rapport
        st.write(f"Surface inondée : {surface_inondee:.2f} hectares")
        st.write(f"Volume d'eau : {volume_eau:.2f} m³")

        # Tracé de la carte statique
        fig, ax = plt.subplots()
        contour = ax.contourf(grid_X, grid_Y, grid_Z, cmap='viridis', levels=100)
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('Profondeur (mètres)')
        ax.set_title("Carte de profondeur")
        st.pyplot(fig)

        # Affichage de la carte dynamique
        st.subheader("Carte interactive avec points")
        folium_map = create_map(df)
        st_folium(folium_map, width=800, height=600)
    else:
        st.error("Erreur : Colonnes X, Y et Z manquantes dans le fichier.")
else:
    st.info("Veuillez téléverser un fichier Excel ou TXT.")


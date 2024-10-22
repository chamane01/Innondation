# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import Polygon
import folium
from streamlit_folium import folium_static
import base64

# --- Fonctions utilitaires pour l'interface ---
def afficher_logo(path, width=200):
    # Fonction pour afficher le logo de l'entreprise
    with open(path, "rb") as file:
        img = file.read()
    encoded = base64.b64encode(img).decode()
    st.markdown(f"<img src='data:image/png;base64,{encoded}' width={width}>", unsafe_allow_html=True)

# --- Initialisation de l'application ---
st.set_page_config(page_title="Simulation de Carte Inondée", layout="wide")
st.title("Simulation de Carte Inondée")
afficher_logo("logo_entreprise.png")  # Remplacez par le chemin de votre logo

# --- Chargement des données ---
st.sidebar.header("Téléverser le fichier")
uploaded_file = st.sidebar.file_uploader("Choisissez un fichier Excel ou TXT", type=["xlsx", "txt"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.txt'):
        df = pd.read_csv(uploaded_file, sep=",", header=None, names=["X", "Y", "Z"])
    
    # Vérification des colonnes
    if 'X' in df.columns and 'Y' in df.columns and 'Z' in df.columns:
        st.success("Fichier chargé avec succès!")
    else:
        st.error("Erreur: colonnes 'X', 'Y', 'Z' manquantes.")
        st.stop()
else:
    st.warning("Veuillez téléverser un fichier pour continuer.")
    st.stop()

# --- Paramètres pour le niveau d'inondation ---
st.sidebar.header("Paramètres d'inondation")
niveau_inondation = st.sidebar.number_input("Niveau d'inondation (m)", value=1.0, step=0.1)
interpolation_method = st.sidebar.selectbox("Méthode d'interpolation", ['linear', 'nearest'], index=0)
resolution = st.sidebar.slider("Résolution de la grille", min_value=100, max_value=1000, value=300, step=100)

# --- Calcul des grilles et interpolation ---
X_min, X_max = df['X'].min(), df['X'].max()
Y_min, Y_max = df['Y'].min(), df['Y'].max()
grid_X, grid_Y = np.mgrid[X_min:X_max:resolution*1j, Y_min:Y_max:resolution*1j]
grid_Z = griddata((df['X'], df['Y']), df['Z'], (grid_X, grid_Y), method=interpolation_method)

# --- Calcul des surfaces et volumes ---
def calculer_surface(niveau_inondation):
    contour = plt.contour(grid_X, grid_Y, grid_Z, levels=[niveau_inondation], colors='none')
    paths = contour.collections[0].get_paths()
    surfaces = [Polygon(path.vertices).area for path in paths]
    return sum(surfaces) / 10000  # Surface en hectares

def calculer_volume(surface_inondee, niveau_inondation):
    return surface_inondee * niveau_inondation * 10000  # Volume en m³

# --- Affichage de la carte ---
def afficher_carte(df, niveau_inondation):
    # Coordonnées moyennes pour centrer la carte
    centre = [(df['Y'].mean(), df['X'].mean())]

    # Créer une carte Folium
    carte = folium.Map(location=centre[0], zoom_start=13, tiles="OpenStreetMap")

    # Ajouter des couches de carte
    folium.TileLayer('openstreetmap').add_to(carte)
    folium.TileLayer('stamenterrain').add_to(carte)
    folium.TileLayer('satellite').add_to(carte)
    folium.LayerControl().add_to(carte)

    # Ajout des points du fichier
    for i, row in df.iterrows():
        folium.CircleMarker(
            location=[row['Y'], row['X']],
            radius=5,
            fill=True,
            fill_color='blue',
            color='blue',
            fill_opacity=0.7,
            popup=f"Altitude: {row['Z']} m"
        ).add_to(carte)

    folium_static(carte)

# --- Afficher les résultats de simulation ---
surface_inondee = calculer_surface(niveau_inondation)
volume_eau = calculer_volume(surface_inondee, niveau_inondation)

# Afficher la carte dynamique
st.header("Carte interactive des points et zones inondées")
afficher_carte(df, niveau_inondation)

# Résultats
st.subheader("Résultats de la simulation")
st.write(f"Surface inondée : **{surface_inondee:.2f} hectares**")
st.write(f"Volume d'eau estimé : **{volume_eau:.2f} m³**")

# --- Afficher les coordonnées projetées (option) ---
if st.checkbox("Afficher le tableau des coordonnées XYZ"):
    st.subheader("Tableau des coordonnées")
    st.dataframe(df)


# Importer les bibliothèques nécessaires
import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from shapely.geometry import Polygon

# Streamlit - Titre de l'application avec logo
st.image("POPOPO.jpg", width=150)
st.title("Carte des zones inondées avec niveaux d'eau et surface")

# Étape 1 : Téléverser le fichier Excel ou TXT
uploaded_file = st.file_uploader("Téléversez un fichier Excel ou TXT", type=["xlsx", "txt"])

if uploaded_file is not None:
    # Étape 2 : Identifier le type de fichier et charger les données en fonction
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.txt'):
        df = pd.read_csv(uploaded_file, sep=",", header=None, names=["X", "Y", "Z"])

    # Séparateur pour organiser l'affichage
    st.markdown("---")  # Ligne de séparation

    # Étape 3 : Vérification du fichier
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
            # Détection des contours de la zone inondée
            contour = plt.contour(grid_X, grid_Y, grid_Z, levels=[niveau_inondation], colors='red')
            paths = contour.collections[0].get_paths()
            polygons = [Polygon(path.vertices) for path in paths]

            # Calculer la surface totale des polygones
            surfaces = [polygon.area for polygon in polygons]
            return sum(surfaces) / 10000  # Retourne en hectares

        # Étape 8 : Calcul du volume d'eau
        def calculer_volume(niveau_inondation, surface_inondee):
            volume = surface_inondee * niveau_inondation * 10000  # Conversion en m³ (1 hectare = 10,000 m²)
            return volume

        surface_inondee = calculer_surface(niveau_inondation)
        volume_eau = calculer_volume(niveau_inondation, surface_inondee)

        # Étape 9 : Affichage des résultats
        st.write(f"Surface inondée : {surface_inondee:.2f} hectares")
        st.write(f"Volume d'eau : {volume_eau:.2f} m³")

else:
    st.warning("Veuillez téléverser un fichier pour démarrer.")

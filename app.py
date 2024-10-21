# Importer les bibliothèques nécessaires
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from shapely.geometry import Polygon, LineString

# Streamlit - Titre de l'application
st.title("Carte des zones inondées avec niveaux d'eau et surface")

# Étape 1 : Téléverser le fichier Excel ou TXT
uploaded_file = st.file_uploader("Téléversez un fichier Excel ou TXT", type=["xlsx", "txt"])

if uploaded_file is not None:
    # Étape 2 : Identifier le type de fichier et charger les données en fonction
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.txt'):
        df = pd.read_csv(uploaded_file, sep=",", header=None, names=["X", "Y", "Z"])

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

        # Étape 7 : Calcul de la surface inondée basée sur les contours fermés
        def calculer_surface_polygones(niveau_inondation):
            contour = plt.contour(grid_X, grid_Y, grid_Z, levels=[niveau_inondation], colors='none')
            paths = contour.collections[0].get_paths()

            # Transformation des chemins (contours) en polygones fermés
            polygones = []
            for path in paths:
                # Extraire les points du contour
                vertices = path.vertices
                if len(vertices) > 2:  # Il faut au moins 3 points pour un polygone
                    # Créer un polygone avec les sommets
                    polygon = Polygon(vertices)
                    if polygon.is_valid and polygon.area > 0:  # S'assurer que le polygone est valide
                        polygones.append(polygon)

            # Calcul de la surface totale en hectares (1 unité de surface = 10,000 m²)
            surface_totale = sum(p.area for p in polygones) / 10000
            return surface_totale, polygones

        surface_inondee, polygones_inondes = calculer_surface_polygones(niveau_inondation)

        # Étape 8 : Calcul du volume d'eau
        def calculer_volume(niveau_inondation, surface_inondee):
            volume = surface_inondee * niveau_inondation * 10000  # Conversion en m³ (1 hectare = 10,000 m²)
            return volume

        volume_eau = calculer_volume(niveau_inondation, surface_inondee)

        # Étape 9 : Fonction pour tracer la carte avec contours actuels et polygones
        def plot_map_with_polygons(niveau_inondation, surface_inondee, volume_eau, polygones):
            plt.close('all')

            # Taille ajustée pour la carte
            fig, ax_map = plt.subplots(figsize=(8, 6))

            # Tracé de la carte de profondeur
            contour = ax_map.contourf(grid_X, grid_Y, grid_Z, cmap='viridis', levels=100)
            cbar = fig.colorbar(contour, ax=ax_map)
            cbar.set_label('Profondeur (mètres)')

            # Tracé des contours actuels du niveau d'inondation
            contours_inondation = ax_map.contour(grid_X, grid_Y, grid_Z, levels=[niveau_inondation], colors='red', linewidths=2)
            ax_map.clabel(contours_inondation, inline=True, fontsize=10, fmt='%1.1f m')

            # Tracé des polygones inondés
            for polygon in polygones:
                patch = plt.Polygon(polygon.exterior.coords, facecolor='blue', edgecolor='blue', alpha=0.5)
                ax_map.add_patch(patch)

            ax_map.set_title("Carte des zones inondées avec polygones fermés")
            ax_map.set_xlabel("Coordonnée X")
            ax_map.set_ylabel("Coordonnée Y")

            # Affichage
            st.pyplot(fig)

        # Étape 10 : Affichage de la carte avec les polygones et le rapport
        if st.button("Afficher la carte"):
            plot_map_with_polygons(niveau_inondation, surface_inondee, volume_eau, polygones_inondes)
            st.write(f"Surface inondée : {surface_inondee:.2f} hectares")
            st.write(f"Volume d'eau : {volume_eau:.2f} m³")

else:
    st.warning("Veuillez téléverser un fichier pour démarrer.")

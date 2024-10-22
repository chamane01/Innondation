# Importer les bibliothèques nécessaires
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from shapely.geometry import Polygon, MultiPolygon

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
        # Afficher les premiers enregistrements pour vérification
        st.write("Aperçu des données :")
        st.dataframe(df.head())

        # Étape 5 : Paramètres du niveau d'inondation
        niveau_inondation = st.number_input("Entrez le niveau d'eau (mètres)", min_value=0.0, step=0.1)
        interpolation_method = st.selectbox("Méthode d'interpolation", ['linear', 'nearest'])

        # Étape 6 : Création de la grille
        X_min, X_max = df['X'].min(), df['X'].max()
        Y_min, Y_max = df['Y'].min(), df['Y'].max()

        resolution = st.number_input("Résolution de la grille", value=300, min_value=100, max_value=1000)
        grid_X, grid_Y = np.mgrid[X_min:X_max:resolution*1j, Y_min:Y_max:resolution*1j]
        grid_Z = griddata((df['X'], df['Y']), df['Z'], (grid_X, grid_Y), method=interpolation_method)

        # Vérification des valeurs de la grille
        st.write("Grille de profondeur (exemple) :")
        st.write(grid_Z)

        # Étape 7 : Calcul de la surface inondée et des polygones fermés
        def calculer_surface(niveau_inondation):
            contour = plt.contour(grid_X, grid_Y, grid_Z, levels=[niveau_inondation], colors='none')
            paths = contour.collections[0].get_paths()
            surfaces = []
            for path in paths:
                polygon = Polygon(path.vertices)
                if polygon.is_valid and polygon.is_empty == False:  # Vérifier que le polygone est valide et non vide
                    surfaces.append(polygon)
            return MultiPolygon(surfaces)  # Retourner un MultiPolygon

        # Étape 8 : Calcul du volume d'eau
        def calculer_volume(surface_inondee, niveau_inondation):
            volume = surface_inondee.area * niveau_inondation * 10000  # Conversion en m³ (1 hectare = 10,000 m²)
            return volume

        if st.button("Calculer la surface et le volume"):
            surfaces_inondees = calculer_surface(niveau_inondation)
            surface_inondee = sum([poly.area for poly in surfaces_inondees.geoms]) / 10000  # Surface en hectares
            volume_eau = calculer_volume(surfaces_inondees, niveau_inondation)

            # Afficher les résultats
            st.write(f"Surface inondée : {surface_inondee:.2f} hectares")
            st.write(f"Volume d'eau : {volume_eau:.2f} m³")
            
            # Étape 4 : Fonction pour tracer la carte avec contours actuels et hachures
            def plot_map_with_hatching(niveau_inondation):
                plt.close('all')

                # Taille ajustée pour la carte
                fig, ax_map = plt.subplots(figsize=(8, 6))

                # Tracé de la carte de profondeur
                contour = ax_map.contourf(grid_X, grid_Y, grid_Z, cmap='viridis', levels=100)
                cbar = fig.colorbar(contour, ax=ax_map)
                cbar.set_label('Profondeur (mètres)')

                # Tracé du contour actuel du niveau d'inondation
                contours_inondation = ax_map.contour(grid_X, grid_Y, grid_Z, levels=[niveau_inondation], colors='red', linewidths=2)
                ax_map.clabel(contours_inondation, inline=True, fontsize=10, fmt='%1.1f m')

                # Tracé des hachures pour la zone inondée
                ax_map.contourf(grid_X, grid_Y, grid_Z, levels=[-np.inf, niveau_inondation], colors='none', hatches=['///'], alpha=0)

                # Tracer les polygones fermés
                for surface in surfaces_inondees.geoms:
                    ax_map.fill(surface.exterior.xy[0], surface.exterior.xy[1], alpha=0.3, fc='blue', ec='black')

                ax_map.set_title("Carte des zones inondées avec hachures et polygones")
                ax_map.set_xlabel("Coordonnée X")
                ax_map.set_ylabel("Coordonnée Y")

                # Affichage
                st.pyplot(fig)

            # Affichage de la carte d'inondation avec hachures
            plot_map_with_hatching(niveau_inondation)

else:
    st.warning("Veuillez téléverser un fichier pour démarrer.")

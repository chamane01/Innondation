# Importer les bibliothèques nécessaires
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import Polygon
import contextily as ctx

# Streamlit - Titre de l'application avec deux logos centrés côte à côte
col1, col2, col3 = st.columns([1, 2, 1])  # Créer des colonnes pour centrer les logos
with col1:
    st.image("POPOPO.jpg", width=150)
with col3:
    st.image("logo.png", width=150)  # Ajoutez le chemin de votre deuxième logo

st.markdown("<h1 style='text-align: center;'>Carte des zones inondées avec niveaux d'eau et surface</h1>", unsafe_allow_html=True)

# Initialiser session_state pour stocker les données d'inondation
if 'flood_data' not in st.session_state:
    st.session_state.flood_data = {
        'surface_inondee': None,
        'volume_eau': None,
        'niveau_inondation': 0.0
    }

# Étape 1 : Téléverser le fichier Excel ou TXT avec une barre de téléversement bleue élégante
uploaded_file = st.file_uploader(
    "<span style='color: #007BFF; font-weight: bold;'>Téléversez un fichier Excel ou TXT</span>", 
    type=["xlsx", "txt"], 
    label_visibility="visible", 
    key="fileUploader",
    help="Format accepté : .xlsx ou .txt",
    unsafe_allow_html=True
)

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
        st.session_state.flood_data['niveau_inondation'] = st.number_input(
            "<span style='color: #0056b3;'>Entrez le niveau d'eau (mètres)</span>", 
            min_value=0.0, 
            step=0.1, 
            format="%.1f", 
            key="niveauEau", 
            unsafe_allow_html=True
        )
        
        interpolation_method = st.selectbox(
            "<span style='color: #0056b3;'>Méthode d'interpolation</span>", 
            ['linear', 'nearest'], 
            key="interpolationSelect", 
            unsafe_allow_html=True
        )

        # Étape 6 : Création de la grille
        X_min, X_max = df['X'].min(), df['X'].max()
        Y_min, Y_max = df['Y'].min(), df['Y'].max()

        resolution = st.number_input(
            "<span style='color: #0056b3;'>Résolution de la grille</span>", 
            value=300, 
            min_value=100, 
            max_value=1000, 
            key="resolutionInput", 
            unsafe_allow_html=True
        )
        grid_X, grid_Y = np.mgrid[X_min:X_max:resolution*1j, Y_min:Y_max:resolution*1j]
        grid_Z = griddata((df['X'], df['Y']), df['Z'], (grid_X, grid_Y), method=interpolation_method)

        # Étape 7 : Calcul de la surface inondée
        def calculer_surface(niveau_inondation):
            contours = []
            for x in range(grid_X.shape[0]):
                for y in range(grid_Y.shape[1]):
                    if grid_Z[x, y] <= niveau_inondation:
                        contours.append((grid_X[x, y], grid_Y[x, y]))

            if contours:
                polygon = Polygon(contours)
                return polygon, polygon.area / 10000  # Retourne le polygone et la surface en hectares
            return None, 0.0

        # Étape 8 : Calcul du volume d'eau
        def calculer_volume(surface_inondee):
            volume = surface_inondee * st.session_state.flood_data['niveau_inondation'] * 10000  # Conversion en m³ (1 hectare = 10,000 m²)
            return volume

        if st.button("<span style='color: #007BFF;'>Afficher la carte d'inondation</span>", unsafe_allow_html=True):
            # Étape 9 : Calcul de la surface et volume
            polygon_inonde, surface_inondee = calculer_surface(st.session_state.flood_data['niveau_inondation'])
            volume_eau = calculer_volume(surface_inondee)

            # Stocker les résultats dans session_state
            st.session_state.flood_data['surface_inondee'] = surface_inondee
            st.session_state.flood_data['volume_eau'] = volume_eau

            # Tracer la carte de profondeur
            fig, ax = plt.subplots(figsize=(8, 6))

            # Tracer le fond OpenStreetMap
            ax.set_xlim(X_min, X_max)
            ax.set_ylim(Y_min, Y_max)
            ctx.add_basemap(ax, crs="EPSG:32630", source=ctx.providers.OpenStreetMap.Mapnik)

            # Tracer la carte de profondeur
            contourf = ax.contourf(grid_X, grid_Y, grid_Z, levels=100, cmap='viridis', alpha=0.5)
            plt.colorbar(contourf, label='Profondeur (mètres)')

            # Tracer le contour du niveau d'inondation
            contours_inondation = ax.contour(grid_X, grid_Y, grid_Z, levels=[st.session_state.flood_data['niveau_inondation']], colors='red', linewidths=1)
            ax.clabel(contours_inondation, inline=True, fontsize=10, fmt='%1.1f m')

            # Tracer la zone inondée
            if polygon_inonde:
                x_poly, y_poly = polygon_inonde.exterior.xy
                ax.fill(x_poly, y_poly, alpha=0.5, fc='cyan', ec='black', lw=1, label='Zone inondée')  # Couleur cyan pour la zone inondée

            ax.set_title("Carte des zones inondées")
            ax.set_xlabel("Coordonnée X")
            ax.set_ylabel("Coordonnée Y")
            ax.legend()

            # Affichage de la carte
            st.pyplot(fig)

            # Affichage des résultats à droite de la carte
            col1, col2 = st.columns([3, 1])  # Créer deux colonnes
            with col2:
                st.write(f"**Surface inondée :** {st.session_state.flood_data['surface_inondee']:.2f} hectares")
                st.write(f"**Volume d'eau :** {st.session_state.flood_data['volume_eau']:.2f} m³")

else:
    st.warning("Veuillez téléverser un fichier pour démarrer.")

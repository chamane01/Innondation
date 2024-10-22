# Importer les bibliothèques nécessaires
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from shapely.geometry import Polygon
import plotly.graph_objects as go  # Bibliothèque pour visualisation 3D
import folium  # Bibliothèque pour la carte interactive
from streamlit_folium import st_folium  # Permet d'afficher Folium dans Streamlit

# Streamlit - Logo et Titre de l'application
st.image("POPOPO.jpg", width=150)  # Remplacez par le chemin de votre image
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
        # Étape 4 : Visualisation des points dans un espace CAD (3D)
        def afficher_points_3D(df):
            fig = go.Figure(data=[go.Scatter3d(
                x=df['X'], y=df['Y'], z=df['Z'], 
                mode='markers', 
                marker=dict(size=5, color=df['Z'], colorscale='Viridis', opacity=0.8)
            )])
            
            fig.update_layout(
                title="Visualisation 3D des points",
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Z",
                ),
                width=700,
                margin=dict(r=20, b=10, l=10, t=50)
            )
            
            st.plotly_chart(fig)

        # Affichage de l'espace CAD (3D) avec les points
        afficher_points_3D(df)

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

        # Étape 9 : Afficher une carte interactive avec des options de styles
        st.subheader("Carte interactive")
        
        # Créer une carte centrée sur un point de coordonnées médianes
        m = folium.Map(location=[df['Y'].mean(), df['X'].mean()], zoom_start=12)

        # Ajouter des couches de base (Satellite, OpenStreetMap, etc.)
        folium.TileLayer('openstreetmap').add_to(m)
        folium.TileLayer('stamenterrain').add_to(m)
        folium.TileLayer('stamenwatercolor').add_to(m)
        folium.TileLayer('cartodbpositron').add_to(m)
        folium.TileLayer('cartodbdark_matter').add_to(m)
        folium.TileLayer('satellite').add_to(m)

        # Ajouter un contrôle de couche
        folium.LayerControl().add_to(m)

        # Afficher la carte dans Streamlit
        st_data = st_folium(m, width=700, height=500)

        # Étape 10 : Fonction pour tracer la carte avec contours actuels et hachures
        def plot_map_with_hatching(niveau_inondation, surface_inondee, volume_eau):
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

            ax_map.set_title("Carte des zones inondées avec hachures")
            ax_map.set_xlabel("Coordonnée X")
            ax_map.set_ylabel("Coordonnée Y")

            # Affichage
            st.pyplot(fig)

        # Étape 11 : Affichage initial de la carte avec hachures et rapport
        if st.button("Afficher la carte"):
            plot_map_with_hatching(niveau_inondation, surface_inondee, volume_eau)
            st.write(f"Surface inondée : {surface_inondee:.2f} hectares")
            st.write(f"Volume d'eau : {volume_eau:.2f} m³")

else:
    st.warning("Veuillez téléverser un fichier pour démarrer.")

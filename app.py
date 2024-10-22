import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

# Affichage du logo
st.image("POPOPO.jpg", use_column_width=True)

# Divider for better visual separation
st.markdown("---")

# Section for interactive map layers (satellite, OpenStreetMap, topographic)
st.subheader("Carte interactive")
m = folium.Map(location=[48.8566, 2.3522], zoom_start=5)  # Default location set to Paris, you can modify

# Adding layers
folium.TileLayer('Stamen Terrain').add_to(m)
folium.TileLayer('OpenStreetMap').add_to(m)
folium.TileLayer('Stamen Toner').add_to(m)
folium.TileLayer('Stamen Watercolor').add_to(m)
folium.LayerControl().add_to(m)

# Display the interactive map
st_folium(m, width=700, height=500)

# Divider to separate map and flood analysis section
st.markdown("---")

# Section for flood analysis map (persistent between refreshes)
st.subheader("Carte des zones inondées avec niveaux d'eau et surface")

uploaded_file = st.file_uploader("Téléversez un fichier Excel ou TXT", type=["xlsx", "txt"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.txt'):
        df = pd.read_csv(uploaded_file, sep=",", header=None, names=["X", "Y", "Z"])

    if 'X' not in df.columns or 'Y' not in df.columns or 'Z' not in df.columns:
        st.error("Erreur : colonnes 'X', 'Y' et 'Z' manquantes.")
    else:
        niveau_inondation = st.number_input("Entrez le niveau d'eau (mètres)", min_value=0.0, step=0.1)
        interpolation_method = st.selectbox("Méthode d'interpolation", ['linear', 'nearest'])

        X_min, X_max = df['X'].min(), df['X'].max()
        Y_min, Y_max = df['Y'].min(), df['Y'].max()

        resolution = st.number_input("Résolution de la grille", value=300, min_value=100, max_value=1000)
        grid_X, grid_Y = np.mgrid[X_min:X_max:resolution*1j, Y_min:Y_max:resolution*1j]
        grid_Z = griddata((df['X'], df['Y']), df['Z'], (grid_X, grid_Y), method=interpolation_method)

        def calculer_surface(niveau_inondation):
            contour = plt.contour(grid_X, grid_Y, grid_Z, levels=[niveau_inondation], colors='none')
            paths = contour.collections[0].get_paths()
            surfaces = [Polygon(path.vertices).area for path in paths]
            return sum(surfaces) / 10000  # Retourne en hectares

        def calculer_volume(niveau_inondation, surface_inondee):
            volume = surface_inondee * niveau_inondation * 10000  # Conversion en m³ (1 hectare = 10,000 m²)
            return volume

        surface_inondee = calculer_surface(niveau_inondation)
        volume_eau = calculer_volume(niveau_inondation, surface_inondee)

        def plot_map_with_hatching(niveau_inondation, surface_inondee, volume_eau):
            plt.close('all')

            fig, ax_map = plt.subplots(figsize=(8, 6))
            contour = ax_map.contourf(grid_X, grid_Y, grid_Z, cmap='viridis', levels=100)
            cbar = fig.colorbar(contour, ax=ax_map)
            cbar.set_label('Profondeur (mètres)')

            contours_inondation = ax_map.contour(grid_X, grid_Y, grid_Z, levels=[niveau_inondation], colors='red', linewidths=2)
            ax_map.clabel(contours_inondation, inline=True, fontsize=10, fmt='%1.1f m')

            ax_map.contourf(grid_X, grid_Y, grid_Z, levels=[-np.inf, niveau_inondation], colors='none', hatches=['///'], alpha=0)
            ax_map.set_title("Carte des zones inondées avec hachures")
            ax_map.set_xlabel("Coordonnée X")
            ax_map.set_ylabel("Coordonnée Y")

            st.pyplot(fig)

        if st.button("Afficher la carte"):
            plot_map_with_hatching(niveau_inondation, surface_inondee, volume_eau)
            st.write(f"Surface inondée : {surface_inondee:.2f} hectares")
            st.write(f"Volume d'eau : {volume_eau:.2f} m³")

else:
    st.warning("Veuillez téléverser un fichier pour démarrer.")

# Importer les bibliothèques nécessaires
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import Polygon, box
from shapely.geometry import MultiPolygon
import contextily as ctx
import ezdxf  # Bibliothèque pour créer des fichiers DXF
from datetime import datetime

# Streamlit - Titre de l'application avec deux logos centrés
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.image("POPOPO.jpg", width=150)
with col2:
    st.image("logo.png", width=150)
with col3:
    st.write("")  # Cette colonne est laissée vide pour centrer les logos

st.title("Carte des zones inondées avec niveaux d'eau et surface")

# Initialiser session_state pour stocker les données d'inondation
if 'flood_data' not in st.session_state:
    st.session_state.flood_data = {
        'surface_bleu': None,  
        'volume_eau': None,
        'niveau_inondation': 0.0
    }

# Étape 1 : Sélectionner un site ou téléverser un fichier
st.markdown("## Sélectionner un site ou téléverser un fichier")
option_site = st.selectbox("Sélectionnez un site", ("Aucun", "AYAME 1", "AYAME 2"))
uploaded_file = st.file_uploader("Téléversez un fichier Excel ou TXT", type=["xlsx", "txt"])

# Charger les données en fonction de l'option sélectionnée
def charger_fichier(fichier, is_uploaded=False):
    try:
        if is_uploaded:
            if fichier.name.endswith('.xlsx'):
                df = pd.read_excel(fichier)
            elif fichier.name.endswith('.txt'):
                df = pd.read_csv(fichier, sep=",", header=None, names=["X", "Y", "Z"])
        else:
            df = pd.read_csv(fichier, sep=",", header=None, names=["X", "Y", "Z"])
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {e}")
        return None

if option_site == "AYAME 1":
    df = charger_fichier('AYAME1.txt')
elif option_site == "AYAME 2":
    df = charger_fichier('AYAME2.txt')
elif uploaded_file is not None:
    df = charger_fichier(uploaded_file, is_uploaded=True)
else:
    st.warning("Veuillez sélectionner un site ou téléverser un fichier pour démarrer.")
    df = None

# Charger et filtrer les bâtiments dans l'emprise de la carte
try:
    batiments_gdf = gpd.read_file("batiments2.geojson")
    if df is not None:
        emprise = box(df['X'].min(), df['Y'].min(), df['X'].max(), df['Y'].max())
        batiments_gdf = batiments_gdf.to_crs(epsg=32630)
        batiments_dans_emprise = batiments_gdf[batiments_gdf.intersects(emprise)]
    else:
        batiments_dans_emprise = None
except Exception as e:
    st.error(f"Erreur lors du chargement des bâtiments : {e}")
    batiments_dans_emprise = None

# Traitement des données si le fichier est chargé
if df is not None:
    st.markdown("---")

    # Vérification du fichier : colonnes X, Y, Z
    if 'X' not in df.columns or 'Y' not in df.columns or 'Z' not in df.columns:
        st.error("Erreur : colonnes 'X', 'Y' et 'Z' manquantes.")
    else:
        st.session_state.flood_data['niveau_inondation'] = st.number_input("Entrez le niveau d'eau (mètres)", min_value=0.0, step=0.1)
        interpolation_method = st.selectbox("Méthode d'interpolation", ['linear', 'nearest'])

        X_min, X_max = df['X'].min(), df['X'].max()
        Y_min, Y_max = df['Y'].min(), df['Y'].max()
        resolution = st.number_input("Résolution de la grille", value=300, min_value=100, max_value=1000)
        grid_X, grid_Y = np.mgrid[X_min:X_max:resolution*1j, Y_min:Y_max:resolution*1j]
        grid_Z = griddata((df['X'], df['Y']), df['Z'], (grid_X, grid_Y), method=interpolation_method)

        def calculer_surface_bleue(niveau_inondation):
            return np.sum((grid_Z <= niveau_inondation)) * (grid_X[1, 0] - grid_X[0, 0]) * (grid_Y[0, 1] - grid_Y[0, 1]) / 10000

        def calculer_volume(surface_bleue):
            return surface_bleue * st.session_state.flood_data['niveau_inondation'] * 10000

        if st.button("Afficher la carte d'inondation"):
            surface_bleue = calculer_surface_bleue(st.session_state.flood_data['niveau_inondation'])
            volume_eau = calculer_volume(surface_bleue)
            st.session_state.flood_data['surface_bleu'] = surface_bleue
            st.session_state.flood_data['volume_eau'] = volume_eau

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlim(X_min, X_max)
            ax.set_ylim(Y_min, Y_max)
            ctx.add_basemap(ax, crs="EPSG:32630", source=ctx.providers.OpenStreetMap.Mapnik)

            # Tracer la zone inondée avec les contours
            contours_inondation = ax.contour(grid_X, grid_Y, grid_Z, levels=[st.session_state.flood_data['niveau_inondation']], colors='red', linewidths=1)
            ax.clabel(contours_inondation, inline=True, fontsize=10, fmt='%1.1f m')
            ax.contourf(grid_X, grid_Y, grid_Z, levels=[-np.inf, st.session_state.flood_data['niveau_inondation']], colors='#007FFF', alpha=0.5)

            # Ajouter des croisillons (lignes de grille)
            grid_spacing = 0.05  # Espacement des croisillons
            x_ticks = np.arange(X_min, X_max, grid_spacing)
            y_ticks = np.arange(Y_min, Y_max, grid_spacing)

            # Tracer les lignes de croisillons
            for x in x_ticks:
                ax.axvline(x, color='gray', linestyle='--', linewidth=0.5)
            for y in y_ticks:
                ax.axhline(y, color='gray', linestyle='--', linewidth=0.5)

            # Transformer les contours en polygones pour analyser les bâtiments
            contour_paths = [Polygon(path.vertices) for collection in contours_inondation.collections for path in collection.get_paths()]
            zone_inondee = gpd.GeoDataFrame(geometry=[MultiPolygon(contour_paths)], crs="EPSG:32630")

            # Filtrer et afficher tous les bâtiments
            if batiments_dans_emprise is not None:
                batiments_dans_emprise.plot(ax=ax, facecolor='grey', edgecolor='black', linewidth=0.5, alpha=0.6, label="Bâtiments non inondés")
                
                # Séparer les bâtiments inondés
                batiments_inondes = batiments_dans_emprise[batiments_dans_emprise.intersects(zone_inondee.unary_union)]
                batiments_inondes.plot(ax=ax, facecolor='red', edgecolor='black', linewidth=0.5, label="Bâtiments inondés")

            ax.legend()
            st.pyplot(fig)

            st.write(f"Surface inondée : {surface_bleue:.2f} ha")
            st.write(f"Volume d'eau estimé : {volume_eau:.2f} m³")

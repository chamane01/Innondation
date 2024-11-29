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
import ezdxf  # Pour créer des fichiers DXF
from datetime import datetime

# Streamlit - Titre avec deux logos centrés
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.image("POPOPO.jpg", width=150)
with col2:
    st.image("logo.png", width=150)
with col3:
    st.write("")  # Colonne vide pour centrer les logos

st.title("Carte des zones inondées avec niveaux d'eau et surfaces")

# Initialiser session_state pour stocker les données d'inondation
if 'flood_data' not in st.session_state:
    st.session_state.flood_data = {
        'surface_bleue': None,
        'volume_eau': None,
        'niveau_inondation': 0.0
    }

# Étape 1 : Sélection du site ou téléversement de fichier
st.markdown("## Sélectionner un site ou téléverser un fichier")
option_site = st.selectbox("Sélectionnez un site", ("Aucun", "AYAME 1", "AYAME 2"))
uploaded_file = st.file_uploader("Téléversez un fichier Excel ou TXT", type=["xlsx", "txt"])

# Fonction pour charger un fichier
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

# Chargement des données en fonction de l'option sélectionnée
if option_site == "AYAME 1":
    df = charger_fichier('AYAME1.txt')
elif option_site == "AYAME 2":
    df = charger_fichier('AYAME2.txt')
elif uploaded_file is not None:
    df = charger_fichier(uploaded_file, is_uploaded=True)
else:
    st.warning("Veuillez sélectionner un site ou téléverser un fichier pour continuer.")
    df = None

# Chargement des bâtiments dans l'emprise de la carte
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

# Traitement des données si un fichier est chargé
if df is not None:
    st.markdown("---")
    # Vérification des colonnes du fichier
    if 'X' not in df.columns or 'Y' not in df.columns or 'Z' not in df.columns:
        st.error("Erreur : colonnes 'X', 'Y' et 'Z' manquantes.")
    else:
        # Configuration de l'interpolation et des paramètres
        st.session_state.flood_data['niveau_inondation'] = st.number_input("Entrez le niveau d'eau (mètres)", min_value=0.0, step=0.1)
        interpolation_method = st.selectbox("Méthode d'interpolation", ['linear', 'nearest'])
        resolution = st.number_input("Résolution de la grille", value=300, min_value=100, max_value=1000)

        X_min, X_max = df['X'].min(), df['X'].max()
        Y_min, Y_max = df['Y'].min(), df['Y'].max()
        grid_X, grid_Y = np.mgrid[X_min:X_max:resolution*1j, Y_min:Y_max:resolution*1j]
        grid_Z = griddata((df['X'], df['Y']), df['Z'], (grid_X, grid_Y), method=interpolation_method)

        # Calcul de la surface inondée et du volume d'eau
        def calculer_surface_bleue(niveau_inondation):
            return np.sum((grid_Z <= niveau_inondation)) * (grid_X[1, 0] - grid_X[0, 0]) * (grid_Y[0, 1] - grid_Y[0, 0]) / 10000

        def calculer_volume(surface_bleue):
            return surface_bleue * st.session_state.flood_data['niveau_inondation'] * 10000

        # Affichage des résultats et création de la carte
        if st.button("Afficher la carte d'inondation"):
            surface_bleue = calculer_surface_bleue(st.session_state.flood_data['niveau_inondation'])
            volume_eau = calculer_volume(surface_bleue)
            st.session_state.flood_data['surface_bleue'] = surface_bleue
            st.session_state.flood_data['volume_eau'] = volume_eau

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlim(X_min, X_max)
            ax.set_ylim(Y_min, Y_max)
            ctx.add_basemap(ax, crs="EPSG:32630", source=ctx.providers.OpenStreetMap.Mapnik)

            contours_inondation = ax.contour(grid_X, grid_Y, grid_Z, levels=[st.session_state.flood_data['niveau_inondation']], colors='red', linewidths=1)
            ax.contourf(grid_X, grid_Y, grid_Z, levels=[-np.inf, st.session_state.flood_data['niveau_inondation']], colors='#007FFF', alpha=0.5)

            # Gestion des bâtiments et des fichiers DXF
            if batiments_dans_emprise is not None:
                batiments_dans_emprise.plot(ax=ax, facecolor='grey', edgecolor='black', alpha=0.6)
                batiments_inondes = batiments_dans_emprise[batiments_dans_emprise.intersects(MultiPolygon([Polygon(path.vertices) for path in contours_inondation.collections[0].get_paths()]).unary_union)]
                batiments_inondes.plot(ax=ax, color='red', alpha=0.8)

            st.pyplot(fig)
            doc = ezdxf.new(dxfversion='R2010')
            msp = doc.modelspace()
            for path in contours_inondation.collections[0].get_paths():
                points = path.vertices
                for i in range(len(points) - 1):
                    msp.add_line(points[i], points[i + 1])

            dxf_file = "contours_inondation.dxf"
            doc.saveas(dxf_file)

            # Résultats et téléchargements
            now = datetime.now()
            st.markdown("## Résultats")
            st.write(f"**Surface inondée :** {surface_bleue:.2f} hectares")
            st.write(f"**Volume d'eau :** {volume_eau:.2f} m³")
            st.write(f"**Date :** {now.strftime('%Y-%m-%d')}")

            with open(dxf_file, "rb") as dxf:
                st.download_button("Télécharger le fichier DXF", data=dxf, file_name=dxf_file)

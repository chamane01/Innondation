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
            return np.sum((grid_Z <= niveau_inondation)) * (grid_X[1, 0] - grid_X[0, 0]) * (grid_Y[0, 1] - grid_Y[0, 0]) / 10000

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
            ax.tick_params(axis='both', which='both', direction='in', length=6, width=1, color='black', labelsize=10)
            ax.set_xticks(np.linspace(X_min, X_max, num=5))
            ax.set_yticks(np.linspace(Y_min, Y_max, num=5))
            ax.xaxis.set_tick_params(labeltop=True)
            ax.yaxis.set_tick_params(labelright=True)

            # Ajouter les lignes pour relier les tirets
            for x in np.linspace(X_min, X_max, num=5):
                ax.axvline(x, color='black', linewidth=0.5, linestyle='--', alpha=0.2)
            for y in np.linspace(Y_min, Y_max, num=5):
                ax.axhline(y, color='black', linewidth=0.5, linestyle='--', alpha=0.2)

            # Ajouter les croisillons aux intersections
            intersections_x = np.linspace(X_min, X_max, num=5)
            intersections_y = np.linspace(Y_min, Y_max, num=5)
            for x in intersections_x:
                for y in intersections_y:
                    ax.plot(x, y, 'k+', markersize=7, alpha=1.0)

            # Tracer la zone inondée avec les contours
            contours_inondation = ax.contour(grid_X, grid_Y, grid_Z, levels=[st.session_state.flood_data['niveau_inondation']], colors='red', linewidths=1)

            # Vérifier si des contours ont été générés
            if contours_inondation.allsegs:
                ax.clabel(contours_inondation, inline=True, fontsize=10, fmt='%1.1f m')
                ax.contourf(grid_X, grid_Y, grid_Z, levels=[-np.inf, st.session_state.flood_data['niveau_inondation']], colors='#007FFF', alpha=0.5)

                # Transformer les contours en polygones pour analyser les bâtiments
                contour_paths = []
                for contour in contours_inondation.allsegs:
                    if len(contour) > 0:
                        for path in contour:
                            if len(path) > 0:
                                contour_paths.append(Polygon(path))

                if len(contour_paths) > 0:
                    zone_inondee = gpd.GeoDataFrame(geometry=[MultiPolygon(contour_paths)], crs="EPSG:32630")
                else:
                    st.warning("Aucun contour valide trouvé pour le niveau d'eau spécifié.")
                    zone_inondee = None
            else:
                st.warning("Aucun contour trouvé pour le niveau d'eau spécifié.")
                zone_inondee = None

            # Filtrer et afficher tous les bâtiments
            if batiments_dans_emprise is not None:
                batiments_dans_emprise.plot(ax=ax, facecolor='grey', edgecolor='black', linewidth=0.5, alpha=0.6, label="Bâtiments non inondés")

                # Séparer les bâtiments inondés
                if zone_inondee is not None:
                    batiments_inondes = batiments_dans_emprise[batiments_dans_emprise.intersects(zone_inondee.unary_union)]
                    nombre_batiments_inondes = len(batiments_inondes)

                    # Afficher les bâtiments inondés en rouge
                    batiments_inondes.plot(ax=ax, facecolor='red', edgecolor='red', linewidth=1, alpha=0.8, label="Bâtiments inondés")

                    st.write(f"Nombre de bâtiments dans la zone inondée : {nombre_batiments_inondes}")
                else:
                    st.write("Aucun bâtiment inondé trouvé.")
                ax.legend()
            else:
                st.write("Aucun bâtiment à analyser dans cette zone.")

            st.pyplot(fig)

            # Enregistrer les contours en fichier DXF
            doc = ezdxf.new(dxfversion='R2010')
            msp = doc.modelspace()
            if contours_inondation.allsegs:
                for contour in contours_inondation.allsegs:
                    for path in contour:
                        for i in range(len(path) - 1):
                            msp.add_line(path[i], path[i + 1])

            dxf_file = "contours_inondation.dxf"
            doc.saveas(dxf_file)
            carte_file = "carte_inondation.png"
            fig.savefig(carte_file)

            with open(carte_file, "rb") as carte:
                st.download_button(label="Télécharger la carte", data=carte, file_name=carte_file, mime="image/png")

            with open(dxf_file, "rb") as dxf:
                st.download_button(label="Télécharger le fichier DXF", data=dxf, file_name=dxf_file, mime="application/dxf")

            # Afficher les résultats
            now = datetime.now()
            st.markdown("## Résultats")
            st.write(f"**Surface inondée :** {surface_bleue:.2f} hectares")
            st.write(f"**Volume d'eau :** {volume_eau:.2f} m³")
            st.write(f"**Niveau d'eau :** {st.session_state.flood_data['niveau_inondation']} m")
            if zone_inondee is not None:
                st.write(f"**Nombre de bâtiments inondés :** {nombre_batiments_inondes}")
            st.write(f"**Date :** {now.strftime('%Y-%m-%d')}")
            st.write(f"**Heure :** {now.strftime('%H:%M:%S')}")
            st.write(f"**Système de projection :** EPSG:32630")

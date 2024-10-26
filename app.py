# Importer les bibliothèques nécessaires
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import Polygon
import contextily as ctx
import geopandas as gpd  # Importer geopandas pour manipuler les shapefiles

# Spécifiez le chemin du shapefile
shapefile_path = "chemin/vers/votre_fichier.shp"  # Remplacez ceci par le chemin de votre fichier shapefile

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
st.markdown("## Sélectionner un site")

# Ajouter une option pour sélectionner parmi des fichiers CSV existants (AYAME 1 et AYAME 2)
option_site = st.selectbox(
    "Sélectionnez un site",
    ("Aucun", "AYAME 1", "AYAME 2")
)

# Fonction pour charger le fichier CSV ou TXT
def charger_fichier(fichier):
    try:
        if fichier.name.endswith('.xlsx'):
            df = pd.read_excel(fichier)
        elif fichier.name.endswith('.txt'):
            df = pd.read_csv(fichier, sep=",", header=None, names=["X", "Y", "Z"])
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {e}")
        return None

# Charger les fichiers selon la sélection
if option_site == "AYAME 1":
    df = charger_fichier('AYAME1.txt')
elif option_site == "AYAME 2":
    df = charger_fichier('AYAME2.txt')
else:
    df = None

# Chargement du shapefile
gdf_batiments = gpd.read_file(shapefile_path)

# Traitement des données si le fichier est chargé
if df is not None:
    # Séparateur pour organiser l'affichage
    st.markdown("---")  # Ligne de séparation

    # Vérification du fichier : s'assurer que les colonnes X, Y, Z sont présentes
    if 'X' not in df.columns or 'Y' not in df.columns or 'Z' not in df.columns:
        st.error("Erreur : colonnes 'X', 'Y' et 'Z' manquantes.")
    else:
        # Étape 5 : Paramètres du niveau d'inondation
        st.session_state.flood_data['niveau_inondation'] = st.number_input("Entrez le niveau d'eau (mètres)", min_value=0.0, step=0.1)
        interpolation_method = st.selectbox("Méthode d'interpolation", ['linear', 'nearest'])

        # Étape 6 : Création de la grille
        X_min, X_max = df['X'].min(), df['X'].max()
        Y_min, Y_max = df['Y'].min(), df['Y'].max()

        resolution = st.number_input("Résolution de la grille", value=300, min_value=100, max_value=1000)
        grid_X, grid_Y = np.mgrid[X_min:X_max:resolution*1j, Y_min:Y_max:resolution*1j]
        grid_Z = griddata((df['X'], df['Y']), df['Z'], (grid_X, grid_Y), method=interpolation_method)

        # Étape 7 : Calcul de la surface occupée par la couleur bleue
        def calculer_surface_bleue(niveau_inondation):
            return np.sum((grid_Z <= niveau_inondation)) * (grid_X[1, 0] - grid_X[0, 0]) * (grid_Y[0, 1] - grid_Y[0, 0]) / 10000  # Conversion en hectares

        # Étape 8 : Calcul du volume d'eau
        def calculer_volume(surface_bleue):
            volume = surface_bleue * st.session_state.flood_data['niveau_inondation'] * 10000  # Conversion en m³ (1 hectare = 10,000 m²)
            return volume

        if st.button("Afficher la carte d'inondation"):
            # Étape 9 : Calcul de la surface bleue et volume
            surface_bleue = calculer_surface_bleue(st.session_state.flood_data['niveau_inondation'])
            volume_eau = calculer_volume(surface_bleue)

            # Stocker les résultats dans session_state
            st.session_state.flood_data['surface_bleu'] = surface_bleue  # Met à jour la surface occupée par la couleur bleue
            st.session_state.flood_data['volume_eau'] = volume_eau

            # Tracer la première carte avec OpenStreetMap et contours
            fig, ax = plt.subplots(figsize=(8, 6))

            # Tracer le fond OpenStreetMap
            ax.set_xlim(X_min, X_max)
            ax.set_ylim(Y_min, Y_max)
            ctx.add_basemap(ax, crs="EPSG:32630", source=ctx.providers.OpenStreetMap.Mapnik)

            # Tracer le contour du niveau d'inondation en rouge
            contours_inondation = ax.contour(grid_X, grid_Y, grid_Z, levels=[st.session_state.flood_data['niveau_inondation']], colors='red', linewidths=1)
            ax.clabel(contours_inondation, inline=True, fontsize=10, fmt='%1.1f m')

            # Tracé des hachures pour la zone inondée
            ax.contourf(grid_X, grid_Y, grid_Z, 
                        levels=[-np.inf, st.session_state.flood_data['niveau_inondation']], 
                        colors='#007FFF', alpha=0.5)  # Couleur bleue semi-transparente

            # Étape 10 : Analyser les bâtiments à partir du shapefile
            # Créer un GeoDataFrame à partir des points de données
            points_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['X'], df['Y']), crs="EPSG:32630")
            
            # Effectuer une intersection pour voir quels bâtiments sont inondés
            batiments_inondes = gdf_batiments[gdf_batiments.intersects(Polygon([(X_min, Y_min), (X_min, Y_max), (X_max, Y_max), (X_max, Y_min)]))]

            # Compter les bâtiments inondés et non inondés
            total_batiments = len(gdf_batiments)
            batiments_inondes_count = len(batiments_inondes)
            batiments_non_inondes_count = total_batiments - batiments_inondes_count

            # Afficher les bâtiments sur la carte
            for building in gdf_batiments.geometry:
                if building.intersects(Polygon([(X_min, Y_min), (X_min, Y_max), (X_max, Y_max), (X_max, Y_min)])):
                    # Bâtiment inondé
                    ax.add_patch(plt.Polygon(list(building.exterior.coords), color='red', alpha=0.5))
                else:
                    # Bâtiment non inondé
                    ax.add_patch(plt.Polygon(list(building.exterior.coords), color='white', alpha=0.5))

            st.write(f"Bâtiments inondés : {batiments_inondes_count}")
            st.write(f"Bâtiments non inondés : {batiments_non_inondes_count}")

            # Afficher le graphique
            st.pyplot(fig)

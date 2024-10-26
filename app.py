# Importer les bibliothèques nécessaires
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import Polygon
import shapely.ops as ops
import contextily as ctx
import ezdxf  # Bibliothèque pour créer des fichiers DXF

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
        'surface_inondee': None,
        'volume_eau': None,
        'niveau_inondation': 0.0,
        'polygons_inonde': []
    }

# Étape 1 : Sélectionner un site ou téléverser un fichier
st.markdown("## Sélectionner un site ou téléverser un fichier")

# Ajouter une option pour sélectionner parmi des fichiers CSV existants (AYAME 1 et AYAME 2)
option_site = st.selectbox(
    "Sélectionnez un site",
    ("Aucun", "AYAME 1", "AYAME 2")
)

# Téléverser un fichier Excel ou TXT
uploaded_file = st.file_uploader("Téléversez un fichier Excel ou TXT", type=["xlsx", "txt"])

# Fonction pour charger le fichier (identique pour les fichiers prédéfinis et téléversés)
def charger_fichier(fichier, is_uploaded=False):
    try:
        if is_uploaded:
            # Si le fichier est téléversé, vérifier son type
            if fichier.name.endswith('.xlsx'):
                df = pd.read_excel(fichier)
            elif fichier.name.endswith('.txt'):
                df = pd.read_csv(fichier, sep=",", header=None, names=["X", "Y", "Z"])
        else:
            # Si le fichier est prédéfini (site), il est déjà connu
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
elif uploaded_file is not None:
    df = charger_fichier(uploaded_file, is_uploaded=True)
else:
    st.warning("Veuillez sélectionner un site ou téléverser un fichier pour démarrer.")
    df = None

# Traitement des données si le fichier est chargé
if df is not None:
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

        # Étape 7 : Calcul des polygones de surface inondée
        def calculer_polygones_inonde(niveau_inondation):
            contours = plt.contour(grid_X, grid_Y, grid_Z, levels=[niveau_inondation], colors='red')
            polygons = []
            for path in contours.collections[0].get_paths():
                poly_coords = path.vertices
                polygon = Polygon(poly_coords)
                if polygon.is_valid:
                    polygons.append(polygon)
            return polygons

        # Calcul de la surface totale inondée
        def calculer_surface_totale(polygons):
            union_polygon = ops.unary_union(polygons)  # Créer un polygone unique
            surface_totale = union_polygon.area / 10000  # Surface en hectares
            return surface_totale

        # Afficher la carte d'inondation
        if st.button("Afficher la carte d'inondation"):
            polygons_inonde = calculer_polygones_inonde(st.session_state.flood_data['niveau_inondation'])
            surface_totale_inondee = calculer_surface_totale(polygons_inonde)
            
            # Stocker les polygones et surface totale dans session_state
            st.session_state.flood_data['polygons_inonde'] = polygons_inonde
            st.session_state.flood_data['surface_inondee'] = surface_totale_inondee

            # Tracer la première carte avec OpenStreetMap et contours
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlim(X_min, X_max)
            ax.set_ylim(Y_min, Y_max)
            ctx.add_basemap(ax, crs="EPSG:32630", source=ctx.providers.OpenStreetMap.Mapnik)

            # Tracer les contours de niveau d'inondation
            ax.contour(grid_X, grid_Y, grid_Z, levels=[st.session_state.flood_data['niveau_inondation']], colors='red', linewidths=1)

            # Tracer les polygones d'inondation sur la carte
            for polygon in polygons_inonde:
                x, y = polygon.exterior.xy
                ax.fill(x, y, color='blue', alpha=0.5)
            
            # Affichage de la première carte
            st.pyplot(fig)

            # Deuxième carte : visualisation des polygones sans basemap
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.set_xlim(X_min, X_max)
            ax2.set_ylim(Y_min, Y_max)
            ax2.set_aspect('equal')

            # Tracer les polygones d'inondation en bleu
            for polygon in polygons_inonde:
                x, y = polygon.exterior.xy
                ax2.fill(x, y, color='blue', alpha=0.5)

            # Affichage de la deuxième carte
            st.pyplot(fig2)

            # Affichage de la surface totale en dessous de la deuxième carte
            st.write(f"**Surface totale d'inondation calculée à partir des polygones :** {surface_totale_inondee:.2f} hectares")

            # Sauvegarde du fichier DXF avec les contours
            doc = ezdxf.new(dxfversion='R2010')
            msp = doc.modelspace()
            for polygon in polygons_inonde:
                x, y = polygon.exterior.xy
                points = list(zip(x, y))
                for i in range(len(points) - 1):
                    msp.add_line(points[i], points[i + 1])

            # Sauvegarder le fichier DXF
            dxf_file = "polygones_inondation.dxf"
            doc.saveas(dxf_file)

            # Proposer le téléchargement du fichier DXF
            with open(dxf_file, "rb") as dxf:
                st.download_button(label="Télécharger le fichier DXF", data=dxf, file_name=dxf_file, mime="application/dxf")
        else:
            st.warning("Cliquez sur 'Afficher la carte d'inondation' pour visualiser la carte.")

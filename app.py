# Importer les bibliothèques nécessaires
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import Polygon
import contextily as ctx
import ezdxf  # Pour lire les fichiers DXF
import io

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
        'niveau_inondation': 0.0
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
    df = charger_fichier('AYAME2.csv')
elif uploaded_file is not None:
    df = charger_fichier(uploaded_file, is_uploaded=True)
else:
    st.warning("Veuillez sélectionner un site ou téléverser un fichier pour démarrer.")
    df = None

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

        # Étape 7 : Calcul de la surface inondée
        def calculer_surface(niveau_inondation):
            contours = []
            for x in range(grid_X.shape[0]):
                for y in range(grid_Y.shape[1]):
                    if grid_Z[x, y] <= niveau_inondation:
                        contours.append((grid_X[x, y], grid_Y[x, y]))

            # Convertir les contours en un polygone
            if contours:
                polygon = Polygon(contours)
                return polygon, polygon.area / 10000  # Retourne le polygone et la surface en hectares
            return None, 0.0

        # Étape 8 : Calcul du volume d'eau
        def calculer_volume(surface_inondee):
            volume = surface_inondee * st.session_state.flood_data['niveau_inondation'] * 10000  # Conversion en m³ (1 hectare = 10,000 m²)
            return volume

        if st.button("Afficher la carte d'inondation"):
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

            # Tracer le contour du niveau d'inondation
            contours_inondation = ax.contour(grid_X, grid_Y, grid_Z, levels=[st.session_state.flood_data['niveau_inondation']], colors='red', linewidths=1)
            ax.clabel(contours_inondation, inline=True, fontsize=10, fmt='%1.1f m')
            # Tracé des hachures pour la zone inondée
            contourf_filled = ax.contourf(grid_X, grid_Y, grid_Z, 
                               levels=[-np.inf, st.session_state.flood_data['niveau_inondation']], 
                               colors='#007FFF', alpha=0.5)  # Couleur bleue semi-transparente

            # Affichage de la carte
            st.pyplot(fig)

            # Affichage des résultats à droite de la carte
            col1, col2 = st.columns([3, 1])  # Créer deux colonnes
            with col2:
                st.write(f"**Surface inondée :** {st.session_state.flood_data['surface_inondee']:.2f} hectares")
                st.write(f"**Volume d'eau :** {st.session_state.flood_data['volume_eau']:.2f} m³")

import io
import ezdxf

# Téléversement et visualisation du fichier DXF
st.markdown("---")
st.markdown("## Téléverser et visualiser un fichier DXF")

uploaded_dxf = st.file_uploader("Téléversez un fichier DXF", type=["dxf"])

if uploaded_dxf is not None:
    try:
        # Lecture du fichier DXF depuis le flux en mémoire
        dxf_data = io.BytesIO(uploaded_dxf.read())  # Lire le fichier téléversé en tant que bytes
        dxf = ezdxf.read(dxf_data)  # Lire le contenu DXF depuis la mémoire

        # Création d'une nouvelle carte pour visualiser les entités DXF
        fig_dxf, ax_dxf = plt.subplots(figsize=(8, 6))

        # Extraire et visualiser les polylignes
        for entity in dxf.entities:
            if entity.dxftype() == 'LWPOLYLINE' or entity.dxftype() == 'POLYLINE':
                points = np.array([point[:2] for point in entity.points()])
                ax_dxf.plot(points[:, 0], points[:, 1], color='green', linewidth=1)

        ax_dxf.set_title("Visualisation des polygones du fichier DXF")
        ax_dxf.set_xlabel("Coordonnée X")
        ax_dxf.set_ylabel("Coordonnée Y")

        # Affichage de la carte DXF
        st.pyplot(fig_dxf)

    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier DXF : {e}")


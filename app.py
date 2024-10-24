# Importer les bibliothèques nécessaires
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import Polygon, MultiPolygon
import contextily as ctx

# Streamlit - Titre de l'application avec deux logos centrés
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.image("POPOPO.jpg", width=150)
with col2:
    st.image("logo.png", width=150)
with col3:
    st.write("")  # Cette colonne est laissée vide pour centrer les logos

st.title("Carte des zones inondées avec niveaux d'eau et surface")

# Initialiser session_state pour stocker les données d'inondation et les contours
if 'flood_data' not in st.session_state:
    st.session_state.flood_data = {
        'surface_inondee': None,
        'volume_eau': None,
        'niveau_inondation': 0.0,
        'contours_inondee': None
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

# Fonction pour charger le fichier
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
    st.markdown("---")  # Ligne de séparation

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

        # Étape 1 : Calcul des contours de la zone inondée
        def calculer_contours(niveau_inondation):
            contours = []
            for x in range(grid_X.shape[0]):
                for y in range(grid_Y.shape[1]):
                    if grid_Z[x, y] <= niveau_inondation:
                        contours.append((grid_X[x, y], grid_Y[x, y]))
            if contours:
                polygon = Polygon(contours)
                return polygon
            return None

        # Étape 2 : Créer une deuxième carte dynamique avec le système de coordonnées locales
        if st.button("Créer la deuxième carte dynamique"):
            contours_inonde = calculer_contours(st.session_state.flood_data['niveau_inondation'])
            st.session_state.flood_data['contours_inondee'] = contours_inonde

            # Afficher la deuxième carte
            if contours_inonde:
                fig2, ax2 = plt.subplots(figsize=(8, 6))

                # Tracer les polygonales sans fond de carte
                x_poly, y_poly = contours_inonde.exterior.xy
                ax2.fill(x_poly, y_poly, alpha=0.5, fc='cyan', ec='black', lw=1, label='Zone inondée')

                ax2.set_title("Carte dynamique avec contours polygonales")
                ax2.set_xlabel("Coordonnée X")
                ax2.set_ylabel("Coordonnée Y")
                ax2.legend()

                st.pyplot(fig2)

        # Étape 3 : Carte principale d'inondation
        if st.button("Afficher la carte d'inondation"):
            contours_inonde = calculer_contours(st.session_state.flood_data['niveau_inondation'])
            if contours_inonde:
                st.session_state.flood_data['surface_inondee'] = contours_inonde.area / 10000

            fig, ax = plt.subplots(figsize=(8, 6))
            ctx.add_basemap(ax, crs="EPSG:32630", source=ctx.providers.OpenStreetMap.Mapnik)

            ax.set_xlim(X_min, X_max)
            ax.set_ylim(Y_min, Y_max)

            if contours_inonde:
                x_poly, y_poly = contours_inonde.exterior.xy
                ax.fill(x_poly, y_poly, alpha=0.5, fc='cyan', ec='black', lw=1)

            st.pyplot(fig)

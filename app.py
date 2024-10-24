# Importer les bibliothèques nécessaires
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from shapely.geometry import Polygon

# Streamlit - Titre de l'application avec deux logos centrés
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.image("POPOPO.jpg", width=150)
with col2:
    st.image("logo.png", width=150)
with col3:
    st.write("")  # Cette colonne est laissée vide pour centrer les logos

st.title("Carte interactive des zones inondées avec niveaux d'eau et surface")

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
    # Vérification du fichier : s'assurer que les colonnes X, Y, Z sont présentes
    if 'X' not in df.columns or 'Y' not in df.columns or 'Z' not in df.columns:
        st.error("Erreur : colonnes 'X', 'Y' et 'Z' manquantes.")
    else:
        # Étape 5 : Paramètres du niveau d'inondation
        st.session_state.flood_data['niveau_inondation'] = st.number_input("Entrez le niveau d'eau (mètres)", min_value=0.0, step=0.1)

        # Étape 6 : Création de la grille
        X_min, X_max = df['X'].min(), df['X'].max()
        Y_min, Y_max = df['Y'].min(), df['Y'].max()

        resolution = st.number_input("Résolution de la grille", value=300, min_value=100, max_value=1000)
        grid_X, grid_Y = np.mgrid[X_min:X_max:resolution*1j, Y_min:Y_max:resolution*1j]

        # Étape 7 : Calcul de la surface inondée
        def calculer_surface(niveau_inondation):
            contours = []
            for x in range(grid_X.shape[0]):
                for y in range(grid_Y.shape[1]):
                    if df['Z'][x] <= niveau_inondation:
                        contours.append((df['X'][x], df['Y'][y]))

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

            # Créer la carte interactive avec folium
            map_center = [(Y_min + Y_max) / 2, (X_min + X_max) / 2]
            m = folium.Map(location=map_center, zoom_start=13)

            # Ajouter les données d'inondation à la carte
            if polygon_inonde:
                folium.Polygon(
                    locations=[(x, y) for x, y in zip(df['X'], df['Y'])],
                    color='blue', 
                    fill=True,
                    fill_opacity=0.5,
                    popup=f'Surface inondée: {surface_inondee:.2f} hectares'
                ).add_to(m)

            # Afficher la carte interactive
            st_data = st_folium(m, width=700, height=500)

            # Afficher les résultats
            st.write(f"**Surface inondée :** {st.session_state.flood_data['surface_inondee']:.2f} hectares")
            st.write(f"**Volume d'eau :** {st.session_state.flood_data['volume_eau']:.2f} m³")

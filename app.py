# Importer les bibliothèques nécessaires
import streamlit as st
import pandas as pd
import numpy as np
import folium
from shapely.geometry import Polygon
from streamlit_folium import st_folium
from pyproj import Transformer

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

# Fonction pour charger le fichier
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

        # Étape 6 : Transformation des coordonnées si nécessaire
        transformer = Transformer.from_crs("EPSG:32630", "EPSG:4326", always_xy=True)  # De UTM (EPSG:32630) à WGS84 (EPSG:4326)
        df['lat'], df['lon'] = transformer.transform(df['X'].values, df['Y'].values)

        # Étape 7 : Création de la carte interactive avec folium
        m = folium.Map(location=[df['lat'].mean(), df['lon'].mean()], zoom_start=13)

        # Ajouter des points au fur et à mesure
        for i, row in df.iterrows():
            if row['Z'] <= st.session_state.flood_data['niveau_inondation']:
                folium.CircleMarker(
                    location=(row['lat'], row['lon']),
                    radius=5,
                    color='blue',
                    fill=True,
                    fill_opacity=0.6,
                ).add_to(m)

        # Tracer un polygone autour des zones inondées
        flooded_points = df[df['Z'] <= st.session_state.flood_data['niveau_inondation']][['lat', 'lon']].values
        if len(flooded_points) > 2:
            polygon_inonde = Polygon(flooded_points)
            folium.Polygon(locations=flooded_points, color='cyan', fill=True, fill_opacity=0.4).add_to(m)

        # Affichage de la carte dans Streamlit
        st_data = st_folium(m, width=700, height=500)

        # Étape 8 : Calcul de la surface inondée et du volume d'eau
        surface_inondee = polygon_inonde.area if len(flooded_points) > 2 else 0.0  # En degrés carrés
        volume_eau = surface_inondee * st.session_state.flood_data['niveau_inondation'] * 10000  # Conversion en m³

        # Stocker les résultats dans session_state
        st.session_state.flood_data['surface_inondee'] = surface_inondee
        st.session_state.flood_data['volume_eau'] = volume_eau

        # Affichage des résultats
        st.write(f"**Surface inondée :** {surface_inondee:.2f} hectares")
        st.write(f"**Volume d'eau :** {volume_eau:.2f} m³")

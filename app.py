# Importer les bibliothèques nécessaires
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import Polygon
import contextily as ctx
import ezdxf  # Bibliothèque pour créer et lire des fichiers DXF

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

# Téléverser un fichier Excel, TXT ou DXF
uploaded_file = st.file_uploader("Téléversez un fichier Excel, TXT ou DXF", type=["xlsx", "txt", "dxf"])

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
df = None
if option_site == "AYAME 1":
    df = charger_fichier('AYAME1.txt')
elif option_site == "AYAME 2":
    df = charger_fichier('AYAME2.txt')
elif uploaded_file is not None:
    if uploaded_file.name.endswith(".dxf"):
        # Traitement spécial pour les fichiers DXF
        try:
            doc = ezdxf.readfile(uploaded_file)
            msp = doc.modelspace()
            dxf_coords = []
            for entity in msp:
                if entity.dxftype() == 'LINE':
                    dxf_coords.append([entity.dxf.start, entity.dxf.end])
            st.success("Fichier DXF chargé avec succès")
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier DXF : {e}")
    else:
        df = charger_fichier(uploaded_file, is_uploaded=True)
else:
    st.warning("Veuillez sélectionner un site ou téléverser un fichier pour démarrer.")

# Traitement des données si le fichier est chargé
if df is not None or uploaded_file is not None:
    # Séparateur pour organiser l'affichage
    st.markdown("---")  # Ligne de séparation

    if df is not None:
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

            # Étape 7 : Calcul de la surface inondée et affichage de la carte d'inondation
            # Même logique de calcul et d'affichage qu'avant...
    
    if uploaded_file is not None and uploaded_file.name.endswith(".dxf"):
        # Visualiser les données du fichier DXF
        st.markdown("### Visualisation des lignes du fichier DXF")
        fig, ax = plt.subplots(figsize=(8, 6))

        for line in dxf_coords:
            start, end = line
            ax.plot([start[0], end[0]], [start[1], end[1]], color="blue", linewidth=1)

        ax.set_title("Visualisation des lignes DXF")
        st.pyplot(fig)

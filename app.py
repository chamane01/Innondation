# Importer les bibliothèques nécessaires
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import Polygon
import contextily as ctx
import ezdxf  # Bibliothèque pour traiter les fichiers DXF
from io import StringIO

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

# Étape 1 : Sélectionner un site, téléverser un fichier ou téléverser un fichier DXF
st.markdown("## Sélectionner un site, téléverser un fichier ou un fichier DXF")

# Ajouter une option pour sélectionner parmi des fichiers CSV existants (AYAME 1 et AYAME 2)
option_site = st.selectbox(
    "Sélectionnez un site",
    ("Aucun", "AYAME 1", "AYAME 2")
)

# Téléverser un fichier Excel ou TXT
uploaded_file = st.file_uploader("Téléversez un fichier Excel, TXT ou DXF", type=["xlsx", "txt", "dxf"])

# Fonction pour charger le fichier CSV ou Excel
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

# Fonction pour charger et afficher les données d'un fichier DXF
def charger_dxf(fichier_dxf):
    try:
        doc = ezdxf.readfile(fichier_dxf)
        msp = doc.modelspace()
        entities = []
        for entity in msp:
            if entity.dxftype() == 'LINE':
                entities.append([entity.start, entity.end])
        return entities
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier DXF : {e}")
        return None

# Charger les fichiers selon la sélection ou le téléversement
if option_site == "AYAME 1":
    df = charger_fichier('AYAME1.txt')
elif option_site == "AYAME 2":
    df = charger_fichier('AYAME2.txt')
elif uploaded_file is not None:
    # Vérifier si le fichier téléversé est un fichier DXF
    if uploaded_file.name.endswith(".dxf"):
        dxf_entities = charger_dxf(uploaded_file)
    else:
        df = charger_fichier(uploaded_file, is_uploaded=True)
else:
    st.warning("Veuillez sélectionner un site ou téléverser un fichier pour démarrer.")
    df = None

# Traitement des données si un fichier CSV ou Excel est chargé
if df is not None:
    # Suite du traitement et de l'affichage des données CSV/Excel

    # ... (même code que précédemment pour la visualisation des données)

# Affichage du fichier DXF si téléversé
if uploaded_file is not None and uploaded_file.name.endswith(".dxf"):
    if dxf_entities:
        st.markdown("---")
        st.markdown("### Visualisation du fichier DXF")

        # Créer une figure Matplotlib
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Tracer les lignes du fichier DXF
        for line in dxf_entities:
            start, end = line
            ax.plot([start[0], end[0]], [start[1], end[1]], color='blue', linewidth=2)

        # Affichage de la carte avec les entités DXF
        st.pyplot(fig)

        # Sauvegarder le fichier DXF (si modifié ou transformé)
        dxf_file = "televerse_dxf.dxf"
        with open(dxf_file, "wb") as dxf:
            dxf.write(uploaded_file.getbuffer())
        
        # Proposer le téléchargement du fichier DXF
        st.download_button(
            label="Télécharger le fichier DXF téléversé",
            data=uploaded_file,
            file_name=uploaded_file.name,
            mime="application/dxf"
        )

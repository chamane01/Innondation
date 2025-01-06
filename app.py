import streamlit as st
from components import render_sidebar, render_dashboard
from utils import process_images, simulate_defects
from map_utils import display_map

# Configuration de la page
st.set_page_config(
    page_title="Détection de Défauts Routiers",
    page_icon="🌍",
    layout="wide",
)

# Barre latérale
render_sidebar()

# Titre principal
st.title("Tableau de Bord - Détection Automatique des Défauts Routiers")

# Sections de l'application
tab1, tab2, tab3 = st.tabs(["📂 Chargement des Images", "🗺️ Carte Interactive", "📊 Statistiques"])

# Onglet 1 : Chargement des images
with tab1:
    st.header("📂 Chargement des Images")
    uploaded_images = st.file_uploader("Uploader des images de drones", accept_multiple_files=True, type=["jpg", "png", "jpeg"])
    if uploaded_images:
        st.success(f"{len(uploaded_images)} images chargées !")
        for img in uploaded_images:
            st.image(img, caption=img.name, use_column_width=True)
        st.write("Traitement des images...")
        processed_data = process_images(uploaded_images)

# Onglet 2 : Carte interactive
with tab2:
    st.header("🗺️ Carte Interactive des Défauts")
    st.markdown("### Carte des Routes avec Défauts Détectés")
    display_map()  # Fonction pour afficher la carte

# Onglet 3 : Statistiques
with tab3:
    st.header("📊 Statistiques des Défauts")
    st.markdown("### Visualisation des Données Statistiques")
    simulate_defects()  # Simule les données et affiche un tableau

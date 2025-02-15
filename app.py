import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from datetime import datetime

# Configuration de la page
st.set_page_config(page_title="Rapport Topographique", layout="wide")

# Style CSS personnalisé
st.markdown("""
    <style>
    .header {
        font-size:24px;
        font-weight:bold;
        color:#2e5266;
        padding-bottom:20px;
    }
    .preview-mode {
        color: red;
        border: 2px solid red;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Fonctions pour générer des données fictives
def generate_sample_map():
    m = folium.Map(location=[48.8566, 2.3522], zoom_start=13)
    folium.Marker([48.8566, 2.3522], tooltip='Point de référence').add_to(m)
    return m

def generate_sample_profile():
    return pd.DataFrame({
        'Distance (m)': np.arange(0, 100, 1),
        'Altitude (m)': np.sin(np.arange(0, 100, 1)) * 10 + 50
    })

# Initialisation des variables de session
if 'preview_mode' not in st.session_state:
    st.session_state.preview_mode = False

# Sidebar pour la configuration
with st.sidebar:
    st.header("Paramètres du rapport")
    report_date = st.date_input("Date du rapport", datetime.now())
    map_center = st.text_input("Centre de la carte (lat,lon)", "48.8566, 2.3522")
    st.session_state.preview_mode = st.checkbox("Mode Prévisualisation PDF")

# Entête du rapport
col1, col2 = st.columns([1, 4])
with col1:
    logo = st.file_uploader("Logo de l'entreprise", type=['png', 'jpg'], disabled=st.session_state.preview_mode)
    if logo:
        st.image(logo, width=150)

with col2:
    st.title("Rapport Topographique" if not st.session_state.preview_mode else "Rapport Topographique - Version PDF")
    st.subheader("Analyse et relevés techniques")

# Mode prévisualisation
if st.session_state.preview_mode:
    st.markdown('<div class="preview-mode">MODE PRÉVISUALISATION PDF</div>', unsafe_allow_html=True)

# Section des informations de base
if not st.session_state.preview_mode:
    with st.form("basic_info"):
        cols = st.columns(4)
        report_number = cols[0].text_input("Numéro du rapport", "TR-2023-001")
        location = cols[1].text_input("Localisation", "Paris, France")
        project_name = cols[2].text_input("Nom du projet", "Projet X")
        client_name = cols[3].text_input("Client", "Société Y")
        st.form_submit_button("Enregistrer les informations")
else:
    st.write(f"**Numéro du rapport:** {report_number}")
    st.write(f"**Localisation:** {location}")
    st.write(f"**Nom du projet:** {project_name}")
    st.write(f"**Client:** {client_name}")

# Section carte géographique
st.markdown('<div class="header">Carte topographique</div>', unsafe_allow_html=True)
with st.container():
    map_col, legend_col = st.columns([3, 1])
    
    with map_col:
        sample_map = generate_sample_map()
        st_folium(sample_map, width=800, height=500)
    
    with legend_col:
        st.subheader("Légende")
        legend_data = {
            "Symbole": ["📍", "🟥", "🟦"],
            "Description": ["Point de référence", "Zone critique", "Cours d'eau"]
        }
        st.table(pd.DataFrame(legend_data))

# Section des profils topographiques
st.markdown('<div class="header">Profils topographiques</div>', unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["Profil longitudinal", "Profil transversal", "Analyse 3D"])

with tab1:
    df = generate_sample_profile()
    st.line_chart(df, x='Distance (m)', y='Altitude (m)')

with tab2:
    df = generate_sample_profile()
    st.area_chart(df, x='Distance (m)', y='Altitude (m)')

with tab3:
    st.write("Visualisation 3D (simulation)")
    # Correction des valeurs pour la visualisation
    arr = np.random.rand(100, 100)  # Génère des valeurs entre 0 et 1
    st.image(arr, use_column_width=True, clamp=True, caption="Modèle numérique de terrain")

# Section des informations techniques
with st.expander("Données techniques détaillées", expanded=st.session_state.preview_mode):
    tech_col1, tech_col2 = st.columns(2)
    
    with tech_col1:
        st.subheader("Spécifications")
        st.metric("Précision horizontale", "2.5 cm")
        st.metric("Précision verticale", "1.8 cm")
        st.metric("Système de coordonnées", "WGS 84")
    
    with tech_col2:
        st.subheader("Équipement utilisé")
        st.write("- Station totale Leica TS16")
        st.write("- Récepteur GNSS Trimble R12")
        st.write("- Logiciel de traitement TBC")

# Pied de page
st.markdown("---")
st.caption(f"Rapport généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Bouton de génération de PDF
if st.button("Générer le PDF", disabled=not st.session_state.preview_mode):
    # Implémenter la logique PDF ici
    st.success("Génération PDF réussie!")
    st.balloons()

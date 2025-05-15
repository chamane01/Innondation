import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import jinja2
from pyproj import Transformer
from html2image import Html2Image
import zipfile
import os
from PIL import Image
import tempfile
import base64
from datetime import datetime

# Configuration initiale
sti = Html2Image(output_path='fiches', size=(800, 1131))  # Format A4 vertical
ENCODING = 'utf-8-sig'

# Template HTML avec Jinja2
template = jinja2.Template("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 1cm; }
        .header { text-align: center; margin-bottom: 20px; }
        .title { font-size: 24px; font-weight: bold; margin: 10px 0; }
        .subtitle { font-size: 18px; margin: 5px 0; }
        .section { margin: 15px 0; border: 2px solid #000; padding: 10px; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #000; padding: 8px; text-align: left; }
        .page-break { page-break-after: always; }
        .photo-section { margin-top: 15px; }
        .photo { max-width: 100%; height: auto; margin: 5px 0; }
    </style>
</head>
<body>
    <div class="header">
        <div class="title">République de Côte d'Ivoire</div>
        <div class="subtitle">Union - Discipline - Travail</div>
        <div class="subtitle">MINISTÈRE DE L’ÉQUIPEMENT ET DE L’ENTRETIEN ROUTIER</div>
        <div class="subtitle">Projet d'Intérêt pour le Développement Urbain et la Conversionité des Agglomérations Secondaires (PIDUCAS)</div>
    </div>

    <div class="section">
        <h2>CADASTRAGE DE LA VILLE DE {{ commune|upper }}</h2>
        <h3>RÉSEAU GÉODÉSIQUE IVOIRIEN DE DÉTAIL</h3>
        <h3>BORNE GÉODÉSIQUE : {{ point_id }}</h3>
        
        <table>
            <tr><th colspan="4">FICHE SIGNALÉTIQUE</th></tr>
            <tr>
                <th>DESIGNATION</th>
                <th>COORDONNÉES GÉOGRAPHIQUES GÉODÉSIQUES</th>
                <th>COORDONNÉES RECTANGULAIRES</th>
                <th>ALTITUDE</th>
            </tr>
            <tr>
                <td>{{ designation }}</td>
                <td>{{ systeme_geodesique }}</td>
                <td>{{ systeme_rectangulaire }}</td>
                <td>{{ altitude }} m</td>
            </tr>
        </table>

        <table>
            <tr>
                <th>ID</th>
                <th>LATITUDE</th>
                <th>LONGITUDE</th>
                <th>X (UTM)</th>
                <th>Y (UTM)</th>
                <th>Z</th>
            </tr>
            <tr>
                <td>{{ point_id }}</td>
                <td>{{ latitude }}</td>
                <td>{{ longitude }}</td>
                <td>{{ x }}</td>
                <td>{{ y }}</td>
                <td>{{ z }}</td>
            </tr>
        </table>
    </div>

    {% if photos %}
    <div class="photo-section">
        <h3>VUES</h3>
        {% for photo in photos %}
        <img class="photo" src="data:image/png;base64,{{ photo }}">
        {% endfor %}
    </div>
    {% endif %}

    <div class="section">
        <h3>DESCRIPTION SUCCINCTE DE LA SITUATION GÉOGRAPHIQUE</h3>
        <p>{{ description }}</p>
        <table>
            <tr>
                <th>Région</th>
                <th>Département</th>
                <th>Commune</th>
            </tr>
            <tr>
                <td>{{ region }}</td>
                <td>{{ departement }}</td>
                <td>{{ commune }}</td>
            </tr>
        </table>
    </div>
</body>
</html>
""")

def convert_utm_to_wgs84(x, y, zone=30):
    transformer = Transformer.from_crs(f"EPSG:326{zone}", "EPSG:4326")
    lat, lon = transformer.transform(x, y)
    return lat, lon

def create_fiche(data, photos, template_vars):
    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = os.path.join(tmpdir, "fiche.html")
        img_path = os.path.join(tmpdir, "fiche.png")
        
        with open(html_path, "w", encoding=ENCODING) as f:
            f.write(template.render(
                point_id=data['ID'],
                x=data['X'],
                y=data['Y'],
                z=data.get('Z', ''),
                latitude=data['Latitude'],
                longitude=data['Longitude'],
                altitude=data.get('Altitude', ''),
                photos=photos,
                **template_vars
            ))
        
        sti.screenshot(html_file=html_path, save_as='fiche.png')
        return img_path

def main():
    st.set_page_config(page_title="Générateur de Fiches Signalétiques", layout="wide")
    st.title("📄 Générateur Automatique de Fiches Signalétiques PIDUCAS")

    # Section des informations génériques
    with st.sidebar:
        st.header("Informations Génériques")
        template_vars = {
            'commune': st.text_input("Commune", "SAN-PEDRO"),
            'region': st.text_input("Région", "SAN-PEDRO"),
            'departement': st.text_input("Département", "SAN-PEDRO"),
            'designation': st.text_input("Désignation", "BLUMBOIS DUVOGLIA - ARPENT PIPPA - VISA"),
            'systeme_geodesique': st.text_input("Système Géodésique", "BLUMBOIS DUVOGLIA"),
            'systeme_rectangulaire': st.text_input("Système Rectangulaire", "UPL - TURBULAR"),
            'description': st.text_area("Description", "Borne géodésique située à l'entrée de la mairie...")
        }

    # Section d'import des données
    st.header("1. Import des Données")
    input_method = st.radio("Source des données", ["Fichier CSV/GeoJSON", "Saisie Manuelle"])
    
    df = pd.DataFrame()
    if input_method == "Fichier CSV/GeoJSON":
        uploaded_file = st.file_uploader("Téléverser le fichier", type=['csv', 'geojson', 'txt'])
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.geojson'):
                gdf = gpd.read_file(uploaded_file)
                df = pd.DataFrame(gdf)
    
    else:
        manual_data = {
            'ID': st.text_input("ID de la Borne (ex: SP01)"),
            'X': st.number_input("Coordonnée X (UTM)"),
            'Y': st.number_input("Coordonnée Y (UTM)"),
            'Z': st.number_input("Altitude (m)", value=34.16)
        }
        df = pd.DataFrame([manual_data])

    # Conversion des coordonnées
    if not df.empty:
        if 'Latitude' not in df.columns:
            zone = st.number_input("Zone UTM", min_value=1, max_value=60, value=30)
            df['Latitude'], df['Longitude'] = zip(*df.apply(
                lambda row: convert_utm_to_wgs84(row['X'], row['Y'], zone), axis=1))
            df['Latitude'] = df['Latitude'].round(6)
            df['Longitude'] = df['Longitude'].round(6)

    # Gestion des photos
    st.header("2. Téléversement des Photos")
    uploaded_photos = st.file_uploader("Ajouter des photos pour la borne", 
                                     type=['jpg', 'png', 'jpeg'], 
                                     accept_multiple_files=True)

    # Génération des fiches
    if st.button("Générer les Fiches Signalétiques"):
        if df.empty:
            st.error("Aucune donnée à traiter!")
            return

        zip_filename = f"fiches_signaletiques_{datetime.now().strftime('%Y%m%d%H%M')}.zip"
        photos_base64 = [base64.b64encode(photo.getvalue()).decode() for photo in uploaded_photos]

        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for _, row in df.iterrows():
                img_path = create_fiche(row, photos_base64, template_vars)
                zipf.write(img_path, os.path.basename(img_path))

        with open(zip_filename, "rb") as f:
            st.download_button("Télécharger toutes les fiches", f, file_name=zip_filename)

if __name__ == "__main__":
    main()

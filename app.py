# app.py
import streamlit as st
import pandas as pd
import json
from fpdf import FPDF
import io
from PIL import Image

st.set_page_config(page_title="Générateur de Fiches Signalétiques", layout="wide")

st.title("📘 Générateur de Catalogue de Fiches Signalétiques")

# --- 1. Upload du fichier de points ---
st.sidebar.header("1. Chargement des données")
uploaded_file = st.sidebar.file_uploader(
    "Importez votre fichier de points (CSV, TXT ou GeoJSON)",
    type=["csv", "txt", "json"],
)

df = None
if uploaded_file:
    if uploaded_file.name.endswith(".json"):
        geo = json.load(uploaded_file)
        # Extraction simple : suppose FeatureCollection de GeoJSON
        features = geo.get("features", [])
        records = []
        for f in features:
            props = f.get("properties", {})
            geom = f.get("geometry", {})
            if geom.get("type") == "Point":
                lon, lat = geom["coordinates"]
                props.update({"X": lon, "Y": lat})
            records.append(props)
        df = pd.DataFrame(records)
    else:
        df = pd.read_csv(uploaded_file, sep=None, engine="python")
    st.success(f"Données chargées : {len(df)} points")

# --- 2. Saisie des informations génériques ---
st.sidebar.header("2. Informations Générales")
republique = st.sidebar.text_input("République / État", "République de Côte d'Ivoire")
ministere = st.sidebar.text_input("Ministère / Projet", "Ministère de l’Équipement et de l’Entretien Routier")
commune = st.sidebar.text_input("Commune / Agglomération", "")
logo = st.sidebar.file_uploader("Logo (PNG, JPG)", type=["png", "jpg", "jpeg"])

# Fonction utilitaire pour ajouter le header commun
def add_header(pdf: FPDF):
    if logo:
        img = Image.open(logo)
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        pdf.image(bio, x=10, y=8, w=30)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, republique, ln=1, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, ministere, ln=1, align="C")
    pdf.ln(4)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

# --- 3. Photos par point (optionnel) ---
st.sidebar.header("3. Photos par point")
st.sidebar.write("Si vous souhaitez joindre des photos pour chaque borne, nommez-les de la forme `IDpoint_1.jpg`, `IDpoint_2.jpg`…")
photos = st.sidebar.file_uploader("Importer plusieurs photos", type=["png","jpg","jpeg"], accept_multiple_files=True)

# Indexer les photos par identifiant de point
photo_dict = {}
for f in photos:
    name = f.name
    key = name.split("_")[0]
    photo_dict.setdefault(key, []).append(f)

# --- 4. Génération du PDF ---
if st.sidebar.button("🖨️ Générer le PDF") and df is not None:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)

    for idx, row in df.iterrows():
        pid = str(row.get("ID", row.get("id", idx)))
        pdf.add_page()
        add_header(pdf)

        # Titre de la fiche
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, f"FICHE SIGNALETIQUE – Borne {pid}", ln=1)

        # Tableau des coordonnées
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(50, 6, "Coordonnées Géographiques :", ln=0)
        pdf.cell(0, 6, f"Lat: {row.get('latitude', row.get('Y',''))}  Lon: {row.get('longitude', row.get('X',''))}", ln=1)
        pdf.cell(50, 6, "Coordonnées Rectangulaires (UTM) :", ln=0)
        pdf.cell(0, 6, f"X: {row.get('X','')}  Y: {row.get('Y','')}", ln=1)
        pdf.cell(50, 6, "Altitude (m) :", ln=0)
        pdf.cell(0, 6, f"{row.get('Z', '')}", ln=1)
        pdf.ln(4)

        # Photos
        if pid in photo_dict:
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 6, "Vues :", ln=1)
            x0, y0 = pdf.get_x(), pdf.get_y()
            max_w, max_h = 60, 45
            for i, f in enumerate(photo_dict[pid]):
                img = Image.open(f)
                bio = io.BytesIO()
                img.thumbnail((max_w*4, max_h*4))
                img.save(bio, format="JPEG")
                x = x0 + (i % 2) * (max_w + 5)
                y = y0 + (i // 2) * (max_h + 5)
                pdf.image(bio, x=x, y=y, w=max_w, h=max_h)
            pdf.ln(max_h*2/3 + 8)

        # Pied de page
        pdf.set_y(-20)
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(0, 5, f"Commune : {commune}", ln=1, align="L")
        pdf.cell(0, 5, f"Fiche générée automatiquement", ln=1, align="R")

    # Préparation du buffer PDF
    pdf_buffer = pdf.output(dest="S").encode("latin-1")
    st.success("✅ Votre catalogue PDF est prêt !")
    st.download_button(
        label="⬇️ Télécharger le PDF",
        data=pdf_buffer,
        file_name="catalogue_fiches_signaletiques.pdf",
        mime="application/pdf"
    )

# app.py
import streamlit as st
import pandas as pd
from fpdf import FPDF
import io
from PIL import Image

st.set_page_config(page_title="Générateur de Fiches Signalétiques", layout="wide")
st.title("📘 Générateur de Catalogue de Fiches Signalétiques")

# 1) Chargement des données
st.sidebar.header("1. Chargement des points")
uploaded = st.sidebar.file_uploader("Votre CSV / TXT / GeoJSON", type=["csv","txt","json"])
df = None
if uploaded:
    if uploaded.name.lower().endswith(".json"):
        geo = pd.json_normalize(pd.read_json(uploaded)["features"])
        # suppose que geometry.coordinates = [lon,lat]
        df = pd.DataFrame({
            "ID": geo["properties.ID"].fillna(geo.index).astype(str),
            "X": geo["geometry.coordinates"].apply(lambda c: c[0]),
            "Y": geo["geometry.coordinates"].apply(lambda c: c[1]),
            "Z": geo["properties.Z"] if "properties.Z" in geo else ""
        })
    else:
        df = pd.read_csv(uploaded, sep=None, engine="python")
        if "ID" not in df.columns:
            df["ID"] = df.index.astype(str)
    st.success(f"{len(df)} points chargés.")

# 2) Infos génériques
st.sidebar.header("2. Informations générales")
republique = st.sidebar.text_input("Republique / Etat", "Republique de Cote d'Ivoire")
ministere = st.sidebar.text_input("Ministere / Projet", "Ministere de l'Equipement et de l'Entretien Routier")
commune = st.sidebar.text_input("Commune / Agglomeration", "")
logo = st.sidebar.file_uploader("Logo (PNG, JPG)", type=["png","jpg","jpeg"])

# 3) Photos par point
photo_dict = {}
if df is not None:
    st.header("📸 Photos par point")
    for _, row in df.iterrows():
        pid = str(row["ID"])
        with st.expander(f"Borne {pid}", expanded=False):
            files = st.file_uploader(
                f"Photos pour {pid}",
                type=["jpg","jpeg","png"],
                accept_multiple_files=True,
                key=f"upl_{pid}"
            )
            if files:
                photo_dict[pid] = files

# 4) Génération du PDF
if st.sidebar.button("🖨️ Générer le PDF") and df is not None:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)

    for _, row in df.iterrows():
        pid = str(row["ID"])
        pdf.add_page()

        # Header
        if logo:
            img = Image.open(logo)
            bio = io.BytesIO()
            img.thumbnail((50, 50))
            img.save(bio, format="PNG")
            pdf.image(bio, x=10, y=8, w=25)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, republique, ln=1, align="C")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, ministere, ln=1, align="C")
        pdf.ln(4)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(6)

        # Titre
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, f"FICHE SIGNALETIQUE – Borne {pid}", ln=1)

        # Coordonnees
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(60, 6, "Coordonnees geographiques :", ln=0)
        pdf.cell(0, 6, f"X: {row.get('X','')}  Y: {row.get('Y','')}", ln=1)
        pdf.cell(60, 6, "Altitude (m) :", ln=0)
        pdf.cell(0, 6, f"{row.get('Z','')}", ln=1)
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
                x = x0 + (i % 2)*(max_w+5)
                y = y0 + (i//2)*(max_h+5)
                pdf.image(bio, x=x, y=y, w=max_w, h=max_h)
            pdf.ln(max_h*2/3 + 8)

        # Pied de page
        pdf.set_y(-20)
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(0, 5, f"Commune : {commune}", ln=1, align="L")
        pdf.cell(0, 5, "Fiche generee automatiquement", ln=1, align="R")

    # On force l'encodage Latin-1 en ignorant les caractères non supportés
    raw = pdf.output(dest="S")
    pdf_bytes = raw.encode("latin-1", errors="ignore")

    st.success("✅ PDF prêt")
    st.download_button(
        "⬇️ Télécharger le PDF",
        data=pdf_bytes,
        file_name="catalogue_fiches_signaletiques.pdf",
        mime="application/pdf"
    )

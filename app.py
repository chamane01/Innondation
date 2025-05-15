# app.py
import streamlit as st
import pandas as pd
import json
from fpdf import FPDF
from PIL import Image
import io
from datetime import datetime

st.set_page_config(page_title="Générateur de Fiches Signalétiques", layout="wide")
st.title("📘 Générateur de Catalogue de Fiches Signalétiques")

# ----------------------------
# 1) Chargement des données
# ----------------------------
st.sidebar.header("1. Chargement des points")
uploaded = st.sidebar.file_uploader(
    "Importez CSV / TXT / GeoJSON", type=["csv", "txt", "json"]
)
df = None
if uploaded:
    try:
        if uploaded.name.lower().endswith(".json"):
            geo = json.load(uploaded)
            feats = geo.get("features", [])
            records = []
            for f in feats:
                props = f.get("properties", {})
                geom = f.get("geometry", {})
                coords = geom.get("coordinates", [None, None])
                props.update({
                    "ID": str(props.get("ID", len(records))),
                    "X": coords[0],
                    "Y": coords[1],
                    "Z": props.get("Z", "")
                })
                records.append(props)
            df = pd.DataFrame(records)
        else:
            df = pd.read_csv(uploaded, sep=None, engine="python")
            if "ID" not in df.columns:
                df.insert(0, "ID", df.index.astype(str))
    except Exception as e:
        st.error(f"Erreur de lecture du fichier : {e}")
    else:
        st.success(f"{len(df)} points chargés")

# ----------------------------
# 2) Informations générales
# ----------------------------
st.sidebar.header("2. Infos générales")
republique = st.sidebar.text_input(
    "République / État", "République de Côte d'Ivoire"
)
ministere = st.sidebar.text_input(
    "Ministère / Projet", "Ministère de l’Équipement et de l’Entretien Routier"
)
projet = st.sidebar.text_input(
    "Projet", "PIDUCAS – Cadastrage de la ville de San-Pedro"
)
commune = st.sidebar.text_input("Commune / Agglomération", "")
logo_file = st.sidebar.file_uploader(
    "Logo (PNG/JPG)", type=["png", "jpg", "jpeg"]
)

# ----------------------------
# 3) Photos par point
# ----------------------------
photo_dict = {}
if df is not None:
    st.header("📸 Photos par point")
    for _, row in df.iterrows():
        pid = str(row["ID"])
        with st.expander(f"Borne {pid}", expanded=False):
            files = st.file_uploader(
                label=f"Photos pour {pid}",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key=f"upl_{pid}"
            )
            if files:
                photo_dict[pid] = files

# ----------------------------
# 4) Génération PDF
# ----------------------------
if st.sidebar.button("🖨️ Générer le PDF") and df is not None:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)

    for idx, row in df.iterrows():
        pid = str(row.get("ID", idx))
        pdf.add_page()

        # ---- Header officiel ----
        if logo_file:
            try:
                img = Image.open(logo_file)
                bio = io.BytesIO()
                img.thumbnail((60, 60))
                img.save(bio, format="PNG")
                bio.seek(0)
                pdf.image(bio, x=12, y=8, w=25, type="PNG")
            except Exception:
                pass

        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, republique, ln=1, align="C")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"{ministere}", ln=1, align="C")
        pdf.cell(0, 6, f"{projet}", ln=1, align="C")
        pdf.ln(2)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(6)

        # ---- Titre de la fiche ----
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, f"BORNE GÉODÉSIQUE : SP {pid}", ln=1, align="L")
        pdf.set_font("Helvetica", "", 10)
        date_str = datetime.today().strftime("%B %Y")
        pdf.cell(0, 6, f"MAJ : {date_str}", ln=1, align="L")
        pdf.ln(4)

        # ---- Tableau coordonnées ----
        # Entêtes
        pdf.set_font("Helvetica", "B", 10)
        col_w = [35, 35, 35, 40, 40]
        headers = [
            "LATITUDE NORD", "LONGITUDE OUEST", "HAUTEUR / ELLIPSOÏDE",
            "X (m)", "Y (m)"
        ]
        for w, h in zip(col_w, headers):
            pdf.cell(w, 7, h, border=1, align="C")
        pdf.ln()
        # Valeurs (gestion des données manquantes)
        pdf.set_font("Helvetica", "", 10)
        vals = [
            row.get("latitude", ""),
            row.get("longitude", ""),
            str(row.get("Z", "")),
            str(row.get("X", "")),
            str(row.get("Y", ""))
        ]
        for w, v in zip(col_w, vals):
            pdf.cell(w, 6, v or "-", border=1, align="C")
        pdf.ln(10)

        # ---- Vues (photos) ----
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, "VUES :", ln=1)
        x0, y0 = pdf.get_x(), pdf.get_y()
        thumb_w, thumb_h = 60, 45
        photos = photo_dict.get(pid, [])
        if not photos:
            # placeholder si pas de photo
            pdf.set_font("Helvetica", "I", 9)
            pdf.cell(0, 6, "Aucune photo fournie", ln=1, align="L")
            pdf.ln(4)
        else:
            for i, f in enumerate(photos):
                try:
                    img = Image.open(f)
                    bio = io.BytesIO()
                    img.thumbnail((thumb_w*4, thumb_h*4))
                    img.save(bio, format="JPEG")
                    bio.seek(0)
                    x = x0 + (i % 2)*(thumb_w + 5)
                    y = y0 + (i//2)*(thumb_h + 5)
                    pdf.image(bio, x=x, y=y, w=thumb_w, h=thumb_h, type="JPEG")
                except Exception:
                    continue
            pdf.ln(thumb_h*2/3 + 8)

        # ---- Pied de page ----
        pdf.set_y(-25)
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(0, 5, f"Commune : {commune or '-'}", ln=1, align="L")
        pdf.cell(0, 5, "Fiche générée automatiquement", ln=1, align="R")

    # Génération du buffer PDF avec encodage tolérant
    raw = pdf.output(dest="S")
    pdf_bytes = raw.encode("latin-1", errors="ignore")

    st.success("✅ Votre catalogue est prêt !")
    st.download_button(
        label="⬇️ Télécharger le PDF",
        data=pdf_bytes,
        file_name="catalogue_fiches_signaletiques.pdf",
        mime="application/pdf"
    )

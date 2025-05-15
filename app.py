# app.py
import streamlit as st
import pandas as pd
import json
from PIL import Image, ImageDraw, ImageFont
import io
import zipfile
from datetime import datetime

# --- CONFIGURATION A4 en pixels (@150 DPI) ---
A4_W, A4_H = 1240, 1754
MARGIN = 50
CONTENT_W = A4_W - 2 * MARGIN

st.set_page_config(page_title="Fiches bornes – ZIP Images", layout="wide")
st.title("📘 Générateur de fiches A4 par borne (ZIP)")

# 1) Chargement des données
st.sidebar.header("1. Chargement des points")
upl = st.sidebar.file_uploader("CSV/TXT/GeoJSON", type=["csv","txt","json"])
df = None
if upl:
    try:
        if upl.name.lower().endswith(".json"):
            geo = json.load(upl)
            recs = []
            for i,f in enumerate(geo.get("features",[])):
                p, g = f.get("properties",{}), f.get("geometry",{})
                c = g.get("coordinates",[None,None])
                recs.append({
                    "ID": str(p.get("ID",i)),
                    "latitude": p.get("latitude",""), "longitude": p.get("longitude",""),
                    "Z": p.get("Z",""), "X": c[0], "Y": c[1]
                })
            df = pd.DataFrame(recs)
        else:
            df = pd.read_csv(upl, sep=None, engine="python")
            if "ID" not in df.columns:
                df.insert(0,"ID",df.index.astype(str))
    except Exception as e:
        st.error(f"Erreur lecture fichier : {e}")
    else:
        st.success(f"{len(df)} points chargés")

# 2) Infos générales
st.sidebar.header("2. Informations générales")
republique = st.sidebar.text_input("République / État", "République de Côte d'Ivoire")
ministere = st.sidebar.text_input("Ministère / Projet", "Ministère de l’Équipement et de l’Entretien Routier")
projet    = st.sidebar.text_input("Nom du projet", "PIDUCAS – Cadastrage San-Pedro")
commune   = st.sidebar.text_input("Commune", "")
logo_file = st.sidebar.file_uploader("Logo (PNG/JPG)", type=["png","jpg","jpeg"])

# 3) Photos par borne
photo_dict = {}
if df is not None:
    st.header("📸 Photos par borne")
    for _, row in df.iterrows():
        pid = str(row["ID"])
        with st.expander(f"Borne {pid}"):
            files = st.file_uploader(
                f"Photos pour SP {pid}",
                type=["jpg","jpeg","png"],
                accept_multiple_files=True,
                key=f"upl_{pid}"
            )
            if files:
                photo_dict[pid] = files

# Fonction utilitaire pour charger une police TrueType si possible
def load_font(preferred: str, size: int):
    try:
        return ImageFont.truetype(preferred, size)
    except:
        return ImageFont.load_default()

# 4) Génération et ZIP
if st.sidebar.button("🖼️ Générer fiches et télécharger ZIP") and df is not None:
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zipf:
        # Polices
        title_font   = load_font("DejaVuSans-Bold.ttf", 40)
        header_font  = load_font("DejaVuSans.ttf", 18)
        table_h_font = load_font("DejaVuSans-Bold.ttf", 14)
        table_f_font = load_font("DejaVuSans.ttf", 12)
        footer_font  = load_font("DejaVuSans.ttf", 10)

        for _, row in df.iterrows():
            pid = str(row["ID"])
            # Nouvelle image A4
            img = Image.new("RGB", (A4_W, A4_H), "white")
            draw = ImageDraw.Draw(img)

            # --- Logo en haut à gauche
            if logo_file:
                try:
                    logo = Image.open(logo_file)
                    logo.thumbnail((120,120))
                    img.paste(logo, (MARGIN, MARGIN))
                except:
                    pass

            # --- Texte header centré ---
            y = MARGIN
            # République (gros)
            w = draw.textlength(republique, font=title_font)
            draw.text(((A4_W - w)/2, y), republique, font=title_font, fill="black")
            y += 50
            # Ministère & projet (plus petit)
            for txt in (ministere, projet):
                w = draw.textlength(txt, font=header_font)
                draw.text(((A4_W - w)/2, y), txt, font=header_font, fill="black")
                y += 30

            # Ligne de séparation
            draw.line((MARGIN, y, A4_W - MARGIN, y), fill="black", width=2)
            y += 20

            # --- Titre de fiche ---
            titre = f"BORNE GÉODÉSIQUE SP {pid}"
            draw.text((MARGIN, y), titre, font=header_font, fill="black")
            # Date de MAJ à droite
            date_str = datetime.today().strftime("%B %Y")
            ds_w = draw.textlength(f"MAJ : {date_str}", font=table_f_font)
            draw.text((A4_W - MARGIN - ds_w, y+2), f"MAJ : {date_str}", font=table_f_font, fill="black")
            y += 40

            # --- Tableau coordonnées ---
            labels = ["LAT NORD", "LON OUEST", "HAUTEUR (m)", "X (m)", "Y (m)"]
            values = [
                row.get("latitude","") or "-",
                row.get("longitude","") or "-",
                str(row.get("Z","") or "-"),
                str(row.get("X","") or "-"),
                str(row.get("Y","") or "-")
            ]
            cols = len(labels)
            col_w = CONTENT_W // cols
            x0 = MARGIN
            # En-têtes
            for i, lab in enumerate(labels):
                draw.rectangle([x0 + i*col_w, y, x0+(i+1)*col_w, y+35], outline="black", width=1)
                tw = draw.textlength(lab, font=table_h_font)
                draw.text((x0 + i*col_w + (col_w-tw)/2, y+8), lab, font=table_h_font, fill="black")
            y += 35
            # Valeurs
            for i, val in enumerate(values):
                draw.rectangle([x0 + i*col_w, y, x0+(i+1)*col_w, y+30], outline="black", width=1)
                tw = draw.textlength(val, font=table_f_font)
                draw.text((x0 + i*col_w + (col_w-tw)/2, y+7), val, font=table_f_font, fill="black")
            y += 60

            # --- Vues / photos ---
            draw.text((MARGIN, y), "VUES :", font=header_font, fill="black")
            y += 30
            photos = photo_dict.get(pid, [])
            if not photos:
                draw.text((MARGIN, y), "Aucune photo fournie", font=table_f_font, fill="gray")
                y += 30
            else:
                thumb_w, thumb_h = (CONTENT_W - 20) // 2, 200
                for i, f in enumerate(photos[:4]):  # max 4 vues
                    try:
                        p = Image.open(f)
                        p.thumbnail((thumb_w, thumb_h))
                        xi = MARGIN + (i%2)*(thumb_w+20)
                        yi = y + (i//2)*(thumb_h+20)
                        img.paste(p, (xi, yi))
                    except:
                        continue
                y += thumb_h* ((len(photos[:4]) + 1)//2) + 20

            # --- Pied de page ---
            foot_left  = f"Commune : {commune or '-'}"
            foot_right = "Généré automatiquement"
            draw.text((MARGIN, A4_H - MARGIN - 20), foot_left, font=footer_font, fill="black")
            tw = draw.textlength(foot_right, font=footer_font)
            draw.text((A4_W - MARGIN - tw, A4_H - MARGIN - 20), foot_right, font=footer_font, fill="black")

            # Sauvegarde en PNG
            out = io.BytesIO()
            img.save(out, format="PNG")
            zipf.writestr(f"borne_{pid}.png", out.getvalue())

    zip_buf.seek(0)
    st.success("✅ ZIP prêt : vos fiches A4 par borne")
    st.download_button(
        "⬇️ Télécharger le ZIP",
        data=zip_buf,
        file_name="fiches_bornes.zip",
        mime="application/zip"
    )

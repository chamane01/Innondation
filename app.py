import streamlit as st
import pandas as pd
import json
from PIL import Image, ImageDraw, ImageFont
import io
import zipfile
from datetime import datetime
from pyproj import Transformer

# --- CONFIGURATION A4 (pixels) ---
# Format A4 à 150 DPI ≃ 1240×1754 px
A4_W, A4_H = 1240, 1754
MARGIN = 50

# UTM zone for Côte d'Ivoire (zone 30N)
transformer = Transformer.from_crs("EPSG:32630", "EPSG:4326", always_xy=True)

st.set_page_config(page_title="Générateur d’Images A4 par Borne (ZIP)", layout="wide")
st.title("📘 Générateur d’Images A4 par Borne et ZIP")

# 1) Chargement des données
st.sidebar.header("1. Chargement des points")
uploaded = st.sidebar.file_uploader("CSV / TXT / GeoJSON", type=["csv","txt","json"])
df = None
if uploaded:
    try:
        if uploaded.name.lower().endswith(".json"):
            geo = json.load(uploaded)
            feats = geo.get("features", [])
            recs = []
            for i, f in enumerate(feats):
                p = f.get("properties", {})
                g = f.get("geometry", {})
                c = g.get("coordinates", [None, None])
                recs.append({
                    "ID": str(p.get("ID", i)),
                    "X": c[0], "Y": c[1], "Z": p.get("Z",""),
                })
            df = pd.DataFrame(recs)
        else:
            df = pd.read_csv(uploaded, sep=None, engine="python")
            if "ID" not in df.columns:
                df.insert(0, "ID", df.index.astype(str))
    except Exception as e:
        st.error(f"Erreur lecture fichier : {e}")
    else:
        st.success(f"{len(df)} points chargés")
        # calculer lat/long
        df[['longitude', 'latitude']] = df.apply(
            lambda r: transformer.transform(r['X'], r['Y'])[0:2][::-1] if pd.notnull(r['X']) and pd.notnull(r['Y']) else (None, None),
            axis=1, result_type='expand')

# 2) Infos générales
st.sidebar.header("2. Infos générales")
republique = st.sidebar.text_input("République / État", "République de Côte d'Ivoire")
ministere = st.sidebar.text_input("Ministère / Projet", "Ministère de l’Équipement et de l’Entretien Routier")
projet    = st.sidebar.text_input("Projet", "PIDUCAS – Cadastrage San-Pedro")
commune   = st.sidebar.text_input("Commune", "")
# logos
logo_ivory = st.sidebar.file_uploader("Logo Côte d'Ivoire (PNG/JPG)", type=["png","jpg","jpeg"])
logo_mo   = st.sidebar.file_uploader("Logo Maître d'œuvre (PNG/JPG)", type=["png","jpg","jpeg"])
logo_exec = st.sidebar.file_uploader("Logo Entreprise d'exécution (PNG/JPG)", type=["png","jpg","jpeg"])

# 3) Photos par point
photo_dict = {}
if df is not None:
    st.header("📸 Photos par borne")
    for _, row in df.iterrows():
        pid = str(row["ID"])
        with st.expander(f"Borne {pid}"):
            files = st.file_uploader(
                f"Photos pour {pid}",
                type=["jpg","jpeg","png"],
                accept_multiple_files=True,
                key=f"upl_{pid}"
            )
            if files:
                photo_dict[pid] = files

# 4) Génération des images + ZIP
if st.sidebar.button("🖼️ Générer images et télécharger ZIP") and df is not None:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        # Fonts
        try:
            font_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
            font_reg  = ImageFont.truetype("DejaVuSans.ttf", 20)
        except:
            font_bold = ImageFont.load_default()
            font_reg  = ImageFont.load_default()

        for _, row in df.iterrows():
            pid = str(row["ID"])
            img = Image.new("RGB", (A4_W, A4_H), "white")
            draw = ImageDraw.Draw(img)

            # --- Logos en entête ---
            x_l = MARGIN
            for logo_file in [logo_ivory, logo_mo, logo_exec]:
                if logo_file:
                    try:
                        lg = Image.open(logo_file)
                        lg.thumbnail((150,150))
                        img.paste(lg, (x_l, MARGIN), lg if lg.mode=='RGBA' else None)
                        x_l += lg.width + 30
                    except:
                        continue

            # Header texte
            y0 = MARGIN + 160
            draw.text((MARGIN, y0), republique, font=font_bold, fill="black")
            y0 += 30
            draw.text((MARGIN, y0), ministere, font=font_reg, fill="black")
            y0 += 25
            draw.text((MARGIN, y0), projet, font=font_reg, fill="black")

            # Ligne séparatrice
            y_sep = y0 + 40
            draw.line((MARGIN, y_sep, A4_W - MARGIN, y_sep), fill="black", width=2)

            # Titre fiche
            y = y_sep + 30
            draw.text((MARGIN, y), f"BORNE GÉODÉSIQUE SP {pid}", font=font_bold, fill="black")
            y += 35
            date_str = datetime.today().strftime("%d %B %Y")
            draw.text((MARGIN, y), f"MAJ : {date_str}", font=font_reg, fill="black")

            # Tableau coordonnées
            y += 50
            headers = ["LAT NORD","LON OUEST","HAUTEUR (m)","X (m)","Y (m)"]
            vals = [
                f"{row['latitude']:.6f}" if pd.notnull(row['latitude']) else "-",
                f"{row['longitude']:.6f}" if pd.notnull(row['longitude']) else "-",
                str(row.get("Z","-")),
                f"{row['X']:.3f}",
                f"{row['Y']:.3f}"
            ]
            col_w = (A4_W - 2 * MARGIN) // len(headers)
            # Entêtes
            for i, h in enumerate(headers):
                x0 = MARGIN + i * col_w
                draw.rectangle([x0, y, x0 + col_w, y + 40], outline="black", width=1)
                draw.text((x0 + 5, y + 5), h, font=font_bold, fill="black")
            # Valeurs
            y_val = y + 45
            for i, v in enumerate(vals):
                x0 = MARGIN + i * col_w
                draw.rectangle([x0, y_val, x0 + col_w, y_val + 40], outline="black", width=1)
                draw.text((x0 + 5, y_val + 10), v, font=font_reg, fill="black")

            # Vues / photos
            y_ph = y_val + 80
            draw.text((MARGIN, y_ph), "VUES :", font=font_bold, fill="black")
            y_ph += 30
            photos = photo_dict.get(pid, [])
            if not photos:
                draw.text((MARGIN, y_ph), "Aucune photo fournie", font=font_reg, fill="gray")
            else:
                thumb_w, thumb_h = 300, 200
                for i, f in enumerate(photos):
                    try:
                        p = Image.open(f)
                        p.thumbnail((thumb_w, thumb_h))
                        x_ph = MARGIN + (i % 2) * (thumb_w + 20)
                        y_cur = y_ph + (i // 2) * (thumb_h + 20)
                        img.paste(p, (x_ph, y_cur))
                    except:
                        continue

            # Pied de page
            text_cf = commune or "-"
            draw.text((MARGIN, A4_H - 100), f"Commune : {text_cf}", font=font_reg, fill="black")
            draw.text((A4_W - MARGIN - 300, A4_H - 100), "Généré automatiquement", font=font_reg, fill="black")

            # Sauvegarde PNG
            out = io.BytesIO()
            img.save(out, format="PNG")
            zipf.writestr(f"borne_{pid}.png", out.getvalue())

    zip_buffer.seek(0)
    st.success("✅ ZIP prêt : toutes vos images A4 par borne")
    st.download_button(
        "⬇️ Télécharger le ZIP",
        data=zip_buffer,
        file_name="catalogue_bornes_images.zip",
        mime="application/zip"
    )

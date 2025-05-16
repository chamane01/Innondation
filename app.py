# app.py

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

st.set_page_config(page_title="Générateur d’Images A4 par Borne (ZIP)", layout="wide")
st.title("📘 Générateur d’Images A4 par Borne et ZIP")

# 1) Chargement des données
st.sidebar.header("1. Chargement des points")
uploaded = st.sidebar.file_uploader("CSV / TXT / GeoJSON", type=["csv", "txt", "json"])
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
                    "X": c[0], "Y": c[1], "Z": p.get("Z", ""),
                    "latitude": p.get("latitude", ""), "longitude": p.get("longitude", "")
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

        # 1a) conversion UTM -> lat/long si besoin
        if "X" in df.columns and "Y" in df.columns:
            utm_zone = st.sidebar.number_input("Zone UTM", min_value=1, max_value=60, value=31)
            hemisphere = st.sidebar.selectbox("Hémisphère UTM", ["Nord", "Sud"])
            crs_utm = f"+proj=utm +zone={utm_zone} +{'north' if hemisphere=='Nord' else 'south'} +datum=WGS84 +units=m +no_defs"
            transformer = Transformer.from_crs(crs_utm, "EPSG:4326", always_xy=True)
            try:
                lons, lats = transformer.transform(df["X"].values, df["Y"].values)
                df["longitude"] = lons
                df["latitude"] = lats
            except Exception as e:
                st.warning(f"Conversion UTM → lat/long échouée : {e}")

# 2) Infos générales
st.sidebar.header("2. Infos générales")
republique = st.sidebar.text_input("République / État", "République de Côte d'Ivoire")
ministere = st.sidebar.text_input("Ministère / Projet", "Ministère de l’Équipement et de l’Entretien Routier")
projet    = st.sidebar.text_input("Projet", "PIDUCAS – Cadastrage San-Pedro")
commune   = st.sidebar.text_input("Commune", "")

# 2a) Logos multiples
st.sidebar.header("3. Logos")
logo_ci   = st.sidebar.file_uploader("Logo Côte d'Ivoire", type=["png", "jpg", "jpeg"])
logo_moe  = st.sidebar.file_uploader("Logo Maître d'Œuvre", type=["png", "jpg", "jpeg"])
logo_exec = st.sidebar.file_uploader("Logo Entreprise d'Exécution", type=["png", "jpg", "jpeg"])

# 3) Photos par point
photo_dict = {}
if df is not None:
    st.header("📸 Photos par borne")
    for _, row in df.iterrows():
        pid = str(row["ID"])
        with st.expander(f"Borne {pid}"):
            files = st.file_uploader(
                f"Photos pour {pid}",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key=f"upl_{pid}"
            )
            if files:
                photo_dict[pid] = files

# 4) Génération des images + ZIP
if st.sidebar.button("🖼️ Générer images et télécharger ZIP") and df is not None:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        # Chargement des polices
        try:
            font_bold = ImageFont.truetype("arialbd.ttf", size=32)
            font_reg = ImageFont.truetype("arial.ttf", size=24)
        except IOError:
            font_bold = ImageFont.load_default()
            font_reg = ImageFont.load_default()

        for _, row in df.iterrows():
            pid = str(row["ID"])
            img = Image.new("RGB", (A4_W, A4_H), "white")
            draw = ImageDraw.Draw(img)

            # --- Logos en en-tête ---
            logos = [logo_ci, logo_moe, logo_exec]
            logo_max_h = 100
            spacing = 20
            x_cursor = MARGIN
            for logo_file in logos:
                if logo_file:
                    try:
                        logo = Image.open(logo_file)
                        w, h = logo.size
                        ratio = logo_max_h / h
                        logo = logo.resize((int(w * ratio), logo_max_h), Image.ANTIALIAS)
                        img.paste(logo, (x_cursor, MARGIN), logo.convert("RGBA"))
                        x_cursor += logo.width + spacing
                    except:
                        continue

            # Texte header à droite des logos
            header_x = x_cursor
            y0 = MARGIN
            draw.text((header_x, y0), republique, font=font_bold, fill="black")
            y0 += 35
            draw.text((header_x, y0), ministere, font=font_reg, fill="black")
            y0 += 30
            draw.text((header_x, y0), projet, font=font_reg, fill="black")

            # Ligne séparatrice
            y_sep = MARGIN + logo_max_h + 20
            draw.line((MARGIN, y_sep, A4_W - MARGIN, y_sep), fill="black", width=2)

            # --- Titre fiche ---
            y = y_sep + 30
            draw.text((MARGIN, y), f"BORNE GÉODÉSIQUE SP {pid}", font=font_bold, fill="black")
            y += 35
            date_str = datetime.today().strftime("%B %Y")
            draw.text((MARGIN, y), f"MAJ : {date_str}", font=font_reg, fill="black")

            # --- Préparation des valeurs coordonnées ---
            def fmt(val):
                try:
                    return f"{float(val):.6f}"
                except:
                    return "-"
            
            lat_txt = fmt(row.get("latitude", ""))
            lon_txt = fmt(row.get("longitude", ""))
            z_txt   = str(row.get("Z", "") or "-")
            x_txt   = str(row.get("X", "") or "-")
            y_txt   = str(row.get("Y", "") or "-")

            # --- Tableau coordonnées ---
            y += 60
            headers = ["LAT NORD", "LON OUEST", "HAUTEUR", "X (m)", "Y (m)"]
            vals = [lat_txt, lon_txt, z_txt, x_txt, y_txt]
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

            # --- Vues / photos ---
            y_ph = y_val + 80
            draw.text((MARGIN, y_ph), "VUES :", font=font_bold, fill="black")
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

            # --- Pied de page ---
            text_cf = commune or "-"
            draw.text((MARGIN, A4_H - 100), f"Commune : {text_cf}", font=font_reg, fill="black")
            draw.text((A4_W - MARGIN - 300, A4_H - 100), "Généré automatiquement", font=font_reg, fill="black")

            # Sauvegarde PNG en mémoire
            out = io.BytesIO()
            img.save(out, format="PNG")
            zipf.writestr(f"borne_{pid}.png", out.getvalue())

    zip_buffer.seek(0)
    st.success("✅ ZIP prêt : toutes vos images A4 par borne")
    st.download_button(
        "⬇️ Télécharger le ZIP",
        data=zip_buffer,
        file_name="catalogue_bornes_images.zip",
        mime="application/zip"
    )

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
# A4 à 150 DPI ≃ 1240×1754 px
A4_W, A4_H = 1240, 1754
MARGIN = 50

st.set_page_config(page_title="Générateur de Fiches Géodésiques", layout="wide")
st.title("Générateur de Fiches Géodésiques")

# --- SIDEBAR ---
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
                    "X": c[0], "Y": c[1], "Z": p.get("Z",""),
                    "latitude": p.get("latitude",""), "longitude": p.get("longitude","")
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

        # Conversion UTM → WGS84
        if "X" in df.columns and "Y" in df.columns:
            utm_zone = st.sidebar.number_input("Zone UTM", min_value=1, max_value=60, value=31)
            hemisphere = st.sidebar.selectbox("Hémisphère UTM", ["Nord","Sud"])
            crs_utm = f"+proj=utm +zone={utm_zone} +{'north' if hemisphere=='Nord' else 'south'} +datum=WGS84 +units=m +no_defs"
            transformer = Transformer.from_crs(crs_utm, "EPSG:4326", always_xy=True)
            try:
                lons, lats = transformer.transform(df["X"].values, df["Y"].values)
                df["longitude"], df["latitude"] = lons, lats
            except Exception as e:
                st.warning(f"Conversion UTM → lat/long échouée : {e}")

st.sidebar.header("2. Infos générales")
commune    = st.sidebar.text_input("Commune", "SAN-PEDRO")
republique = st.sidebar.text_input("République / État", "République de Côte d'Ivoire")
ministere  = st.sidebar.text_input("Ministère / Projet", "Ministère de l’Équipement et de l’Entretien Routier")
projet     = st.sidebar.text_input("Projet", f"CADASTRAGE DE LA VILLE DE {commune}")

st.sidebar.header("3. Logos")
logo_ci   = st.sidebar.file_uploader("Logo Côte d'Ivoire", type=["png","jpg","jpeg"])
logo_moe  = st.sidebar.file_uploader("Logo Maître d'Œuvre",  type=["png","jpg","jpeg"])
logo_exec = st.sidebar.file_uploader("Logo Exécution",      type=["png","jpg","jpeg"])

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

# 4) Génération + ZIP
if st.sidebar.button("Générer fiches et ZIP") and df is not None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zipf:
        # polices Times New Roman
        try:
            font_bold = ImageFont.truetype("Times New Roman Bold.ttf", size=32)
            font_reg  = ImageFont.truetype("Times New Roman.ttf", size=24)
            font_small= ImageFont.truetype("Times New Roman.ttf", size=18)
        except IOError:
            font_bold = ImageFont.load_default()
            font_reg  = ImageFont.load_default()
            font_small= ImageFont.load_default()

        for _, row in df.iterrows():
            pid = str(row["ID"])
            img = Image.new("RGB", (A4_W, A4_H), "#f5f5f5")
            draw = ImageDraw.Draw(img)

            # container blanc avec ombre
            box = [MARGIN, MARGIN, A4_W-MARGIN, A4_H-MARGIN]
            # ombre
            draw.rectangle([MARGIN+5, MARGIN+5, A4_W-MARGIN+5, A4_H-MARGIN+5], fill="#ddd")
            # fond
            draw.rectangle(box, fill="white")

            # bar de drapeau
            bar_h = 15
            bar_y = MARGIN + bar_h
            draw.rectangle([box[0], box[0], box[2], box[0]+bar_h],
                           fill=None, outline=None)
            # 3 bandes
            third = (box[2]-box[0])//3
            draw.rectangle([box[0], box[0], box[0]+third, box[0]+bar_h], fill="#ff9b00")
            draw.rectangle([box[0]+third, box[0], box[0]+2*third, box[0]+bar_h], fill="white")
            draw.rectangle([box[0]+2*third,box[0], box[2], box[0]+bar_h], fill="#009e49")

            # logos positionnés absolu
            logos = [logo_ci, logo_moe, logo_exec]
            x0 = box[0] + 10
            for lf in logos:
                if lf:
                    try:
                        logo = Image.open(lf).convert("RGBA")
                        h_ratio = 60 / logo.height
                        logo = logo.resize((int(logo.width*h_ratio), 60), Image.ANTIALIAS)
                        img.paste(logo, (x0, box[0]+bar_h+10), logo)
                        x0 += logo.width + 20
                    except:
                        continue

            # header texte centré
            text_x = box[0] + 10
            y = box[0] + bar_h + 80
            draw.text((text_x, y), republique, font=font_bold, fill="black")
            y += 40
            draw.text((text_x, y), ministere, font=font_reg, fill="black")
            y += 35
            draw.text((text_x, y), projet, font=font_reg, fill="black")

            # ligne séparatrice
            y_sep = y + 50
            draw.line([box[0]+10, y_sep, box[2]-10, y_sep], fill="black", width=2)

            # titre fiche
            y2 = y_sep + 30
            draw.text((box[0]+10, y2), f"BORNE GÉODÉSIQUE SP {pid}", font=font_bold, fill="black")
            y2 += 35
            draw.text((box[0]+10, y2), f"MAJ : {datetime.today():%B %Y}", font=font_reg, fill="black")

            # formatage coordonnées
            def fmt(v):
                try: return f"{float(v):.6f}"
                except: return "-"

            lat = fmt(row.get("latitude",""))
            lon = fmt(row.get("longitude",""))
            z   = str(row.get("Z","") or "-")
            x   = str(row.get("X","") or "-")
            y_p = str(row.get("Y","") or "-")

            # tableau
            table_y = y2 + 50
            cols = ["LAT NORD","LON OUEST","HAUTEUR","X (m)","Y (m)"]
            vals = [lat, lon, z, x, y_p]
            col_w = (box[2]-box[0]-20)//len(cols)
            for i, h in enumerate(cols):
                xh = box[0]+10 + i*col_w
                draw.rectangle([xh, table_y, xh+col_w, table_y+35], outline="black")
                draw.text((xh+5, table_y+5), h, font=font_small, fill="black")
                draw.rectangle([xh, table_y+40, xh+col_w, table_y+75], outline="black")
                draw.text((xh+5, table_y+45), vals[i], font=font_small, fill="black")

            # photos
            photo_y = table_y + 120
            draw.text((box[0]+10, photo_y), "VUES :", font=font_bold, fill="black")
            photo_y += 40
            thumbs = photo_dict.get(pid, [])
            if not thumbs:
                draw.text((box[0]+10, photo_y), "Aucune photo fournie", font=font_reg, fill="#666")
            else:
                tw, th = 300, 200
                for i, f in enumerate(thumbs):
                    try:
                        p = Image.open(f)
                        p.thumbnail((tw, th))
                        px = box[0]+10 + (i%2)*(tw+20)
                        py = photo_y + (i//2)*(th+20)
                        img.paste(p, (px, py))
                    except:
                        continue

            # pied de page
            draw.text((box[0]+10, box[3]-70), f"Commune : {commune}", font=font_reg, fill="black")
            draw.text((box[2]-250, box[3]-70), "Généré automatiquement", font=font_reg, fill="black")

            # export
            buf2 = io.BytesIO()
            img.crop(box).save(buf2, format="PNG")
            zipf.writestr(f"fiche_SP_{pid}.png", buf2.getvalue())

    buffer.seek(0)
    st.success("✅ ZIP prêt")
    st.download_button("⬇️ Télécharger ZIP", data=buffer,
                       file_name="fiches_geodesiques.zip", mime="application/zip")

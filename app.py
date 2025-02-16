import streamlit as st
import os
import rasterio
import rasterio.merge
import rasterio.mask
import folium
import math
import matplotlib.pyplot as plt
import numpy as np
from streamlit_folium import st_folium
from folium.plugins import Draw
from io import BytesIO
from datetime import date, datetime

# ------------------------------
# Partie ReportLab pour le rapport
# ------------------------------
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph
from reportlab.lib import colors

# Dimensions standard pour le PDF
PAGE_WIDTH, PAGE_HEIGHT = A4
SECTION_HEIGHT = PAGE_HEIGHT / 3
COLUMN_WIDTH = PAGE_WIDTH / 2

# ==============================
# Fonctions utilitaires - ANALYSE SPATIALE
# ==============================

def load_tiff_files(folder_path):
    """Charge les fichiers TIFF contenus dans un dossier."""
    try:
        tiff_files = [os.path.join(folder_path, f)
                      for f in os.listdir(folder_path) if f.lower().endswith('.tif')]
    except Exception as e:
        st.error(f"Erreur lors de la lecture du dossier {folder_path} : {e}")
        return []
    
    if not tiff_files:
        st.error("Aucun fichier TIFF trouvé dans le dossier.")
        return []
    
    return [f for f in tiff_files if os.path.exists(f)]

def build_mosaic(tiff_files, mosaic_path="mosaic.tif"):
    """Construit une mosaïque à partir de fichiers TIFF."""
    try:
        src_files = [rasterio.open(fp) for fp in tiff_files]
        mosaic, out_trans = rasterio.merge.merge(src_files)
        out_meta = src_files[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans
        })
        with rasterio.open(mosaic_path, "w", **out_meta) as dest:
            dest.write(mosaic)
        for src in src_files:
            src.close()
        return mosaic_path
    except Exception as e:
        st.error(f"Erreur lors de la création de la mosaïque : {e}")
        return None

def create_map(mosaic_file):
    """
    Crée une carte Folium affichant l'emprise de la mosaïque et intégrant
    l'outil de dessin pour pouvoir tracer à la fois des rectangles et des lignes.
    """
    m = folium.Map(location=[0, 0], zoom_start=2)
    
    # Calque indiquant l'emprise de la mosaïque
    mosaic_group = folium.FeatureGroup(name="Mosaïque")
    try:
        with rasterio.open(mosaic_file) as src:
            bounds = src.bounds
            folium.Rectangle(
                bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                color='blue',
                fill=False,
                tooltip="Emprise de la mosaïque"
            ).add_to(mosaic_group)
    except Exception as e:
        st.error(f"Erreur lors de l'ouverture de la mosaïque : {e}")
    
    mosaic_group.add_to(m)
    
    # Outils de dessin : rectangle pour contours, polyline pour profils
    Draw(
        draw_options={
            'rectangle': True,
            'polyline': {'allowIntersection': False},
            'polygon': False,
            'circle': False,
            'marker': False,
            'circlemarker': False
        },
        edit_options={'edit': True, 'remove': True}
    ).add_to(m)
    
    folium.LayerControl().add_to(m)
    
    return m

def generate_contours(mosaic_file, drawing_geometry):
    """
    Génère les courbes de niveau à partir d'une zone définie par drawing_geometry.
    Retourne la figure matplotlib.
    """
    try:
        with rasterio.open(mosaic_file) as src:
            if drawing_geometry is not None:
                out_image, out_transform = rasterio.mask.mask(src, [drawing_geometry], crop=True)
                data = out_image[0]
            else:
                data = src.read(1)
                out_transform = src.transform
            nodata = src.nodata
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier TIFF : {e}")
        return None
    
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)
    
    nrows, ncols = data.shape
    x_coords = np.arange(ncols) * out_transform.a + out_transform.c + out_transform.a/2
    y_coords = np.arange(nrows) * out_transform.e + out_transform.f + out_transform.e/2
    X, Y = np.meshgrid(x_coords, y_coords)
    
    try:
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
    except Exception as e:
        st.error(f"Erreur lors du calcul des valeurs min et max : {e}")
        return None
    
    levels = np.linspace(vmin, vmax, 15)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contour(X, Y, data, levels=levels, cmap='terrain')
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_title("Contours d'élévation")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    return fig

def haversine(lon1, lat1, lon2, lat2):
    """Calcule la distance (m) entre deux points GPS."""
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371000
    return c * r

def interpolate_line(coords, step=50):
    """Interpole des points le long d'une ligne pour obtenir des échantillons réguliers."""
    if len(coords) < 2:
        return coords, [0]
    sampled_points = [coords[0]]
    cumulative_dist = [0]
    for i in range(len(coords)-1):
        start = coords[i]
        end = coords[i+1]
        seg_distance = haversine(start[0], start[1], end[0], end[1])
        num_steps = max(int(seg_distance // step), 1)
        for j in range(1, num_steps+1):
            fraction = j / num_steps
            lon = start[0] + fraction * (end[0]-start[0])
            lat = start[1] + fraction * (end[1]-start[1])
            dist = haversine(sampled_points[-1][0], sampled_points[-1][1], lon, lat)
            sampled_points.append([lon, lat])
            cumulative_dist.append(cumulative_dist[-1] + dist)
    return sampled_points, cumulative_dist

def generate_profile(mosaic_file, coords, profile_title):
    """
    Génère un profil d'élévation le long d'une ligne définie par 'coords'.
    Retourne la figure matplotlib.
    """
    try:
        points, distances = interpolate_line(coords)
        with rasterio.open(mosaic_file) as src:
            elevations = [list(src.sample([p]))[0][0] for p in points]
    except Exception as e:
        st.error(f"Erreur lors de la génération du profil : {e}")
        return None
    
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(distances, elevations, 'b-', linewidth=1.5)
    ax.set_title(profile_title)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Altitude (m)")
    return fig

def store_figure(fig, result_type, title):
    """
    Sauvegarde la figure matplotlib dans un buffer et stocke le résultat
    dans st.session_state["analysis_results"].
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    if "analysis_results" not in st.session_state:
        st.session_state["analysis_results"] = []
    st.session_state["analysis_results"].append({
        "type": result_type,
        "title": title,
        "image": buf.getvalue()
    })

# ==============================
# Interface - Analyse Spatiale
# ==============================

def run_analysis_spatiale():
    st.title("🔍 Analyse Spatiale")
    st.info("Ce module vous permet de générer des contours (à partir de rectangles dessinés) ou des profils d'élévation (à partir de lignes).")
    
    # Initialisation du mode pour cette partie (id unique)
    if "analysis_mode" not in st.session_state:
        st.session_state["analysis_mode"] = "none"
    
    # Initialisation pour conserver les dessins
    if "raw_drawings" not in st.session_state:
        st.session_state["raw_drawings"] = []
    
    # Saisie du nom de la carte
    map_name = st.text_input("Nom de votre carte", value="Ma Carte", key="analysis_map_name")
    
    # Chargement et construction de la mosaïque
    folder_path = "TIFF"
    if not os.path.exists(folder_path):
        st.error("Dossier TIFF introuvable")
        return
    tiff_files = load_tiff_files(folder_path)
    if not tiff_files:
        return
    mosaic_path = build_mosaic(tiff_files)
    if not mosaic_path:
        return
    
    # Création de la carte interactive avec outils de dessin
    m = create_map(mosaic_path)
    st.write("**Utilisez l'outil de dessin sur la carte ci-dessous.**")
    map_data = st_folium(m, width=700, height=500, key="analysis_map")
    
    # Sauvegarder les dessins dans la session pour persistance
    if map_data is not None and isinstance(map_data, dict) and "all_drawings" in map_data:
        st.session_state["raw_drawings"] = map_data["all_drawings"]
    
    # Zone d'options sous la carte
    options_container = st.container()
    if st.session_state["analysis_mode"] == "none":
        col1, col2 = options_container.columns(2)
        if col1.button("Tracer des profils", key="btn_profiles"):
            st.session_state["analysis_mode"] = "profiles"
        if col2.button("Générer des contours", key="btn_contours"):
            st.session_state["analysis_mode"] = "contours"
    
    # Mode Générer des contours (rectangle)
    if st.session_state["analysis_mode"] == "contours":
        st.subheader("Générer des contours")
        drawing_geometries = []
        raw_drawings = st.session_state.get("raw_drawings", [])
        if isinstance(raw_drawings, list):
            for drawing in raw_drawings:
                # Les rectangles dessinés apparaissent en tant que Polygones
                if isinstance(drawing, dict) and drawing.get("geometry", {}).get("type") == "Polygon":
                    drawing_geometries.append(drawing.get("geometry"))
        if not drawing_geometries:
            st.warning("Veuillez dessiner un rectangle sur la carte pour définir une zone.")
        else:
            for i, geom in enumerate(drawing_geometries, start=1):
                st.markdown(f"**Contours pour l'emprise {i}**")
                fig = generate_contours(mosaic_path, geom)
                if fig is not None:
                    st.pyplot(fig)
                    store_figure(fig, "contour", f"Contours - Emprise {i}")
        if st.button("Retour", key="retour_contours"):
            st.session_state["analysis_mode"] = "none"
    
    # Mode Tracer des profils (ligne)
    if st.session_state["analysis_mode"] == "profiles":
        st.subheader("Tracer des profils")
        raw_drawings = st.session_state.get("raw_drawings", [])
        current_drawings = []
        if isinstance(raw_drawings, list):
            current_drawings = [d for d in raw_drawings if isinstance(d, dict) and d.get("geometry", {}).get("type") == "LineString"]
        if not current_drawings:
            st.info("Aucune ligne tracée pour le moment. Veuillez dessiner une ligne sur la carte.")
        else:
            for i, drawing in enumerate(current_drawings):
                profile_title = f"{map_name} - Profil {i+1}"
                st.markdown(f"#### {profile_title}")
                try:
                    fig = generate_profile(mosaic_path, drawing["geometry"]["coordinates"], profile_title)
                    if fig is not None:
                        st.pyplot(fig)
                        store_figure(fig, "profile", profile_title)
                except Exception as e:
                    st.error(f"Erreur de traitement : {e}")
        if st.button("Retour", key="retour_profiles"):
            st.session_state["analysis_mode"] = "none"

# ==============================
# Fonctions utilitaires - RAPPORT
# ==============================

def create_element_controller():
    with st.expander("➕ Ajouter un élément", expanded=True, key="rapport_elem_expander"):
        col1, col2 = st.columns(2)
        with col1:
            elem_type = st.selectbox("Type", ["Image", "Texte"], key="elem_type")
            size = st.selectbox("Taille", ["Grand", "Moyen", "Petit"], key="elem_size")
        with col2:
            vertical_pos = st.selectbox("Position verticale", ["Haut", "Milieu", "Bas"], key="v_pos")
            horizontal_options = ["Gauche", "Droite", "Centre"] if size == "Petit" else ["Gauche", "Droite"]
            horizontal_pos = st.selectbox("Position horizontale", horizontal_options, key="h_pos")
        
        if elem_type == "Image":
            content = st.file_uploader("Contenu (image)", type=["png", "jpg", "jpeg"], key="content_image")
            image_title = st.text_input("Titre de l'image", max_chars=50, key="image_title")
            description = st.text_input("Description brève (max 100 caractères)", max_chars=100, key="image_desc")
        else:
            content = st.text_area("Contenu", key="content_text")
        
        if st.button("Valider l'élément", key="validate_element"):
            if elem_type == "Image" and content is None:
                st.error("Veuillez charger une image pour cet élément.")
                return None
            element_data = {
                "type": elem_type,
                "size": size,
                "v_pos": vertical_pos,
                "h_pos": horizontal_pos,
                "content": content,
            }
            if elem_type == "Image":
                element_data["image_title"] = image_title
                element_data["description"] = description
            return element_data
    return None

def calculate_dimensions(size):
    dimensions = {
        "Grand": (PAGE_WIDTH, SECTION_HEIGHT),
        "Moyen": (COLUMN_WIDTH, SECTION_HEIGHT),
        "Petit": (COLUMN_WIDTH / 1.5, SECTION_HEIGHT)
    }
    return dimensions.get(size, (PAGE_WIDTH, SECTION_HEIGHT))

def calculate_position(element):
    vertical_offset = {"Haut": 0, "Milieu": SECTION_HEIGHT, "Bas": SECTION_HEIGHT*2}[element['v_pos']]
    
    if element['size'] == "Grand":
        return (0, PAGE_HEIGHT - vertical_offset - SECTION_HEIGHT)
    
    if element['h_pos'] == "Gauche":
        x = 0
    elif element['h_pos'] == "Droite":
        x = COLUMN_WIDTH
    else:  # Centre
        x = COLUMN_WIDTH / 2 - calculate_dimensions(element['size'])[0] / 2
    
    return (x, PAGE_HEIGHT - vertical_offset - SECTION_HEIGHT)

def draw_metadata(c, metadata):
    margin = 40
    x_left = margin
    y_top = PAGE_HEIGHT - margin
    line_height = 16

    logo_drawn = False
    if metadata['logo']:
        try:
            img = ImageReader(metadata['logo'])
            img_width, img_height = img.getSize()
            aspect = img_height / img_width
            desired_width = 40
            desired_height = desired_width * aspect
            
            c.drawImage(img, x_left, y_top - desired_height, width=desired_width, height=desired_height, preserveAspectRatio=True, mask='auto')
            logo_drawn = True
        except Exception as e:
            st.error(f"Erreur de chargement du logo: {str(e)}")
    
    if logo_drawn:
        x_title = x_left + 50
        y_title = y_top - 20
    else:
        x_title = x_left
        y_title = y_top - 20
    
    c.setFont("Helvetica-Bold", 20)
    c.setFillColor(colors.black)
    if metadata.get('titre'):
        c.drawString(x_title, y_title, metadata['titre'])
    
    c.setFont("Helvetica", 14)
    y_company = y_title - 25
    if metadata.get('company'):
        c.drawString(x_title, y_company, metadata['company'])
    
    y_line = y_company - 10
    c.setStrokeColor(colors.darkgray)
    c.setLineWidth(2)
    c.line(x_left, y_line, x_left + 150, y_line)
    c.setLineWidth(1)
    
    y_text = y_line - 20
    infos = [
        ("ID Rapport", metadata['report_id']),
        ("Date", metadata['date'].strftime('%d/%m/%Y') if hasattr(metadata['date'], "strftime") else metadata['date']),
        ("Heure", metadata['time'].strftime('%H:%M') if hasattr(metadata['time'], "strftime") else metadata['time']),
        ("Éditeur", metadata['editor']),
        ("Localisation", metadata['location'])
    ]
    
    value_x_offset = x_left + 70
    for label, value in infos:
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(colors.black)
        c.drawString(x_left, y_text, label + ":")
        c.setFont("Helvetica", 10)
        c.drawString(value_x_offset, y_text, str(value))
        y_text -= line_height

def generate_pdf(elements, metadata):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    
    c.setAuthor(metadata['editor'])
    c.setTitle(metadata['report_id'])
    
    for element in elements:
        width, height = calculate_dimensions(element['size'])
        x, y = calculate_position(element)
        
        if element['type'] == "Image":
            if element["content"] is not None:
                try:
                    img = ImageReader(element["content"])
                    top_margin = 20
                    bottom_margin = 20
                    horizontal_scale = 0.9
                    image_actual_width = width * horizontal_scale
                    image_actual_height = height - top_margin - bottom_margin
                    image_x = x + (width - image_actual_width) / 2
                    image_y = y + bottom_margin
                    c.drawImage(img, image_x, image_y, width=image_actual_width, height=image_actual_height, preserveAspectRatio=True, mask='auto')
                    
                    if element.get("image_title"):
                        c.setFont("Helvetica-Bold", 12)
                        image_title = element["image_title"].upper()
                        c.drawCentredString(x + width / 2, y + height - top_margin / 2, image_title)
                    
                    if element.get("description"):
                        c.setFont("Helvetica", 10)
                        c.setFillColor(colors.gray)
                        c.drawRightString(x + width - 10, y + bottom_margin / 2, element["description"][:100])
                        c.setFillColor(colors.black)
                except Exception as e:
                    st.error(f"Erreur d'image: {str(e)}")
            else:
                st.error("Une image validée est introuvable.")
        else:
            text = element['content']
            style = getSampleStyleSheet()["Normal"]
            style.fontSize = 14 if element['size'] == "Grand" else 12 if element['size'] == "Moyen" else 10
            p = Paragraph(text, style)
            p.wrapOn(c, width, height)
            p.drawOn(c, x, y)
    
    draw_metadata(c, metadata)
    
    c.save()
    buffer.seek(0)
    return buffer

def display_elements_preview(elements):
    st.markdown("## Aperçu des éléments validés")
    for idx, element in enumerate(elements, start=1):
        st.markdown(f"**Élément {idx}**")
        if element["type"] == "Image":
            st.image(element["content"], width=200)
            if element.get("image_title"):
                st.markdown(f"*Titre de l'image :* **{element['image_title'].upper()}**")
            if element.get("description"):
                st.markdown(
                    f"<span style='color:gray'>*Description :* {element['description']}</span>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown(f"**Texte :** {element['content']}")
        st.markdown("---")

# ==============================
# Interface - Rapport
# ==============================

def run_report():
    st.title("📄 Génération de Rapport")
    
    # Sidebar dédiée au rapport (identifiant unique)
    with st.sidebar:
        st.header("📝 Métadonnées du Rapport")
        titre = st.text_input("Titre principal", key="rapport_titre")
        report_id = st.text_input("ID du rapport", key="rapport_id")
        report_date = st.date_input("Date du rapport", date.today(), key="rapport_date")
        report_time = st.time_input("Heure du rapport", datetime.now().time(), key="rapport_time")
        editor = st.text_input("Éditeur", key="rapport_editor")
        location = st.text_input("Localisation", key="rapport_location")
        company = st.text_input("Société", key="rapport_company")
        logo = st.file_uploader("Logo", type=["png", "jpg", "jpeg"], key="rapport_logo")
    
    metadata = {
        'titre': titre,
        'report_id': report_id,
        'date': report_date,
        'time': report_time,
        'editor': editor,
        'location': location,
        'company': company,
        'logo': logo
    }
    
    # Initialisation des éléments du rapport dans la session
    if "elements" not in st.session_state:
        st.session_state["elements"] = []
    elements = st.session_state["elements"]
    
    st.markdown("### 📌 Sélectionner des cartes issues de l'analyse spatiale")
    if "analysis_results" in st.session_state and st.session_state["analysis_results"]:
        # Sélection multiple des résultats d'analyse spatiale
        options = {f"{i+1} - {res['title']}": i for i, res in enumerate(st.session_state["analysis_results"])}
        selected = st.multiselect("Choisissez les cartes à ajouter au rapport", list(options.keys()), key="rapport_select_analysis")
        for opt in selected:
            idx = options[opt]
            image_data = st.session_state["analysis_results"][idx]["image"]
            # Ajout de l'image en évitant les doublons
            if not any(el.get("analysis_ref") == idx for el in elements if el["type"] == "Image"):
                elements.append({
                    "type": "Image",
                    "size": "Grand",
                    "v_pos": "Haut",
                    "h_pos": "Gauche",
                    "content": image_data,
                    "image_title": st.session_state["analysis_results"][idx]["title"],
                    "description": "Carte générée depuis l'analyse spatiale",
                    "analysis_ref": idx
                })
        st.success("Les cartes sélectionnées ont été ajoutées aux éléments du rapport.")
    else:
        st.info("Aucune carte issue de l'analyse spatiale n'est disponible pour le moment.")
    
    st.markdown("### Ajouter d'autres éléments")
    new_element = create_element_controller()
    if new_element:
        elements.append(new_element)
        st.session_state["elements"] = elements
        st.success("Élément validé avec succès !")
    
    if elements:
        display_elements_preview(elements)
    
    if elements and st.button("Générer le PDF", key="generate_pdf"):
        pdf = generate_pdf(elements, metadata)
        st.success("✅ Rapport généré avec succès!")
        st.download_button("Télécharger le PDF", pdf, "rapport_structuré.pdf", "application/pdf")

# ==============================
# Application Principale
# ==============================

def main():
    st.set_page_config(page_title="Application SIG & Rapport", layout="wide")
    
    # Menu principal dans la sidebar
    menu = st.sidebar.radio("Menu Principal", ["Analyse Spatiale", "Rapport"], key="main_menu")
    
    if menu == "Analyse Spatiale":
        run_analysis_spatiale()
    elif menu == "Rapport":
        run_report()

if __name__ == "__main__":
    main()

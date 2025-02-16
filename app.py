import streamlit as st
from datetime import date, datetime
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph
from reportlab.lib import colors

# Configuration de la page
st.set_page_config(page_title="Générateur Structuré", layout="centered")

# Dimensions standard
PAGE_WIDTH, PAGE_HEIGHT = A4
SECTION_HEIGHT = PAGE_HEIGHT / 3
COLUMN_WIDTH = PAGE_WIDTH / 2

def create_element_controller():
    with st.expander("➕ Ajouter un élément", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            elem_type = st.selectbox("Type", ["Image", "Texte"], key="elem_type")
            size = st.selectbox("Taille", ["Grand", "Moyen", "Petit"], key="elem_size")
        with col2:
            vertical_pos = st.selectbox("Position verticale", ["Haut", "Milieu", "Bas"], key="v_pos")
            horizontal_pos = st.selectbox("Position horizontale", ["Gauche", "Droite", "Centre"] 
                                      if size == "Petit" else ["Gauche", "Droite"], key="h_pos")
        
        content = st.file_uploader("Contenu", type=["png", "jpg", "jpeg"]) if elem_type == "Image" else st.text_area("Contenu")
        
        if st.button("Valider l'élément"):
            return {
                "type": elem_type,
                "size": size,
                "v_pos": vertical_pos,
                "h_pos": horizontal_pos,
                "content": content
            }
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
    else: # Centre
        x = COLUMN_WIDTH / 2 - calculate_dimensions(element['size'])[0] / 2
    
    return (x, PAGE_HEIGHT - vertical_offset - SECTION_HEIGHT)

def draw_metadata(c, metadata):
    margin = 40
    x_left = margin
    y_top = PAGE_HEIGHT - margin
    line_height = 14
    
    # Logo compact
    if metadata['logo']:
        try:
            img = ImageReader(metadata['logo'])
            img_width, img_height = img.getSize()
            aspect = img_height / img_width
            desired_width = 40  # Taille réduite
            desired_height = desired_width * aspect
            
            c.drawImage(img, x_left, y_top - desired_height, 
                       width=desired_width, height=desired_height, 
                       preserveAspectRatio=True, mask='auto')
            y_top -= desired_height + 10
        except Exception as e:
            st.error(f"Erreur de chargement du logo: {str(e)}")

    # Style minimaliste
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.darkgray)
    
    # Ligne de séparation fine
    c.line(x_left, y_top - 5, x_left + 150, y_top - 5)
    y_top -= 15
    
    # Métadonnées essentielles
    infos = [
        ("ID", metadata['report_id']),
        ("Date", metadata['date'].strftime('%d/%m/%Y')),
        ("Heure", metadata['time'].strftime('%H:%M')),
        (metadata['location'], metadata['editor']),
        (metadata['company'], "")
    ]
    
    for line in infos:
        c.setFont("Helvetica-Bold", 8)
        c.drawString(x_left, y_top, line[0])
        c.setFont("Helvetica", 8)
        c.drawString(x_left + 60, y_top, line[1])
        y_top -= line_height

def generate_pdf(elements, metadata):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    
    # Métadonnées techniques
    c.setAuthor(metadata['editor'])
    c.setTitle(metadata['report_id'])
    
    # Éléments principaux
    for element in elements:
        width, height = calculate_dimensions(element['size'])
        x, y = calculate_position(element)
        
        if element['type'] == "Image":
            try:
                img = ImageReader(element['content'])
                c.drawImage(img, x, y, width=width, height=height, preserveAspectRatio=True, mask='auto')
            except Exception as e:
                st.error(f"Erreur d'image: {str(e)}")
        else:
            text = element['content']
            style = getSampleStyleSheet()["Normal"]
            style.fontSize = 14 if element['size'] == "Grand" else 12 if element['size'] == "Moyen" else 10
            p = Paragraph(text, style)
            p.wrapOn(c, width, height)
            p.drawOn(c, x, y)
    
    # Métadonnées visuelles
    draw_metadata(c, metadata)
    
    c.save()
    buffer.seek(0)
    return buffer

def main():
    st.title("📐 Conception de Rapport Structuré")
    
    # Configuration des métadonnées
    with st.expander("⚙️ Métadonnées", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            report_id = st.text_input("ID du rapport", value=f"RPT-{datetime.now().strftime('%y%m%d%H%M')}")
            company = st.text_input("Société", value="SARL")
        with col2:
            report_date = st.date_input("Date", value=date.today())
            report_time = st.time_input("Heure", value=datetime.now().time())
            location = st.text_input("Lieu", value="VILLE")
            editor = st.text_input("Rédacteur", value="NOM")
            logo = st.file_uploader("Logo", type=["png", "jpg", "jpeg"])
    
    metadata = {
        "report_id": report_id,
        "company": company,
        "date": report_date,
        "time": report_time,
        "location": location,
        "editor": editor,
        "logo": logo
    }
    
    # Gestion des éléments
    elements = []
    new_element = create_element_controller()
    if new_element:
        elements.append(new_element)
    
    # Affichage et génération
    if elements:
        with st.container():
            st.markdown("### Prévisualisation")
            for element in elements:
                with st.container(border=True):
                    cols = st.columns([1,4])
                    with cols[0]:
                        st.markdown(f"**{element['size']}** ({element['v_pos']}-{element['h_pos']})")
                    with cols[1]:
                        if element['type'] == "Image":
                            st.image(element['content'], use_container_width=True)
                        else:
                            st.markdown(element['content'])
        
        pdf_buffer = generate_pdf(elements, metadata)
        st.download_button(
            label="📤 Exporter PDF",
            data=pdf_buffer,
            file_name=f"{report_id}.pdf",
            mime="application/pdf"
        )

if __name__ == "__main__":
    main()

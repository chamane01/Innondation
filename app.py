import streamlit as st
from datetime import datetime
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph

# Configuration
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 20
HEADER_HEIGHT = 80
SECTION_HEIGHT = (PAGE_HEIGHT - HEADER_HEIGHT - MARGIN*2) / 3
COLUMN_WIDTH = (PAGE_WIDTH - MARGIN*2) / 2

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
        image_title = st.text_input("Titre de l'image") if elem_type == "Image" else ""
        image_desc = st.text_area("Description de l'image") if elem_type == "Image" else ""
        
        if st.button("Valider l'élément"):
            return {
                "type": elem_type,
                "size": size,
                "v_pos": vertical_pos,
                "h_pos": horizontal_pos,
                "content": content,
                "title": image_title,
                "description": image_desc
            }
    return None

def calculate_dimensions(size):
    return {
        "Grand": (PAGE_WIDTH - MARGIN*2, SECTION_HEIGHT),
        "Moyen": (COLUMN_WIDTH - MARGIN, SECTION_HEIGHT),
        "Petit": ((COLUMN_WIDTH - MARGIN*2)/1.5, SECTION_HEIGHT)
    }[size]

def calculate_position(element):
    vertical_offset = {"Haut": HEADER_HEIGHT + MARGIN, 
                      "Milieu": HEADER_HEIGHT + SECTION_HEIGHT + MARGIN*2,
                      "Bas": HEADER_HEIGHT + SECTION_HEIGHT*2 + MARGIN*3}[element['v_pos']]
    
    if element['size'] == "Grand":
        return (MARGIN, PAGE_HEIGHT - vertical_offset - SECTION_HEIGHT)
    
    if element['h_pos'] == "Gauche":
        x = MARGIN
    elif element['h_pos'] == "Droite":
        x = COLUMN_WIDTH + MARGIN
    else: # Centre
        x = COLUMN_WIDTH/2 + MARGIN - calculate_dimensions(element['size'])[0]/2
    
    return (x, PAGE_HEIGHT - vertical_offset - SECTION_HEIGHT)

def draw_header(c, metadata):
    styles = getSampleStyleSheet()
    header_style = ParagraphStyle(
        name='Header',
        parent=styles['Normal'],
        fontSize=8,
        leading=10,
        alignment=2
    )
    
    # Logo
    if metadata['logo']:
        try:
            logo = ImageReader(metadata['logo'])
            c.drawImage(logo, PAGE_WIDTH-100, PAGE_HEIGHT-50, width=50, height=50, preserveAspectRatio=True)
        except: pass
    
    # Métadonnées
    header_text = f"""
    <b>ID Rapport:</b> {metadata['id']}<br/>
    <b>Société:</b> {metadata['company']}<br/>
    <b>Rédacteur:</b> {metadata['author']}<br/>
    <b>Date:</b> {metadata['date']}<br/>
    <b>Heure:</b> {metadata['time']}<br/>
    <b>Lieu:</b> {metadata['location']}
    """
    p = Paragraph(header_text, header_style)
    p.wrapOn(c, 200, 100)
    p.drawOn(c, PAGE_WIDTH - 200 - MARGIN, PAGE_HEIGHT - 60)

def generate_pdf(elements, metadata):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    
    # En-tête
    draw_header(c, metadata)
    
    # Contenu principal
    for element in elements:
        width, height = calculate_dimensions(element['size'])
        x, y = calculate_position(element)
        
        if element['type'] == "Image":
            try:
                # Titre
                c.setFont("Helvetica-Bold", 12)
                c.drawString(x + 5, y + height - 15, element['title'][:50])
                
                # Image
                img = ImageReader(element['content'])
                c.drawImage(img, x, y, width=width, height=height-40, preserveAspectRatio=True)
                
                # Description
                c.setFont("Helvetica", 8)
                c.drawString(x + 5, y + 5, element['description'][:100])
            except: pass
        else:
            c.setFont("Helvetica", 10)
            text = element['content'].replace('\n', '<br/>')
            style = ParagraphStyle(
                name='TextContent',
                fontSize=14 if element['size'] == "Grand" else 12,
                leading=16,
                textColor='#000000'
            )
            p = Paragraph(text, style)
            p.wrapOn(c, width, height)
            p.drawOn(c, x, y)
    
    c.save()
    buffer.seek(0)
    return buffer

def main():
    st.title("📐 Conception de Rapport Structuré")
    
    # Métadonnées
    with st.expander("⚙️ Métadonnées", expanded=True):
        cols = st.columns(2)
        metadata = {
            'id': cols[0].text_input("ID du rapport", value=f"RAPPORT-{datetime.now().strftime('%Y%m%d')}"),
            'company': cols[1].text_input("Société", value="ENTREPRISE SARL"),
            'author': cols[0].text_input("Rédacteur", value="Jean Dupont"),
            'date': cols[1].date_input("Date", datetime.now()).strftime('%d/%m/%Y'),
            'time': cols[0].time_input("Heure", datetime.now()).strftime('%H:%M'),
            'location': cols[1].text_input("Lieu", value="Paris"),
            'logo': st.file_uploader("Logo", type=["png", "jpg", "jpeg"])
        }
    
    # Gestion des éléments
    elements = []
    new_element = create_element_controller()
    if new_element:
        elements.append(new_element)
    
    # Génération PDF
    if elements:
        pdf_buffer = generate_pdf(elements, metadata)
        st.download_button(
            label="📤 Exporter le PDF",
            data=pdf_buffer,
            file_name=f"{metadata['id']}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

if __name__ == "__main__":
    main()

import streamlit as st
from datetime import datetime
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph

# Configuration réaliste pour A4
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 30
HEADER_HEIGHT = 60
CONTENT_WIDTH = PAGE_WIDTH - MARGIN*2

# Tailles réalistes
SIZES = {
    "Grand": (CONTENT_WIDTH, 200),
    "Moyen": (CONTENT_WIDTH/2 - 10, 150),
    "Petit": (CONTENT_WIDTH/3 - 15, 100)
}

def create_element_controller():
    with st.expander("➕ Ajouter un élément", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            elem_type = st.selectbox("Type", ["Image", "Texte"], key="elem_type")
            size = st.selectbox("Taille", list(SIZES.keys()), key="elem_size")
        
        # Prévisualisation miniature
        preview = None
        if elem_type == "Image":
            content = st.file_uploader("Téléverser l'image", type=["png", "jpg", "jpeg"])
            if content:
                preview = content
                st.image(content, caption="Aperçu de l'image", width=200)
        else:
            content = st.text_area("Contenu textuel")
        
        if content:
            title = st.text_input("Titre (images uniquement)") if elem_type == "Image" else ""
            description = st.text_area("Description (images uniquement)") if elem_type == "Image" else ""
            
            if st.button("Ajouter au rapport"):
                return {
                    "type": elem_type,
                    "size": size,
                    "content": content,
                    "title": title,
                    "description": description
                }
    return None

def calculate_position(elements, new_element):
    y_position = HEADER_HEIGHT + MARGIN
    for elem in elements:
        y_position += SIZES[elem['size']][1] + 10
    return y_position

def draw_header(c, metadata):
    # Logo et ID
    if metadata['logo']:
        try:
            logo = ImageReader(metadata['logo'])
            c.drawImage(logo, PAGE_WIDTH-100, PAGE_HEIGHT-50, width=50, height=50, preserveAspectRatio=True)
        except: pass
    
    c.setFont("Helvetica-Bold", 10)
    c.drawRightString(PAGE_WIDTH-MARGIN, PAGE_HEIGHT-40, metadata['id'])
    c.setFont("Helvetica", 8)
    c.drawRightString(PAGE_WIDTH-MARGIN, PAGE_HEIGHT-55, f"Rédigé par {metadata['author']}")
    c.drawRightString(PAGE_WIDTH-MARGIN, PAGE_HEIGHT-70, f"{metadata['date']} | {metadata['location']}")

def generate_pdf(elements, metadata):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    
    # En-tête
    draw_header(c, metadata)
    
    # Contenu principal
    y_position = PAGE_HEIGHT - HEADER_HEIGHT - MARGIN
    for element in elements:
        width, height = SIZES[element['size']]
        
        # Positionnement automatique
        if y_position - height < MARGIN:
            c.showPage()
            draw_header(c, metadata)
            y_position = PAGE_HEIGHT - HEADER_HEIGHT - MARGIN
        
        x_position = MARGIN
        if element['size'] == "Moyen" and len(elements) > 1:
            x_position = CONTENT_WIDTH/2 + MARGIN if len(elements)%2 else MARGIN
        
        # Dessin de l'élément
        if element['type'] == "Image":
            try:
                img = ImageReader(element['content'])
                c.drawImage(img, x_position, y_position - height, width=width, height=height-30, preserveAspectRatio=True)
                
                # Titre et description
                c.setFont("Helvetica-Bold", 12)
                c.drawString(x_position + 5, y_position - height + 20, element['title'][:50])
                c.setFont("Helvetica", 8)
                c.drawString(x_position + 5, y_position - height + 5, element['description'][:75])
            except: pass
        else:
            c.setFont("Helvetica", 12)
            text = element['content']
            p = Paragraph(text, ParagraphStyle(name='Normal', fontSize=12, leading=14))
            p.wrap(width, height)
            p.drawOn(c, x_position, y_position - height)
        
        y_position -= height + 20
    
    c.save()
    buffer.seek(0)
    return buffer

def main():
    st.title("📄 Constructeur de Rapport Professionnel")
    
    # Métadonnées
    with st.expander("📌 Métadonnées", expanded=True):
        cols = st.columns(2)
        metadata = {
            'id': cols[0].text_input("ID du rapport", value=f"RAPPORT-{datetime.now().strftime('%y%m%d%H%M')}"),
            'author': cols[1].text_input("Rédacteur", value="Jean Dupont"),
            'date': cols[0].date_input("Date", datetime.now()).strftime('%d/%m/%Y'),
            'location': cols[1].text_input("Lieu", value="Paris"),
            'logo': st.file_uploader("Logo d'entreprise", type=["png", "jpg", "jpeg"])
        }
    
    # Éléments du rapport
    elements = []
    new_element = create_element_controller()
    if new_element:
        elements.append(new_element)
    
    # Prévisualisation structurée
    if elements:
        st.subheader("Éléments ajoutés")
        for idx, elem in enumerate(elements, 1):
            with st.container(border=True):
                cols = st.columns([1,4])
                with cols[0]:
                    st.markdown(f"**Élément {idx}**\n\n*{elem['size']}*")
                with cols[1]:
                    if elem['type'] == "Image":
                        st.image(elem['content'], width=150)
                        st.caption(f"**{elem['title']}**\n{elem['description']}")
                    else:
                        st.markdown(elem['content'])
        
        # Génération PDF
        pdf_buffer = generate_pdf(elements, metadata)
        st.download_button(
            label="📥 Télécharger le PDF final",
            data=pdf_buffer,
            file_name=f"{metadata['id']}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

if __name__ == "__main__":
    main()

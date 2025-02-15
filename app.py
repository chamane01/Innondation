import streamlit as st
from datetime import date
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import base64

# Configuration de la page
st.set_page_config(page_title="Générateur de Rapport", layout="wide")

def create_report():
    st.title("📄 Générateur de Rapport Professionnel")

    # Réinitialisation de l'état
    if 'preview_ready' not in st.session_state:
        st.session_state.preview_ready = False

    # Section de configuration
    with st.form("config_form"):
        st.header("⚙️ Configuration du Rapport")
        
        # Logo et texte
        col1, col2 = st.columns([1, 3])
        with col1:
            logo = st.file_uploader("Logo (300x300px)", type=["png", "jpg", "jpeg"])
            logo_text = st.text_input("Texte sous le logo")
        
        # Informations de base
        with col2:
            report_date = st.date_input("Date du rapport", date.today())
            report_id = st.text_input("ID du rapport", value=f"RAPPORT-{date.today().isoformat()}")
        
        # Section images
        st.subheader("🖼️ Contenu Visuel")
        num_images = st.number_input("Nombre d'images", min_value=1, max_value=5, value=1)
        
        images = []
        for i in range(num_images):
            with st.expander(f"Image #{i+1}", expanded=(i == 0)):
                img = st.file_uploader(f"Fichier image {i+1}", type=["png", "jpg", "jpeg"], key=f"img{i}")
                title = st.text_input(f"Titre {i+1}", key=f"title{i}")
                description = st.text_area(f"Description {i+1}", key=f"desc{i}")
                images.append((img, title, description))
        
        # Notes générales
        general_notes = st.text_area("📝 Notes Générales", height=150)
        
        if st.form_submit_button("👁️ Prévisualiser le Rapport"):
            st.session_state.preview_ready = True

    # Affichage du rapport
    if st.session_state.preview_ready:
        st.success("✅ Rapport prêt pour l'export!")
        
        with st.container():
            # Style CSS pour la prévisualisation
            st.markdown("""
                <style>
                .report-preview {
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                    margin: 20px 0;
                }
                .header-section {
                    border-bottom: 2px solid #eee;
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                }
                .image-section {
                    margin: 25px 0;
                }
                </style>
            """, unsafe_allow_html=True)

            # Début du conteneur de prévisualisation
            with st.markdown('<div class="report-preview">', unsafe_allow_html=True):
                
                # En-tête
                header_cols = st.columns([1, 3])
                with header_cols[0]:
                    if logo:
                        st.image(logo, width=200)
                        if logo_text:
                            st.markdown(f"<div style='text-align: center; color: #666;'>{logo_text}</div>", 
                                      unsafe_allow_html=True)
                
                with header_cols[1]:
                    st.markdown(f"<h1 style='color: #2c3e50; margin-top: 0;'>Rapport #{report_id}</h1>", 
                              unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size: 16px; color: #7f8c8d;'>Date : {report_date.strftime('%d %B %Y')}</div>", 
                              unsafe_allow_html=True)
                
                st.markdown('<div class="header-section"></div>', unsafe_allow_html=True)
                
                # Contenu principal
                for idx, (img, title, description) in enumerate(images):
                    if img:
                        cols = st.columns([1, 2])
                        with cols[0]:
                            st.image(img, use_container_width=True)
                        with cols[1]:
                            st.markdown(f"<h3 style='color: #34495e;'>{title}</h3>", 
                                      unsafe_allow_html=True)
                            st.markdown(f"<div style='color: #666; line-height: 1.6;'>{description}</div>", 
                                      unsafe_allow_html=True)
                        if idx < len(images)-1:
                            st.markdown('<hr style="border: 1px solid #eee; margin: 30px 0;">', 
                                      unsafe_allow_html=True)
                
                # Notes générales
                if general_notes:
                    st.markdown('<hr style="border: 1px solid #eee; margin: 30px 0;">', 
                              unsafe_allow_html=True)
                    st.markdown("<h3 style='color: #34495e;'>Notes Générales</h3>", 
                              unsafe_allow_html=True)
                    st.markdown(f"<div style='color: #666; line-height: 1.6;'>{general_notes}</div>", 
                              unsafe_allow_html=True)
            
            # Boutons d'action
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Créer un nouveau rapport"):
                    st.session_state.preview_ready = False
                    st.rerun()
            with col2:
                pdf_buffer = generate_pdf_with_reportlab(logo, report_id, report_date, images, general_notes, logo_text)
                st.download_button(
                    label="⬇️ Exporter en PDF",
                    data=pdf_buffer,
                    file_name=f"rapport_{report_id}.pdf",
                    mime="application/pdf"
                )

def generate_pdf_with_reportlab(logo, report_id, report_date, images, general_notes, logo_text):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # En-tête
    if logo:
        logo_img = ImageReader(logo)
        c.drawImage(logo_img, 50, height-150, width=100, height=100)
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(170, height-100, f"Rapport : {report_id}")
    c.setFont("Helvetica", 12)
    c.drawString(170, height-120, f"Date : {report_date.strftime('%d %B %Y')}")
    
    if logo_text:
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(50, height-160, logo_text)
    
    # Contenu
    y_position = height - 200
    for img, title, description in images:
        if img:
            img_reader = ImageReader(img)
            c.drawImage(img_reader, 50, y_position-150, width=200, height=150)
            c.setFont("Helvetica-Bold", 14)
            c.drawString(260, y_position-30, title)
            c.setFont("Helvetica", 12)
            text_object = c.beginText(260, y_position-50)
            text_object.textLines(description)
            c.drawText(text_object)
            y_position -= 200
    
    # Notes générales
    if general_notes:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position-50, "Notes Générales :")
        c.setFont("Helvetica", 12)
        text_object = c.beginText(50, y_position-70)
        text_object.textLines(general_notes)
        c.drawText(text_object)
    
    c.save()
    buffer.seek(0)
    return buffer

if __name__ == "__main__":
    create_report()

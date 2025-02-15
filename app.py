import streamlit as st
from datetime import date
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, Spacer

# Configuration de la page
st.set_page_config(page_title="Générateur de Rapport Pro", layout="centered")

def create_report():
    st.title("📝 Générateur de Rapport Professionnel")

    # Réinitialisation de l'état
    if 'preview_ready' not in st.session_state:
        st.session_state.preview_ready = False

    # Section de configuration
    with st.form("config_form"):
        st.header("⚙️ Configuration du Rapport")
        
        # Section Entreprise
        with st.expander("🏢 Informations Entreprise", expanded=True):
            col1, col2 = st.columns([1, 3])
            with col1:
                logo = st.file_uploader("Logo Entreprise (300x300px)", type=["png", "jpg", "jpeg"])
                logo_text = st.text_input("Texte sous le logo", value="Expertise & Qualité")
            
            with col2:
                company_name = st.text_input("Nom de l'entreprise", value="CONSULTECH SARL")
                report_author = st.text_input("Rédacteur du rapport", value="Jean Dupont")
                report_id = st.text_input("ID du rapport", value=f"RAPPORT-{date.today().isoformat()}")
                report_date = st.date_input("Date du rapport", date.today())

        # Section Contenu
        with st.expander("📷 Contenu du Rapport", expanded=True):
            num_images = st.number_input("Nombre d'images", min_value=1, max_value=5, value=1)
            
            images = []
            for i in range(num_images):
                with st.container(border=True):
                    st.markdown(f"#### Élément visuel #{i+1}")
                    img = st.file_uploader(f"Image {i+1}", type=["png", "jpg", "jpeg"], key=f"img{i}")
                    title = st.text_input(f"Titre {i+1}", key=f"title{i}")
                    description = st.text_area(f"Description {i+1}", key=f"desc{i}")
                    images.append((img, title, description))
        
        # Section Notes
        with st.expander("📝 Notes Finales", expanded=True):
            general_notes = st.text_area("Notes générales", height=150, 
                                       placeholder="Saisissez vos observations finales...")

        if st.form_submit_button("👁️ Générer la Prévisualisation"):
            st.session_state.preview_ready = True

    # Affichage du rapport
    if st.session_state.preview_ready:
        st.success("✅ Rapport prêt pour l'export!")
        
        with st.container(border=True):
            # Style CSS amélioré
            st.markdown("""
                <style>
                .report-title {
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 0.5rem;
                }
                .section-title {
                    color: #34495e;
                    margin-top: 1.5rem !important;
                }
                .image-caption {
                    font-style: italic;
                    color: #7f8c8d;
                }
                </style>
            """, unsafe_allow_html=True)

            # En-tête
            header_cols = st.columns([1, 3])
            with header_cols[0]:
                if logo:
                    st.image(logo, use_container_width=True)
                    st.caption(logo_text)

            

            with header_cols[1]:
                st.markdown(f"<h1 class='report-title'>Rapport {report_id}</h1>", unsafe_allow_html=True)
                st.markdown(
                    f"**Entreprise:** {company_name}<br>"
                    f"**Rédacteur:** {report_author}<br>"
                    f"**Date:** {report_date.strftime('%d/%m/%Y')}",
                    unsafe_allow_html=True
                )
    
    
     
    


            st.divider()

            # Contenu principal
            for idx, (img, title, description) in enumerate(images):
                if img:
                    with st.container(border=True):
                        cols = st.columns([1, 2])
                        with cols[0]:
                            st.image(img, use_container_width=True)
                        with cols[1]:
                            st.markdown(f"<h3 class='section-title'>{title}</h3>", unsafe_allow_html=True)
                            st.markdown(f"<div class='image-caption'>{description}</div>", unsafe_allow_html=True)
                
                if idx < len(images)-1:
                    st.divider()

            # Notes générales
            if general_notes:
                with st.container(border=True):
                    st.markdown("#### Notes Générales")
                    st.write(general_notes)

        # Boutons d'action
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Nouveau rapport", use_container_width=True):
                st.session_state.preview_ready = False
                st.rerun()
        with col2:
            pdf_buffer = generate_pdf_with_reportlab(
                logo, company_name, report_author, report_id, 
                report_date, images, general_notes, logo_text
            )
            st.download_button(
                label="⬇️ Exporter PDF",
                data=pdf_buffer,
                file_name=f"rapport_{report_id}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

def generate_pdf_with_reportlab(logo, company, author, report_id, date, images, notes, logo_text):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    styles = getSampleStyleSheet()
    
    # Styles personnalisés
    styles.add(ParagraphStyle(
        name='Justify',
        parent=styles['Normal'],
        alignment=4,
        fontSize=12,
        leading=14
    ))
    
    # En-tête
    header_y = height - 50
    if logo:
        try:
            logo_img = ImageReader(logo)
            c.drawImage(logo_img, 50, header_y - 80, width=80, height=80, preserveAspectRatio=True)
        except:
            pass
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(150, header_y - 30, company)
    c.setFont("Helvetica", 12)
    c.drawString(150, header_y - 50, f"Rédigé par: {author}")
    c.drawString(150, header_y - 70, f"Date: {date.strftime('%d/%m/%Y')}")
    c.drawString(width - 200, header_y - 30, f"ID Rapport: {report_id}")
    
    if logo_text:
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(50, header_y - 100, logo_text)
    
    # Contenu principal
    content_y = header_y - 150
    for img, title, description in images:
        if img:
            try:
                img_reader = ImageReader(img)
                c.drawImage(img_reader, 50, content_y - 150, width=200, height=150, preserveAspectRatio=True)
            except:
                pass
            
            c.setFont("Helvetica-Bold", 14)
            c.drawString(260, content_y - 30, title)
            p = Paragraph(description, styles['Justify'])
            p.wrapOn(c, 300, 100)
            p.drawOn(c, 260, content_y - 50)
            
            content_y -= 200
    
    # Notes générales
    if notes:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, content_y - 50, "Notes Générales:")
        p = Paragraph(notes, styles['Justify'])
        p.wrapOn(c, 500, 100)
        p.drawOn(c, 50, content_y - 80)
    
    c.save()
    buffer.seek(0)
    return buffer

if __name__ == "__main__":
    create_report()

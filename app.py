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
                location = st.text_input("Lieu", value="Paris, France")

        # Section Contenu
        with st.expander("📷 Contenu du Rapport", expanded=True):
            # Image principale
            st.subheader("Image Principale")
            main_img = st.file_uploader("Image principale (largeur complète)", type=["png", "jpg", "jpeg"])
            main_title = st.text_input("Titre principal")
            main_desc = st.text_area("Description principale")
            
            # Images secondaires
            st.subheader("Images Secondaires")
            num_secondary = st.number_input("Nombre d'images secondaires", min_value=0, max_value=9, value=0)
            
            secondary_images = []
            for i in range(num_secondary):
                with st.container(border=True):
                    st.markdown(f"#### Image secondaire #{i+1}")
                    img = st.file_uploader(f"Fichier {i+1}", type=["png", "jpg", "jpeg"], key=f"sec_img{i}")
                    title = st.text_input(f"Titre {i+1}", key=f"sec_title{i}")
                    description = st.text_area(f"Description {i+1}", key=f"sec_desc{i}")
                    secondary_images.append((img, title, description))
        
        # Section Notes
        with st.expander("📝 Notes Finales", expanded=True):
            general_notes = st.text_area("Notes générales", height=150, 
                                       placeholder="Saisissez vos observations finales...")

        submitted = st.form_submit_button("✅ Générer le Rapport")

    # Affichage du rapport après soumission
    if submitted:
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
                .secondary-grid {
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 1rem;
                    margin-top: 2rem;
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
                st.markdown(f"<h1 class='report-title'>{report_id}</h1>", unsafe_allow_html=True)
                st.markdown(
                    f"**Entreprise:** {company_name}<br>"
                    f"**Rédacteur:** {report_author}<br>"
                    f"**Date:** {report_date.strftime('%d/%m/%Y')}<br>"
                    f"**Lieu:** {location}",
                    unsafe_allow_html=True
                )

            st.divider()

            # Image principale
            if main_img:
                with st.container(border=True):
                    st.image(main_img, use_container_width=True)
                    st.markdown(f"<h3 class='section-title'>{main_title}</h3>", unsafe_allow_html=True)
                    st.markdown(f"<div class='image-caption'>{main_desc}</div>", unsafe_allow_html=True)
                    st.divider()

            # Images secondaires
            if secondary_images:
                st.markdown("#### Galerie Secondaire")
                
                # Création des groupes de 3 images
                for i in range(0, len(secondary_images), 3):
                    cols = st.columns(3)
                    group = secondary_images[i:i+3]
                    
                    for col, (img, title, desc) in zip(cols, group):
                        with col:
                            with st.container(border=True):
                                if img:
                                    st.image(img, use_container_width=True)
                                    st.markdown(f"**{title}**")
                                    st.caption(desc)

            # Notes générales
            if general_notes:
                with st.container(border=True):
                    st.markdown("#### Notes Générales")
                    st.write(general_notes)

        # Boutons d'action
        pdf_buffer = generate_pdf_with_reportlab(
            logo, company_name, report_author, report_id, 
            report_date, location, main_img, main_title, main_desc,
            secondary_images, general_notes, logo_text
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Nouveau rapport", use_container_width=True):
                st.rerun()
        with col2:
            st.download_button(
                label="⬇️ Exporter PDF",
                data=pdf_buffer,
                file_name=f"rapport_{report_id}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

def generate_pdf_with_reportlab(logo, company, author, report_id, date, location, 
                               main_img, main_title, main_desc, secondary_images, 
                               notes, logo_text):
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
    c.drawString(150, header_y - 90, f"Lieu: {location}")
    c.drawString(width - 200, header_y - 30, report_id)
    
    if logo_text:
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(50, header_y - 100, logo_text)
    
    # Contenu principal
    content_y = header_y - 150
    
    # Image principale
    if main_img:
        try:
            img_reader = ImageReader(main_img)
            c.drawImage(img_reader, 50, content_y - 200, width=500, height=300, preserveAspectRatio=True)
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, content_y - 220, main_title)
            p = Paragraph(main_desc, styles['Justify'])
            p.wrapOn(c, 500, 100)
            p.drawOn(c, 50, content_y - 250)
            content_y -= 320
        except:
            pass
    
    # Images secondaires
    if secondary_images:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, content_y - 40, "Galerie Secondaire")
        content_y -= 60
        
        x_pos = 50
        for idx, (img, title, desc) in enumerate(secondary_images):
            if idx % 3 == 0 and idx != 0:
                content_y -= 150
                x_pos = 50
                
            try:
                img_reader = ImageReader(img)
                c.drawImage(img_reader, x_pos, content_y - 100, width=150, height=100, preserveAspectRatio=True)
                c.setFont("Helvetica", 10)
                c.drawString(x_pos, content_y - 110, title[:30])
                x_pos += 160
            except:
                pass
    
    # Notes générales
    if notes:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, content_y - 150, "Notes Générales:")
        p = Paragraph(notes, styles['Justify'])
        p.wrapOn(c, 500, 100)
        p.drawOn(c, 50, content_y - 180)
    
    c.save()
    buffer.seek(0)
    return buffer

if __name__ == "__main__":
    create_report()

import streamlit as st
from datetime import date
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph

# Configuration de la page
st.set_page_config(page_title="Générateur de Rapport Pro", layout="centered")

def create_report():
    st.title("📝 Générateur de Rapport Professionnel")

    with st.form("config_form"):
        # Section Entreprise
        with st.expander("🏢 Informations Entreprise", expanded=True):
            cols = st.columns([1, 3])
            with cols[0]:
                logo = st.file_uploader("Logo (300x300px)", type=["png", "jpg", "jpeg"])
            
            with cols[1]:
                company_name = st.text_input("Nom de l'entreprise", value="CONSULTECH SARL")
                report_author = st.text_input("Rédacteur", value="Jean Dupont")
                report_id = st.text_input("ID du rapport", value=f"RAPPORT-{date.today().isoformat()}")
                report_date = st.date_input("Date", date.today())
                location = st.text_input("Lieu", value="Paris, France")

        # Section Contenu
        with st.expander("📷 Contenu du Rapport", expanded=True):
            # Image principale
            st.subheader("Image Principale")
            main_img = st.file_uploader("Image principale", type=["png", "jpg", "jpeg"])
            main_title = st.text_input("Titre principal")
            main_desc = st.text_area("Description principale")
            
            # Images secondaires
            st.subheader("Images Secondaires")
            num_secondary = st.number_input("Nombre d'images secondaires", min_value=0, max_value=9, value=0)
            
            secondary_images = []
            for i in range(num_secondary):
                with st.container(border=True):
                    cols = st.columns([2, 3])
                    with cols[0]:
                        img = st.file_uploader(f"Image {i+1}", type=["png", "jpg", "jpeg"], key=f"sec_img{i}")
                    with cols[1]:
                        title = st.text_input(f"Titre {i+1}", key=f"sec_title{i}")
                        description = st.text_area(f"Description {i+1}", key=f"sec_desc{i}")
                    secondary_images.append((img, title, description))

        # Section Notes
        with st.expander("📝 Notes Finales", expanded=True):
            general_notes = st.text_area("Notes générales", height=150)

        submitted = st.form_submit_button("✅ Générer le Rapport")

    if submitted:
        # Style CSS personnalisé
        st.markdown("""
            <style>
            .header-card {
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                padding: 1.5rem;
                margin-bottom: 2rem;
                background: #f8f9fa;
            }
            .main-image-card {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 1rem;
                margin: 1rem 0;
            }
            .secondary-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 1.5rem;
                margin: 2rem 0;
            }
            .image-card {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 1rem;
                transition: transform 0.2s;
            }
            .image-card:hover {
                transform: translateY(-2px);
            }
            .notes-card {
                border-left: 4px solid #3498db;
                background: #f8fbfe;
                padding: 1rem;
                margin: 2rem 0;
            }
            </style>
        """, unsafe_allow_html=True)

        # En-tête
        with st.container():
            st.markdown(f"<div class='header-card'>", unsafe_allow_html=True)
            
            cols = st.columns([1, 4])
            with cols[0]:
                if logo:
                    st.image(logo, use_container_width=True)
            
            with cols[1]:
                st.markdown(f"""
                    <h1 style='margin-bottom: 0.5rem;'>{report_id}</h1>
                    <div style='color: #6c757d;'>
                        <p style='margin: 0.2rem 0;'><b>{company_name}</b></p>
                        <p style='margin: 0.2rem 0;'>Rédigé par {report_author}</p>
                        <p style='margin: 0.2rem 0;'>{report_date.strftime('%d/%m/%Y')} | {location}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

        # Image principale
        if main_img:
            with st.container():
                st.markdown(f"<div class='main-image-card'>", unsafe_allow_html=True)
                st.markdown(f"**{main_title}**")
                st.image(main_img, use_container_width=True)
                st.markdown(f"<div style='color: #6c757d;'>{main_desc}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        # Images secondaires
        if secondary_images:
            st.markdown("<div class='secondary-grid'>", unsafe_allow_html=True)
            for img, title, desc in secondary_images:
                if img:
                    with st.container():
                        st.markdown("<div class='image-card'>", unsafe_allow_html=True)
                        st.markdown(f"**{title}**")
                        st.image(img, use_container_width=True)
                        st.markdown(f"<div style='color: #6c757d; font-size: 0.9em;'>{desc}</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Notes générales
        if general_notes:
            with st.container():
                st.markdown("<div class='notes-card'>", unsafe_allow_html=True)
                st.markdown("**Notes Générales**")
                st.write(general_notes)
                st.markdown("</div>", unsafe_allow_html=True)

        # Export PDF
        pdf_buffer = generate_pdf(
            logo, company_name, report_author, report_id,
            report_date, location, main_img, main_title, main_desc,
            secondary_images, general_notes
        )
        
        st.download_button(
            label="⬇️ Exporter en PDF",
            data=pdf_buffer,
            file_name=f"rapport_{report_id}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

def generate_pdf(logo, company, author, report_id, date, location, 
                main_img, main_title, main_desc, secondary_images, notes):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    styles = getSampleStyleSheet()
    
    # Styles personnalisés
    styles.add(ParagraphStyle(
        name='Content',
        parent=styles['Normal'],
        fontSize=12,
        leading=14,
        textColor='#4a4a4a'
    ))
    
    # En-tête
    header_y = height - 50
    if logo:
        try:
            logo_img = ImageReader(logo)
            c.drawImage(logo_img, 50, header_y - 60, width=60, height=60, preserveAspectRatio=True)
        except:
            pass
    
    c.setFont("Helvetica-Bold", 18)
    c.drawString(120, header_y - 30, report_id)
    c.setFont("Helvetica", 10)
    c.drawString(120, header_y - 50, company)
    c.drawString(120, header_y - 65, f"Rédigé par {author}")
    c.drawString(120, header_y - 80, f"{date.strftime('%d/%m/%Y')} | {location}")
    
    # Contenu principal
    content_y = header_y - 120
    
    # Image principale
    if main_img:
        try:
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, content_y, main_title)
            img_reader = ImageReader(main_img)
            c.drawImage(img_reader, 50, content_y - 150, width=500, height=250, preserveAspectRatio=True)
            p = Paragraph(main_desc, styles['Content'])
            p.wrapOn(c, 500, 100)
            p.drawOn(c, 50, content_y - 170)
            content_y -= 300
        except:
            pass
    
    # Images secondaires
    if secondary_images:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, content_y - 30, "Galerie Secondaire")
        content_y -= 50
        
        x_pos = 50
        for idx, (img, title, desc) in enumerate(secondary_images):
            if idx % 2 == 0 and idx != 0:
                content_y -= 180
                x_pos = 50
            
            try:
                c.setFont("Helvetica-Bold", 12)
                c.drawString(x_pos, content_y - 10, title[:35])
                img_reader = ImageReader(img)
                c.drawImage(img_reader, x_pos, content_y - 120, width=240, height=160, preserveAspectRatio=True)
                p = Paragraph(desc[:200], styles['Content'])
                p.wrapOn(c, 240, 100)
                p.drawOn(c, x_pos, content_y - 140)
                x_pos += 250
            except:
                pass
    
    # Notes générales
    if notes:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, content_y - 200, "Notes Générales:")
        p = Paragraph(notes, styles['Content'])
        p.wrapOn(c, 500, 200)
        p.drawOn(c, 50, content_y - 220)
    
    c.save()
    buffer.seek(0)
    return buffer

if __name__ == "__main__":
    create_report()

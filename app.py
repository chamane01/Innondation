import streamlit as st
from datetime import date
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

st.set_page_config(page_title="Générateur de Rapport Expert", layout="centered")

def create_report():
    st.title("📋 Rapport Technique")
    
    with st.form("main_form"):
        # Section Entête
        with st.container(border=True):
            col1, col2 = st.columns([1, 3])
            with col1:
                logo = st.file_uploader("Logo", type=["png", "jpg", "jpeg"])
            with col2:
                report_date = st.date_input("Date", date.today())
                lieu = st.text_input("Lieu")
                report_id = st.text_input("N° Rapport", value=f"{date.today().strftime('%Y%m%d')}-001")

        # Section Image Principale
        with st.container(border=True):
            st.subheader("Image Principale")
            main_img = st.file_uploader("Télécharger l'image principale", 
                                      type=["png", "jpg", "jpeg"], 
                                      key="main_image")
            main_title = st.text_input("Titre de l'image principale")
            main_desc = st.text_area("Description de l'image principale", height=100)

        # Section Images Secondaires
        with st.container(border=True):
            st.subheader("Images Secondaires")
            num_secondary = st.number_input("Nombre d'images secondaires", 
                                           min_value=0, 
                                           max_value=9, 
                                           value=3)
            
            secondary_images = []
            for i in range(num_secondary):
                cols = st.columns(2)
                with cols[0]:
                    img = st.file_uploader(f"Image secondaire {i+1}", 
                                          type=["png", "jpg", "jpeg"], 
                                          key=f"sec_img{i}")
                with cols[1]:
                    title = st.text_input(f"Titre {i+1}", key=f"sec_title{i}")
                    desc = st.text_area(f"Description {i+1}", key=f"sec_desc{i}")
                secondary_images.append((img, title, desc))

        # Section Notes
        with st.container(border=True):
            notes = st.text_area("Observations Techniques", height=150)

        if st.form_submit_button("Générer le PDF"):
            generate_pdf(logo, report_id, report_date, lieu, main_img, main_title, main_desc, secondary_images, notes)

def generate_pdf(logo, report_id, date, lieu, main_img, main_title, main_desc, secondary_images, notes):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # Entête PDF
    if logo:
        logo_img = ImageReader(logo)
        c.drawImage(logo_img, 50, height-120, width=80, height=80)
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(150, height-80, f"Rapport {report_id}")
    c.setFont("Helvetica", 12)
    c.drawString(150, height-105, f"Date: {date} - Lieu: {lieu}")
    
    # Image Principale
    y_position = height-200
    if main_img:
        main_img_reader = ImageReader(main_img)
        c.drawImage(main_img_reader, 50, y_position-200, width=500, height=300)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position-220, main_title)
        c.setFont("Helvetica", 12)
        text = c.beginText(50, y_position-240)
        text.textLines(main_desc)
        c.drawText(text)
        y_position -= 320
    
    # Images Secondaires
    if secondary_images:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position-40, "Documents complémentaires:")
        y_position -= 60
        
        for i, (img, title, desc) in enumerate(secondary_images):
            if img and i % 3 == 0:
                x_positions = [50, 200, 350]
                y_position -= 200 if i > 0 else 0
            
            if img and i % 3 < 3:
                img_reader = ImageReader(img)
                c.drawImage(img_reader, x_positions[i%3], y_position-150, width=120, height=100)
                c.setFont("Helvetica", 10)
                c.drawString(x_positions[i%3], y_position-160, title)
    
    # Notes
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 200, "Observations Techniques:")
    c.setFont("Helvetica", 12)
    text = c.beginText(50, 180)
    text.textLines(notes)
    c.drawText(text)
    
    c.save()
    buffer.seek(0)
    
    # Téléchargement automatique
    st.download_button(
        label="⬇️ Télécharger PDF Final",
        data=buffer,
        file_name=f"Rapport_{report_id}.pdf",
        mime="application/pdf"
    )

if __name__ == "__main__":
    create_report()

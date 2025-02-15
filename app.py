import streamlit as st
from datetime import datetime
import base64
from PIL import Image
import io
import pdfkit
from jinja2 import Template

# Configuration de la page
st.set_page_config(page_title="Générateur de Rapport Technique", layout="wide")

# Style CSS personnalisé
CSS = """
<style>
.report-container { max-width: 210mm; min-height: 297mm; margin: auto; padding: 20mm; background: white; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
.header { border-bottom: 2px solid #333; margin-bottom: 20px; }
.footer { border-top: 2px solid #333; margin-top: 20px; padding-top: 10px; }
.section-title { background-color: #f0f0f0; padding: 5px; margin: 15px 0; }
</style>
"""

# Classe pour générer le rapport
class ReportGenerator:
    def __init__(self):
        self.elements = {}
        
    def add_element(self, key, value):
        self.elements[key] = value
        
    def generate_html(self):
        with open('template.html') as f:
            template = Template(f.read())
        return template.render(data=self.elements)

# Initialisation de l'état de session
if 'sections' not in st.session_state:
    st.session_state.sections = []

if 'report' not in st.session_state:
    st.session_state.report = ReportGenerator()

# Sidebar pour les paramètres
with st.sidebar:
    st.header("Paramètres du Rapport")
    date_rapport = st.date_input("Date du rapport")
    heure_rapport = st.time_input("Heure du rapport")
    titre_rapport = st.text_input("Titre principal")
    note_rapport = st.text_area("Note générale")
    logo = st.file_uploader("Logo (PNG/JPG)", type=["png", "jpg"])
    
    # Gestion des sections
    with st.expander("Ajouter une section"):
        new_section_title = st.text_input("Titre de section")
        new_section_content = st.text_area("Contenu de section")
        if st.button("Ajouter Section"):
            st.session_state.sections.append({
                'title': new_section_title,
                'content': new_section_content
            })

# Conteneur principal pour l'aperçu
col1, col2 = st.columns([1, 1])
with col1:
    st.header("Configuration du Rapport")
    
with col2:
    st.header("Aperçu du Rapport en Temps Réel")
    with st.container():
        # Construction du HTML
        html_content = f"""
        {CSS}
        <div class="report-container">
            <div class="header">
                {f'<img src="data:image/png;base64,{base64.b64encode(logo.getvalue()).decode()}" style="height: 50px; float: right;">' if logo else ''}
                <h1>{titre_rapport}</h1>
                <p>Date: {date_rapport} | Heure: {heure_rapport}</p>
            </div>
            
            <div class="content">
                <div class="section-title">Note Générale</div>
                <p>{note_rapport}</p>
        """
        
        for i, section in enumerate(st.session_state.sections, 1):
            html_content += f"""
                <div class="section-title">Plan {i}: {section['title']}</div>
                <p>{section['content']}</p>
            """
        
        html_content += """
            </div>
            <div class="footer">
                <p>Rapport généré le """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """</p>
            </div>
        </div>
        """
        
        # Affichage de l'aperçu
        st.markdown(html_content, unsafe_allow_html=True)
        
        # Options d'exportation
        export_type = st.selectbox("Format d'export", ["PDF", "PNG"])
        if st.button(f"Exporter en {export_type}"):
            if export_type == "PDF":
                pdfkit.from_string(html_content, 'rapport.pdf')
                with open("rapport.pdf", "rb") as f:
                    st.download_button("Télécharger PDF", f.read(), file_name="rapport_technique.pdf")
            else:
                img = Image.open(io.BytesIO(html2image.html2image(html_content)[0]))
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG')
                st.download_button("Télécharger PNG", img_bytes.getvalue(), file_name="rapport.png")

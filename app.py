import streamlit as st
from datetime import date
import base64

def create_report():
    st.title("📊 Générateur de Rapport Interactif")

    # Réinitialisation de l'état
    if 'report_generated' not in st.session_state:
        st.session_state.report_generated = False

    # Section de configuration
    with st.form("report_form"):
        st.header("Paramètres du Rapport")
        
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
        st.subheader("📷 Contenu Visuel")
        num_images = st.number_input("Nombre d'images", min_value=1, max_value=5, value=1)
        
        images = []
        for i in range(num_images):
            with st.expander(f"Image #{i+1}", expanded=True if i == 0 else False):
                img = st.file_uploader(f"Fichier image {i+1}", type=["png", "jpg", "jpeg"], key=f"img{i}")
                title = st.text_input(f"Titre {i+1}", key=f"title{i}")
                description = st.text_area(f"Description {i+1}", key=f"desc{i}")
                images.append((img, title, description))
        
        # Notes générales
        general_notes = st.text_area("📝 Notes Générales", height=150)
        
        submitted = st.form_submit_button("👀 Prévisualiser le Rapport")

    # Affichage du rapport
    if submitted or st.session_state.report_generated:
        st.session_state.report_generated = True
        st.success("✅ Rapport généré avec succès!")
        
        with st.container():
            st.markdown(
                """
                <style>
                .report-container {
                    background-color: white;
                    padding: 2rem;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            
            with st.markdown('<div class="report-container">', unsafe_allow_html=True):
                # En-tête
                header_cols = st.columns([1, 3])
                with header_cols[0]:
                    if logo:
                        st.image(logo, width=200)
                        if logo_text:
                            st.markdown(f"<div style='text-align: center;'><i>{logo_text}</i></div>", unsafe_allow_html=True)
                
                with header_cols[1]:
                    st.markdown(f"<h2 style='margin-top: 0;'>Rapport #{report_id}</h2>", unsafe_allow_html=True)
                    st.markdown(f"**Date :** {report_date.strftime('%d %B %Y')}")
                
                st.markdown("---")
                
                # Contenu principal
                for idx, (img, title, description) in enumerate(images):
                    if img:
                        cols = st.columns([1, 2])
                        with cols[0]:
                            st.image(img, use_column_width=True)
                        with cols[1]:
                            st.subheader(title)
                            st.markdown(description)
                        if idx < len(images)-1:
                            st.markdown("---")
                
                # Notes générales
                if general_notes:
                    st.markdown("---")
                    st.subheader("Notes Générales")
                    st.markdown(general_notes)
            
            # Boutons d'action
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Générer un nouveau rapport"):
                    st.session_state.report_generated = False
                    st.experimental_rerun()
            with col2:
                # Fonction de génération PDF à implémenter
                st.warning("La fonction d'export PDF nécessite l'implémentation de ReportLab")

def generate_pdf():
    # À implémenter avec ReportLab
    pass

if __name__ == "__main__":
    create_report()

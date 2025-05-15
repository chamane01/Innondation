import streamlit as st
import pandas as pd
from fpdf import FPDF
import tempfile
import base64
from PIL import Image
import datetime

# Configuration du PDF
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, st.session_state.republique, 0, 1, 'C')
        self.set_font('Arial', 'B', 12)
        self.cell(0, 6, "Union - Discipline - Travail", 0, 1, 'C')
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_borne_page(pdf, data, images):
    # En-tête ministériel
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 8, st.session_state.ministere, 0, 1, 'C')
    pdf.cell(0, 8, st.session_state.projet, 0, 1, 'C')
    pdf.ln(5)
    
    # Titre section
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 8, f"CADASTRAGE DE LA VILLE DE {st.session_state.commune}", 0, 1, 'C')
    pdf.ln(10)
    
    # Informations de la borne
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 8, f"BORNE GEODESIQUE : {data['Borne']}", 0, 1)
    pdf.cell(0, 8, "FICHE SIGNALETIQUE", 0, 1)
    pdf.ln(5)
    
    # Tableau des coordonnées
    pdf.set_font('Arial', 'B', 10)
    col_width = pdf.w / 4.5
    row_height = 8
    
    # Tableau principal
    headers = ['DESIGNATION', 'COORDONNÉES GÉOGRAPHIQUES', 'COORDONNÉES RECTANGULAIRES', 'ALTITUDE']
    for header in headers:
        pdf.cell(col_width, row_height, header, border=1)
    pdf.ln(row_height)
    
    # Données
    pdf.set_font('Arial', '', 10)
    pdf.cell(col_width, row_height, data['Borne'], border=1)
    pdf.cell(col_width, row_height, f"Lat: {data['Latitude']}\nLon: {data['Longitude']}", border=1)
    pdf.cell(col_width, row_height, f"X: {data['X']}\nY: {data['Y']}\nZ: {data['Z']}", border=1)
    pdf.cell(col_width, row_height, str(data['Altitude']) + ' m', border=1)
    pdf.ln(15)
    
    # Section des vues
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, "VUES", 0, 1)
    
    # Insertion des images
    if images:
        img_width = 80
        for idx, img in enumerate(images):
            pdf.image(img, x=10 + (idx%2)*100, y=pdf.get_y(), w=img_width)
            pdf.ln(img_width + 5) if idx%2 else None
    
    # Description géographique
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, "DESCRIPTION SUCCINCTE DE LA SITUATION GEOGRAPHIQUE", 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 6, f"Borne géodésique située {data['Description']}")
    
    # Tableau administratif
    pdf.ln(5)
    col_width = pdf.w / 3
    headers = ['Région', 'Département', 'Commune']
    pdf.set_font('Arial', 'B', 10)
    for header in headers:
        pdf.cell(col_width, 8, header, border=1)
    pdf.ln(8)
    pdf.set_font('Arial', '', 10)
    for value in [st.session_state.region, st.session_state.departement, st.session_state.commune]:
        pdf.cell(col_width, 8, value, border=1)
    pdf.ln(10)

def main():
    st.set_page_config(page_title="Générateur de Fiches Géodésiques", layout="wide")
    
    st.title("📄 Générateur Automatisé de Fiches Signalétiques")
    st.markdown("---")
    
    # Section de configuration
    with st.sidebar:
        st.header("Configuration Générale")
        st.session_state.republique = st.text_input("République", "République de Côte d'Ivoire")
        st.session_state.ministere = st.text_input("Ministère", "MINISTÈRE DE L'ÉQUIPEMENT ET DE L'ENTRETIEN ROUTIER")
        st.session_state.projet = st.text_input("Nom du projet", "PIDUCAS - Projet d'Intérêt pour le Développement Urbain et la Convivialité des Agglomérations Secondaires")
        
        st.session_state.region = st.text_input("Région", "SAN-PEDRO")
        st.session_state.departement = st.text_input("Département", "SAN-PEDRO")
        st.session_state.commune = st.text_input("Commune", "SAN-PEDRO")
    
    # Téléversement des données
    st.header("1. Importation des données")
    uploaded_file = st.file_uploader("Téléverser le fichier des bornes", type=["csv", "txt", "geojson"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.txt'):
                df = pd.read_csv(uploaded_file, delimiter='\t')
            elif uploaded_file.name.endswith('.geojson'):
                import geopandas as gpd
                gdf = gpd.read_file(uploaded_file)
                df = pd.DataFrame(gdf)
            
            st.success(f"{len(df)} bornes détectées dans le fichier!")
            
            # Aperçu des données
            st.subheader("Aperçu des données")
            st.dataframe(df.head())
            
            # Téléversement des images
            st.header("2. Ajout des photographies")
            uploaded_images = st.file_uploader("Ajouter les photos des bornes (2 par borne)", 
                                            type=["jpg", "png"], 
                                            accept_multiple_files=True)
            
            # Génération du PDF
            if st.button("Générer le catalogue PDF"):
                pdf = PDF()
                pdf.set_auto_page_break(auto=True, margin=15)
                
                progress_bar = st.progress(0)
                total_bornes = len(df)
                
                for idx, row in df.iterrows():
                    pdf.add_page()
                    
                    # Récupération des images
                    borne_images = []
                    if uploaded_images:
                        for img in uploaded_images:
                            if row['Borne'] in img.name:
                                borne_images.append(img.name)
                    
                    # Création de la page
                    create_borne_page(pdf, row, borne_images)
                    progress_bar.progress((idx+1)/total_bornes)
                
                # Sauvegarde temporaire
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    pdf.output(tmp.name)
                    
                    with open(tmp.name, "rb") as f:
                        st.success("Catalogue généré avec succès!")
                        st.download_button(
                            label="📥 Télécharger le PDF",
                            data=f,
                            file_name=f"Catalogue_Fiches_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/octet-stream"
                        )
        except Exception as e:
            st.error(f"Erreur lors du traitement des données: {str(e)}")

if __name__ == "__main__":
    main()

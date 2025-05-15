import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import zipfile
import tempfile
import os
import textwrap
import datetime

# Configuration des polices
FONT_PATH = "arial.ttf"
HEADER_FONT = ImageFont.truetype(FONT_PATH, 40)
TITLE_FONT = ImageFont.truetype(FONT_PATH, 32)
SECTION_FONT = ImageFont.truetype(FONT_PATH, 28)
TABLE_HEADER_FONT = ImageFont.truetype(FONT_PATH, 24)
TABLE_CONTENT_FONT = ImageFont.truetype(FONT_PATH, 22)
DESCRIPTION_FONT = ImageFont.truetype(FONT_PATH, 24)

# Dimensions de l'image
PAGE_WIDTH = 2480  # Format A4 en pixels (300 dpi)
PAGE_HEIGHT = 3508
BACKGROUND_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)

def draw_centered_text(draw, y_position, text, font, image_width):
    text_width = draw.textlength(text, font=font)
    x_position = (image_width - text_width) // 2
    draw.text((x_position, y_position), text, fill=TEXT_COLOR, font=font)
    return y_position + font.size + 20

def create_borne_image(data, images_paths):
    img = Image.new('RGB', (PAGE_WIDTH, PAGE_HEIGHT), color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    
    current_y = 100
    
    # En-tête République
    current_y = draw_centered_text(draw, current_y, st.session_state.republique, HEADER_FONT, PAGE_WIDTH)
    draw_centered_text(draw, current_y, "Union - Discipline - Travail", SECTION_FONT, PAGE_WIDTH)
    current_y += 80
    
    # Ministère et Projet
    draw_centered_text(draw, current_y, st.session_state.ministere, TITLE_FONT, PAGE_WIDTH)
    current_y = draw_centered_text(draw, current_y, st.session_state.projet, SECTION_FONT, PAGE_WIDTH)
    current_y += 100
    
    # Titre Cadastrage
    draw_centered_text(draw, current_y, f"CADASTRAGE DE LA VILLE DE {st.session_state.commune}", TITLE_FONT, PAGE_WIDTH)
    current_y += 120
    
    # Section Borne
    draw.text((200, current_y), f"BORNE GEODESIQUE : {data['Borne']}", fill=TEXT_COLOR, font=SECTION_FONT)
    current_y += 80
    draw.text((200, current_y), "FICHE SIGNALETIQUE", fill=TEXT_COLOR, font=SECTION_FONT)
    current_y += 150
    
    # Tableau des coordonnées
    table_x = 200
    col_width = 500
    
    # En-têtes du tableau
    headers = ["DESIGNATION", "COORDONNÉES GÉOGRAPHIQUES", "COORDONNÉES RECTANGULAIRES", "ALTITUDE"]
    for i, header in enumerate(headers):
        draw.rectangle([(table_x + i*col_width, current_y), (table_x + (i+1)*col_width, current_y + 80)], outline=TEXT_COLOR)
        draw.text((table_x + i*col_width + 20, current_y + 20), header, fill=TEXT_COLOR, font=TABLE_HEADER_FONT)
    
    current_y += 80
    
    # Contenu du tableau
    content = [
        data['Borne'],
        f"Latitude: {data['Latitude']}\nLongitude: {data['Longitude']}",
        f"X: {data['X']}\nY: {data['Y']}\nZ: {data.get('Z', '')}",
        f"{data['Altitude']} m"
    ]
    
    for i, item in enumerate(content):
        for j, line in enumerate(str(item).split('\n')):
            draw.text((table_x + i*col_width + 20, current_y + 20 + j*30), line, fill=TEXT_COLOR, font=TABLE_CONTENT_FONT)
    
    current_y += 200
    
    # Section Vues
    if images_paths:
        draw.text((200, current_y), "VUES", fill=TEXT_COLOR, font=SECTION_FONT)
        current_y += 80
        
        photo_size = 800
        x_positions = [300, 1500]
        for idx, img_path in enumerate(images_paths[:2]):
            try:
                photo = Image.open(img_path)
                photo.thumbnail((photo_size, photo_size))
                img.paste(photo, (x_positions[idx], current_y))
            except Exception as e:
                draw.text((x_positions[idx], current_y), "Image non disponible", fill=TEXT_COLOR, font=TABLE_CONTENT_FONT)
        
        current_y += photo_size + 100
    
    # Description géographique
    draw.text((200, current_y), "DESCRIPTION SUCCINCTE DE LA SITUATION GEOGRAPHIQUE", fill=TEXT_COLOR, font=SECTION_FONT)
    current_y += 80
    description = textwrap.fill(data['Description'], width=100)
    draw.text((200, current_y), description, fill=TEXT_COLOR, font=DESCRIPTION_FONT)
    current_y += 200
    
    # Tableau administratif
    admin_x = 200
    col_width = (PAGE_WIDTH - 400) // 3
    headers = ["Région", "Département", "Commune"]
    values = [st.session_state.region, st.session_state.departement, st.session_state.commune]
    
    for i in range(3):
        draw.rectangle([(admin_x + i*col_width, current_y), (admin_x + (i+1)*col_width, current_y + 80)], outline=TEXT_COLOR)
        draw.text((admin_x + i*col_width + 20, current_y + 20), headers[i], fill=TEXT_COLOR, font=TABLE_HEADER_FONT)
        draw.text((admin_x + i*col_width + 20, current_y + 50), values[i], fill=TEXT_COLOR, font=TABLE_CONTENT_FONT)
    
    return img

def main():
    st.set_page_config(page_title="Générateur d'Images Géodésiques", layout="wide")
    
    st.title("🖼️ Générateur Automatisé de Fiches en Image")
    st.markdown("---")
    
    with st.sidebar:
        st.header("Configuration Générale")
        st.session_state.republique = st.text_input("République", "République de Côte d'Ivoire")
        st.session_state.ministere = st.text_input("Ministère", "MINISTÈRE DE L'ÉQUIPEMENT ET DE L'ENTRETIEN ROUTIER")
        st.session_state.projet = st.text_input("Nom du projet", "PIDUCAS - Projet d'Intérêt pour le Développement Urbain et la Convivialité des Agglomérations Secondaires")
        
        st.session_state.region = st.text_input("Région", "SAN-PEDRO")
        st.session_state.departement = st.text_input("Département", "SAN-PEDRO")
        st.session_state.commune = st.text_input("Commune", "SAN-PEDRO")
    
    st.header("1. Importation des données")
    uploaded_file = st.file_uploader("Téléverser le fichier des bornes", type=["csv", "txt"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_csv(uploaded_file, delimiter='\t')
            required_columns = ['Borne', 'Latitude', 'Longitude', 'X', 'Y', 'Altitude', 'Description']
            
            if not all(col in df.columns for col in required_columns):
                st.error("Colonnes manquantes dans le fichier!")
                return
            
            st.success(f"{len(df)} bornes détectées!")
            st.dataframe(df.head())
            
            st.header("2. Téléversement des photos")
            uploaded_images = st.file_uploader("Ajouter les photos (format: NOMBORNE_1.jpg)", 
                                            type=["jpg", "png"], 
                                            accept_multiple_files=True)
            
            if st.button("Générer les fiches en images"):
                temp_dir = tempfile.mkdtemp()
                image_paths = []
                
                # Sauvegarde des images uploadées
                saved_images = {}
                for img in uploaded_images:
                    save_path = os.path.join(temp_dir, img.name)
                    with open(save_path, "wb") as f:
                        f.write(img.getbuffer())
                    saved_images[img.name] = save_path
                
                # Génération des images
                progress_bar = st.progress(0)
                for idx, row in df.iterrows():
                    # Recherche des images correspondantes
                    borne_images = []
                    for img_name in saved_images:
                        if row['Borne'] in img_name:
                            borne_images.append(saved_images[img_name])
                    
                    # Création de l'image
                    img = create_borne_image(row, borne_images)
                    img_path = os.path.join(temp_dir, f"Fiche_{row['Borne']}.jpg")
                    img.save(img_path)
                    image_paths.append(img_path)
                    progress_bar.progress((idx+1)/len(df))
                
                # Création du ZIP
                zip_path = os.path.join(temp_dir, "Fiches_Geodesiques.zip")
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for img_path in image_paths:
                        zipf.write(img_path, os.path.basename(img_path))
                
                # Téléchargement
                with open(zip_path, "rb") as f:
                    st.success("Génération terminée!")
                    st.download_button(
                        label="📥 Télécharger toutes les fiches (ZIP)",
                        data=f,
                        file_name=f"Fiches_Geodesiques_{datetime.datetime.now().strftime('%Y%m%d')}.zip",
                        mime="application/zip"
                    )
                
        except Exception as e:
            st.error(f"Erreur: {str(e)}")

if __name__ == "__main__":
    main()

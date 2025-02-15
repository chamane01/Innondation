import streamlit as st
from datetime import datetime
import pandas as pd
import random
from PIL import Image, ImageDraw

# Titre principal
st.title("Rapport de Topographie")

# Informations générales
st.header("Informations Générales")
col1, col2 = st.columns(2)
with col1:
    date_rapport = st.date_input("Date du rapport", datetime.today())
    heure_rapport = st.time_input("Heure du rapport", datetime.now().time())
    numero_rapport = st.text_input("Numéro du rapport", "RPT-001")
with col2:
    lieu = st.text_input("Lieu du relevé", "Site Alpha")
    titre_rapport = st.text_input("Titre du rapport", "Étude topographique du terrain X")

# Section Cartes
st.header("Cartes et Plans")
st.write("Ajoutez vos cartes en les téléchargeant ci-dessous.")
uploaded_maps = st.file_uploader("Téléchargez une carte", type=["jpg", "png", "pdf"], accept_multiple_files=True)

# Section Profils
st.header("Profils Topographiques")
st.write("Ajoutez vos profils topographiques.")
uploaded_profiles = st.file_uploader("Téléchargez un profil topographique", type=["jpg", "png", "pdf"], accept_multiple_files=True)

# Section Légendes
st.header("Légendes et Explications")
legendes = st.text_area("Ajoutez vos légendes ici", "Description des symboles et codes utilisés...")

# Informations techniques fictives pour test
st.header("Informations Techniques")
nb_points = st.number_input("Nombre de points relevés", min_value=1, value=random.randint(10, 100))
precision = st.slider("Précision des relevés (cm)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
altitude_moyenne = st.number_input("Altitude moyenne (m)", min_value=0.0, value=random.uniform(50, 300))

# Tableau de données fictives
data = {
    "Point": [f"PT-{i+1}" for i in range(nb_points)],
    "Latitude": [round(random.uniform(-90, 90), 6) for _ in range(nb_points)],
    "Longitude": [round(random.uniform(-180, 180), 6) for _ in range(nb_points)],
    "Altitude (m)": [round(random.uniform(altitude_moyenne - 10, altitude_moyenne + 10), 2) for _ in range(nb_points)],
}
df = pd.DataFrame(data)

# Génération d'un aperçu visuel du rapport
st.header("Aperçu Visuel du Rapport")
canvas_width, canvas_height = 800, 600
image = Image.new("RGB", (canvas_width, canvas_height), "white")
draw = ImageDraw.Draw(image)
draw.text((50, 50), f"Rapport de Topographie - {titre_rapport}", fill="black")
draw.text((50, 100), f"Date: {date_rapport}", fill="black")
draw.text((50, 150), f"Heure: {heure_rapport}", fill="black")
draw.text((50, 200), f"Lieu: {lieu}", fill="black")
draw.text((50, 250), f"Numéro du rapport: {numero_rapport}", fill="black")
draw.text((50, 300), f"Légendes: {legendes}", fill="black")

st.image(image, caption="Aperçu du rapport", use_column_width=True)

# Bouton d'export
st.header("Export du rapport")
if st.button("Générer le rapport PDF"):
    st.success("Le rapport a été généré avec succès !")

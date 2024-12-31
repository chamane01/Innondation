import streamlit as st
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import time
import io

def create_serpent(image, frame, radius=50, length=0, num_segments=50, glow_color=(255, 255, 255)):
    """ Crée un serpent lumineux en forme de ligne qui s'enroule autour du logo """
    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    center_x, center_y = width // 2, height // 2

    # Calcul des coordonnées des segments du serpent
    angle_step = 2 * np.pi / num_segments
    segments = []
    for i in range(num_segments):
        angle = angle_step * i + (frame * 0.05)  # Animation par rotation
        x = center_x + int(np.cos(angle) * (radius + length))  # Calcul de la position x
        y = center_y + int(np.sin(angle) * (radius + length))  # Calcul de la position y
        segments.append((x, y))

    # Dessiner les segments du serpent (ligne continue)
    for i in range(1, len(segments)):
        draw.line([segments[i - 1], segments[i]], fill=glow_color, width=2)

    # Créer un effet de scintillement lorsque le serpent mord sa queue
    if length >= 100:  # Scintillement lorsque le serpent est fermé
        for i in range(len(segments)):
            x, y = segments[i]
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=glow_color, outline=None)

    return image

def main():
    st.title("Effet Lumineux : Serpent autour du logo")

    uploaded_file = st.file_uploader("Choisissez une image PNG", type=["png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Paramètres du serpent
        radius = st.slider("Rayon du serpent", min_value=30, max_value=150, value=50)
        length = st.slider("Longueur du serpent", min_value=0, max_value=150, value=0)
        glow_color = st.color_picker("Choisissez la couleur du serpent lumineux", "#FFFFFF")
        glow_color_rgb = tuple(int(glow_color[i:i+2], 16) for i in (1, 3, 5))

        placeholder = st.empty()
        frame = 0
        frames = []  # Pour stocker les frames

        # Animation dynamique
        while True:
            # Créer le serpent et l'appliquer sur l'image
            animated_image = create_serpent(image.copy(), frame, radius=radius, length=length, glow_color=glow_color_rgb)

            # Convertir chaque image en bytes et les ajouter à la liste des frames
            with io.BytesIO() as img_byte_array:
                animated_image.save(img_byte_array, format='PNG')
                frames.append(img_byte_array.getvalue())

            placeholder.image(animated_image, caption="Serpent lumineux en animation", use_container_width=True)
            frame += 1
            length += 1  # Augmenter la longueur du serpent pour l'animer
            time.sleep(0.05)  # Ajuster la vitesse pour fluidité

            # Arrêter l'animation après un certain temps pour la démonstration
            if frame > 100:  # 100 frames comme exemple
                break

        # Créer un fichier GIF à partir des frames générées
        if frames:
            gif_buffer = io.BytesIO()
            # Créer un fichier GIF à partir des frames (frames doit être en format PNG)
            first_frame = Image.open(io.BytesIO(frames[0]))
            first_frame.save(gif_buffer, format='GIF', append_images=[Image.open(io.BytesIO(frame)) for frame in frames[1:]],
                             save_all=True, duration=100, loop=0)
            gif_buffer.seek(0)

            # Ajouter le bouton de téléchargement
            st.download_button(
                label="Télécharger l'animation",
                data=gif_buffer,
                file_name="serpent_lumineux.gif",
                mime="image/gif"
            )

if __name__ == "__main__":
    main()

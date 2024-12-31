import streamlit as st
from PIL import Image, ImageFilter, ImageDraw
import time
import io

def add_dynamic_glow(image, frame, intensity=5, glow_color=(255, 255, 255)):
    """ Effet de lueur dynamique """
    image = image.convert("RGBA")
    glow_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(glow_image)
    offset = (frame % 20) * 2
    for i in range(intensity):
        alpha = image.split()[-1].filter(ImageFilter.GaussianBlur(2))
        draw.bitmap((offset, offset), alpha, fill=glow_color + (50,))
    result = Image.alpha_composite(glow_image, image)
    return result

def add_glorious_glow(image, frame, intensity=5, glow_color=(255, 255, 255)):
    """ Effet de lueur glorieuse """
    image = image.convert("RGBA")
    glow_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(glow_image)
    offset = (frame % 20) * 2
    blur_radius = (frame % 5) + 2
    alpha = image.split()[-1]
    for i in range(intensity):
        alpha = alpha.filter(ImageFilter.GaussianBlur(blur_radius))
        draw.bitmap((offset, offset), alpha, fill=glow_color + (int(255 * (i / intensity)),))
    result = Image.alpha_composite(glow_image, image)
    return result

def add_pulsating_glow(image, frame, intensity=5, glow_color=(255, 255, 255)):
    """ Effet de lueur pulsante """
    image = image.convert("RGBA")
    glow_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(glow_image)
    pulse_size = int(10 + (frame % 10) * 2)
    for i in range(intensity):
        alpha = image.split()[-1].filter(ImageFilter.GaussianBlur(2))
        draw.bitmap((pulse_size, pulse_size), alpha, fill=glow_color + (50,))
    result = Image.alpha_composite(glow_image, image)
    return result

def add_radiant_glow(image, frame, intensity=5, glow_color=(255, 255, 255)):
    """ Effet de lueur rayonnante """
    image = image.convert("RGBA")
    glow_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(glow_image)
    radius = int(10 + (frame % 20) * 3)  # Varied radial glow
    for i in range(intensity):
        alpha = image.split()[-1].filter(ImageFilter.GaussianBlur(2))
        draw.bitmap((radius, radius), alpha, fill=glow_color + (int(255 * (i / intensity)),))
    result = Image.alpha_composite(glow_image, image)
    return result

def main():
    st.title("Effet Lumineux sur une Image PNG")

    uploaded_file = st.file_uploader("Choisissez une image PNG", type=["png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Effet lumineux à appliquer
        effect_style = st.selectbox("Choisissez un effet lumineux", 
                                    ["Lueur dynamique", "Lueur glorieuse", "Lueur pulsante", "Lueur rayonnante"])

        # Paramètres des effets
        intensity = st.slider("Intensité de l'effet lumineux", min_value=1, max_value=10, value=5)
        glow_color = st.color_picker("Choisissez la couleur de l'effet lumineux", "#FFFFFF")
        glow_color_rgb = tuple(int(glow_color[i:i+2], 16) for i in (1, 3, 5))

        # Vitesse de l'animation
        speed = st.slider("Vitesse de l'animation", min_value=1, max_value=10, value=5)

        placeholder = st.empty()
        frame = 0
        frames = []  # Pour stocker les frames

        # Animation dynamique
        while True:
            # Appliquer l'effet choisi
            if effect_style == "Lueur dynamique":
                animated_image = add_dynamic_glow(image, frame, intensity=intensity, glow_color=glow_color_rgb)
            elif effect_style == "Lueur glorieuse":
                animated_image = add_glorious_glow(image, frame, intensity=intensity, glow_color=glow_color_rgb)
            elif effect_style == "Lueur pulsante":
                animated_image = add_pulsating_glow(image, frame, intensity=intensity, glow_color=glow_color_rgb)
            elif effect_style == "Lueur rayonnante":
                animated_image = add_radiant_glow(image, frame, intensity=intensity, glow_color=glow_color_rgb)

            # Convertir chaque image en bytes et les ajouter à la liste des frames
            with io.BytesIO() as img_byte_array:
                animated_image.save(img_byte_array, format='PNG')
                frames.append(img_byte_array.getvalue())

            placeholder.image(animated_image, caption="Image avec effet lumineux", use_container_width=True)
            frame += 1
            time.sleep(1 / speed)

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
                file_name="animation_lumineuse.gif",
                mime="image/gif"
            )

if __name__ == "__main__":
    main()

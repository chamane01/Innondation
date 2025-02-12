import streamlit as st
import os
import rasterio
import rasterio.merge
import folium
import math
import matplotlib.pyplot as plt
import numpy as np
from streamlit_folium import st_folium
from folium.plugins import Draw

#########################
# Fonctions utilitaires #
#########################

def load_tiff_files(folder_path):
    """Charge les fichiers TIFF contenus dans un dossier."""
    try:
        tiff_files = [os.path.join(folder_path, f) 
                      for f in os.listdir(folder_path) if f.lower().endswith('.tif')]
    except Exception as e:
        st.error(f"Erreur lors de la lecture du dossier {folder_path}: {e}")
        return []
    
    if not tiff_files:
        st.error("Aucun fichier TIFF trouvé dans le dossier.")
        return []
    
    return [f for f in tiff_files if os.path.exists(f)]

def build_mosaic(tiff_files, mosaic_path="mosaic.tif"):
    """Construit une mosaïque à partir de fichiers TIFF."""
    try:
        src_files = [rasterio.open(fp) for fp in tiff_files]
        mosaic, out_trans = rasterio.merge.merge(src_files)
        out_meta = src_files[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans
        })
        with rasterio.open(mosaic_path, "w", **out_meta) as dest:
            dest.write(mosaic)
        for src in src_files:
            src.close()
        return mosaic_path
    except Exception as e:
        st.error(f"Erreur lors de la création de la mosaïque : {e}")
        return None

def create_map(mosaic_file):
    """Crée une carte Folium avec l'outil de dessin."""
    m = folium.Map(location=[0, 0], zoom_start=2)

    try:
        with rasterio.open(mosaic_file) as src:
            bounds = src.bounds
            folium.Rectangle(
                bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                color='blue', 
                fill=False,
                tooltip="Emprise de la mosaïque"
            ).add_to(m)
    except Exception as e:
        st.error(f"Erreur lors de l'ouverture de la mosaïque : {e}")

    Draw(
        draw_options={'polyline': True, 'polygon': False, 'circle': False, 
                      'marker': False, 'rectangle': False, 'circlemarker': False},
        edit_options={'edit': True, 'remove': True}
    ).add_to(m)
    
    folium.LayerControl().add_to(m)
    return m

def haversine(lon1, lat1, lon2, lat2):
    """Calcule la distance en mètres entre deux points GPS."""
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371000  # Rayon de la Terre en mètres
    return c * r

##################
# Fonction main  #
##################

def main():
    st.title("Carte Dynamique avec Profils d'Élévation")

    if "mode" not in st.session_state:
        st.session_state.mode = "none"

    map_name = st.text_input("Nom de votre carte", value="Ma Carte")

    folder_path = "TIFF"
    if not os.path.exists(folder_path):
        st.error("Dossier TIFF introuvable")
        return

    tiff_files = load_tiff_files(folder_path)
    if not tiff_files:
        return

    mosaic_path = build_mosaic(tiff_files)
    if not mosaic_path:
        return

    m = create_map(mosaic_path)
    st.write("**Utilisez les outils de dessin sur la carte ci-dessus.**")
    map_data = st_folium(m, width=700, height=500)

    # Menu principal
    if st.session_state.mode == "none":
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Tracer des profils"):
                st.session_state.mode = "profiles"
        with col2:
            if st.button("Générer des contours"):
                st.session_state.mode = "contours"

    # Mode Générer des contours
    if st.session_state.mode == "contours":
        st.subheader("Générer des contours")
        st.write("En cours de développement")
        if st.button("Retour"):
            st.session_state.mode = "none"

    # Mode Tracer des profils
    if st.session_state.mode == "profiles":
        st.subheader("Tracer des profils")

        # Vérification de l'existence des dessins
        raw_drawings = map_data.get("all_drawings", []) if isinstance(map_data, dict) else []
        current_drawings = [d for d in raw_drawings if isinstance(d, dict) and d.get("geometry", {}).get("type") == "LineString"]

        profiles = []
        for i, d in enumerate(current_drawings):
            profiles.append({
                "coords": d["geometry"]["coordinates"],
                "name": f"{map_name} - Profil {i+1}"
            })

        if profiles:
            for i, profile in enumerate(profiles):
                col_a, col_b = st.columns([1, 4])
                with col_a:
                    new_name = st.text_input("Nom du profil", value=profile["name"], key=f"profile_name_{i}")
                    profile["name"] = new_name
                with col_b:
                    try:
                        # Interpolation des points pour régulariser les échantillons
                        points = profile["coords"]
                        distances = [haversine(points[i][0], points[i][1], points[i+1][0], points[i+1][1]) for i in range(len(points)-1)]
                        distances.insert(0, 0)
                        distances = np.cumsum(distances)

                        with rasterio.open(mosaic_path) as src:
                            elevations = [list(src.sample([p]))[0][0] for p in points]

                        fig, ax = plt.subplots(figsize=(8, 3))
                        ax.plot(distances, elevations, 'b-', linewidth=1.5)
                        ax.set_title(profile["name"])
                        ax.set_xlabel("Distance (m)")
                        ax.set_ylabel("Altitude (m)")
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Erreur de traitement : {e}")
        else:
            st.info("Aucun profil dessiné pour le moment.")

        if st.button("Retour"):
            st.session_state.mode = "none"

if __name__ == "__main__":
    main()

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
    """Crée une carte Folium avec l'outil de dessin (uniquement pour les polylignes)
    et intègre un calque indiquant l'emprise de la mosaïque.
    """
    m = folium.Map(location=[0, 0], zoom_start=2)
    
    # Création d'un groupe de couches pour la mosaïque
    mosaic_group = folium.FeatureGroup(name="élevation cote d'ivoire")
    try:
        with rasterio.open(mosaic_file) as src:
            bounds = src.bounds
            folium.Rectangle(
                bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                color='blue', 
                fill=False,
                tooltip="Emprise de la mosaïque"
            ).add_to(mosaic_group)
    except Exception as e:
        st.error(f"Erreur lors de l'ouverture de la mosaïque : {e}")
    
    mosaic_group.add_to(m)
    
    # Ajout de l'outil de dessin (seulement pour les polylignes)
    Draw(
        draw_options={
            'polyline': {'allowIntersection': False},
            'polygon': False,
            'circle': False,
            'marker': False,
            'circlemarker': False,
            'rectangle': False
        },
        edit_options={'edit': True, 'remove': True}
    ).add_to(m)
    
    # Ajout du gestionnaire de couches
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

def interpolate_line(coords, step=50):
    """Interpole des points sur une ligne."""
    if len(coords) < 2:
        return coords, [0]
    sampled_points = [coords[0]]
    cumulative_dist = [0]
    for i in range(len(coords)-1):
        start = coords[i]
        end = coords[i+1]
        seg_distance = haversine(start[0], start[1], end[0], end[1])
        num_steps = max(int(seg_distance // step), 1)
        for j in range(1, num_steps+1):
            fraction = j / num_steps
            lon = start[0] + fraction * (end[0] - start[0])
            lat = start[1] + fraction * (end[1] - start[1])
            dist = haversine(sampled_points[-1][0], sampled_points[-1][1], lon, lat)
            sampled_points.append([lon, lat])
            cumulative_dist.append(cumulative_dist[-1] + dist)
    return sampled_points, cumulative_dist

def generate_contours(mosaic_file):
    """Génère et affiche un graphique de contours à partir de la mosaïque."""
    try:
        with rasterio.open(mosaic_file) as src:
            elevation = src.read(1)
            bounds = src.bounds
            width = src.width
            height = src.height
        # Création des axes X et Y à partir des limites géographiques
        xs = np.linspace(bounds.left, bounds.right, width)
        ys = np.linspace(bounds.bottom, bounds.top, height)
        X, Y = np.meshgrid(xs, ys)
        
        fig, ax = plt.subplots(figsize=(8,6))
        contour = ax.contour(X, Y, elevation, levels=10, cmap='viridis')
        ax.clabel(contour, inline=True, fontsize=8)
        ax.set_title("Contours d'altitude")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Erreur lors de la génération des contours : {e}")

def main():
    st.title("Carte Dynamique avec Profils d'Élévation")
    
    # Initialisation de la session pour stocker les profils (s'il n'existe pas déjà)
    if "profiles" not in st.session_state:
        st.session_state.profiles = []
    
    # Saisie du nom de la carte
    map_name = st.text_input("Nom de votre carte", value="Ma Carte")
    
    # Gestion des fichiers TIFF
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

    # Création et affichage de la carte interactive
    m = create_map(mosaic_path)
    st.write("**Utilisez les outils de dessin sur la carte ci-dessus.**")
    map_data = st_folium(m, width=700, height=500)
    
    # Affichage de deux boutons d'action sous la carte
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Tracer des profils"):
            # Extraction des polylignes dessinées sur la carte
            if isinstance(map_data, dict):
                raw_drawings = map_data.get("all_drawings", [])
                current_drawings = [
                    d for d in raw_drawings 
                    if isinstance(d, dict) and d.get("geometry", {}).get("type") == "LineString"
                ]
                if not current_drawings:
                    st.warning("Aucun profil dessiné trouvé.")
                else:
                    st.session_state.profiles = [{
                        "coords": d["geometry"]["coordinates"],
                        "name": f"{map_name} - Ligne {i+1}"
                    } for i, d in enumerate(current_drawings)]
                    st.success("Profils enregistrés. Les graphiques sont générés ci-dessous.")
            else:
                st.warning("Données de dessin indisponibles.")
        
        # Affichage des profils enregistrés
        if st.session_state.profiles:
            st.subheader("Profils générés")
            for i, profile in enumerate(st.session_state.profiles):
                col_a, col_b = st.columns([1, 4])
                with col_a:
                    new_name = st.text_input(
                        "Nom du profil", 
                        value=profile["name"],
                        key=f"profile_name_{i}"
                    )
                    profile["name"] = new_name
                with col_b:
                    try:
                        points, distances = interpolate_line(profile["coords"])
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
    
    with col2:
        if st.button("Générer des contours"):
            generate_contours(mosaic_path)

if __name__ == "__main__":
    main()

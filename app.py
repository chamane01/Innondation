import streamlit as st
import os
import rasterio
import rasterio.merge
import folium
import math
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
from folium.plugins import Draw

#######################
# Fonctions utilitaires
#######################

def load_tiff_files(folder_path):
    """
    Charge les fichiers TIFF contenus dans un dossier.
    Retourne la liste des chemins et, en parallèle, une liste d'infos (nom et bornes).
    """
    tiff_files = []
    tiff_info = []
    try:
        for f in os.listdir(folder_path):
            if f.lower().endswith('.tif'):
                fp = os.path.join(folder_path, f)
                if os.path.exists(fp):
                    tiff_files.append(fp)
                    # Ouvrir le TIFF pour récupérer ses bornes
                    try:
                        with rasterio.open(fp) as src:
                            bounds = src.bounds
                        tiff_info.append({
                            "file": f,
                            "bounds": bounds
                        })
                    except Exception as e:
                        st.error(f"Erreur lors de l'ouverture de {fp}: {e}")
                else:
                    st.error(f"Le fichier {fp} n'existe pas.")
    except Exception as e:
        st.error(f"Erreur lors de la lecture du dossier {folder_path}: {e}")
    
    if not tiff_files:
        st.error("Aucun fichier TIFF trouvé dans le dossier.")
    return tiff_files, tiff_info

def build_mosaic(tiff_files, mosaic_path="mosaic.tif"):
    """
    Construit une mosaïque à partir d'une liste de fichiers TIFF en utilisant rasterio.merge.
    La mosaïque est sauvegardée sous forme de fichier GeoTIFF.
    """
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
    """
    Crée une carte Folium avec :
      - Un rectangle indiquant l'étendue de la mosaïque.
      - L'outil de dessin (polyline) avec édition et suppression activées.
    """
    m = folium.Map(location=[0, 0], zoom_start=2)
    try:
        with rasterio.open(mosaic_file) as src:
            bounds = src.bounds
            folium.Rectangle(
                bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                color='blue', fill=True, fill_opacity=0.4, tooltip="Mosaïque"
            ).add_to(m)
    except Exception as e:
        st.error(f"Erreur lors de l'ouverture de la mosaïque {mosaic_file}: {e}")
    
    draw = Draw(
        draw_options={
            'polyline': True,   # Outil polyline
            'polygon': False,
            'circle': False,
            'marker': False,
            'circlemarker': False,
            'rectangle': False,
        },
        edit_options={'edit': True, 'remove': True}
    )
    draw.add_to(m)
    return m

def haversine(lon1, lat1, lon2, lat2):
    """
    Calcule la distance (en mètres) entre deux points (lon, lat) via la formule de Haversine.
    """
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371000  # Rayon de la Terre en mètres
    return c * r

def interpolate_line(coords, step=50):
    """
    Interpole les points le long d'une ligne pour obtenir un échantillonnage régulier.
    Retourne la liste des points et la distance cumulative.
    """
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

def get_map_name_for_profile(coords, tiff_info):
    """
    Calcule le point médian de la polyline et recherche, parmi les TIFF d'origine,
    celui dont les bornes contiennent ce point. Retourne le nom du fichier ou "Mosaïque".
    """
    if not coords:
        return "Inconnu"
    lons = [pt[0] for pt in coords]
    lats = [pt[1] for pt in coords]
    mid_lon = sum(lons) / len(lons)
    mid_lat = sum(lats) / len(lats)
    for info in tiff_info:
        b = info["bounds"]
        if b.left <= mid_lon <= b.right and b.bottom <= mid_lat <= b.top:
            return info["file"]
    return "Mosaïque"

#######################
# Fonction principale
#######################

def main():
    st.title("Carte Dynamique avec Profils d'Élévation")
    folder_path = "TIFF"  # Modifiez si nécessaire

    if not os.path.exists(folder_path):
        st.error("Le dossier TIFF n'existe pas.")
        return

    # Charger les TIFF et récupérer aussi leurs infos (nom & bornes)
    tiff_files, tiff_info = load_tiff_files(folder_path)
    if not tiff_files:
        st.error("Aucun fichier TIFF trouvé.")
        return

    # Créer la mosaïque à partir de tous les TIFF
    mosaic_path = build_mosaic(tiff_files, mosaic_path="mosaic.tif")
    if mosaic_path is None:
        st.error("Erreur lors de la création de la mosaïque.")
        return

    # Créer la carte avec l'outil de dessin (polyline avec édition)
    map_object = create_map(mosaic_path)
    st.write("**Dessinez, éditez ou supprimez des lignes sur la carte pour gérer vos profils d'élévation.**")

    # Initialiser le session_state pour la gestion des profils
    if "profiles" not in st.session_state:
        st.session_state.profiles = []

    # Récupération des dessins depuis la carte
    map_data = st_folium(map_object, width=700, height=500)

    drawings = []
    if map_data and isinstance(map_data, dict):
        raw_drawings = map_data.get("all_drawings")
        if isinstance(raw_drawings, list):
            drawings = raw_drawings

    # Mise à jour du session_state en se basant sur les dessins actuels
    st.session_state.profiles = []  # Réinitialisation
    for drawing in drawings:
        geom = drawing.get("geometry", {})
        if geom.get("type") == "LineString":
            coords = geom.get("coordinates", [])
            if coords:
                # Déterminer le nom de la carte associée en fonction du point médian
                map_name = get_map_name_for_profile(coords, tiff_info)
                st.session_state.profiles.append({"coords": coords, "map_name": map_name})

    # Affichage d'une figure distincte pour chaque profil enregistré
    if st.session_state.profiles:
        st.subheader("Profils d'Élévation enregistrés")
        for idx, profile in enumerate(st.session_state.profiles, start=1):
            coords = profile.get("coords", [])
            if not coords:
                continue
            sampled_points, distances = interpolate_line(coords, step=50)
            try:
                with rasterio.open(mosaic_path) as src:
                    elevations = []
                    for point in sampled_points:
                        for val in src.sample([point]):
                            elevations.append(val[0])
                # Création d'une figure pour ce profil (courbe fine sans marqueurs)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(distances, elevations, linewidth=1)
                ax.set_xlabel("Distance (m)")
                ax.set_ylabel("Altitude")
                ax.set_title(f"Profil d'Élévation {idx}")
                st.pyplot(fig)
                st.write(f"**Carte associée :** {profile.get('map_name', 'Inconnue')}")
            except Exception as e:
                st.error(f"Erreur lors de l'extraction du profil {idx}: {e}")
    else:
        st.info("Aucun profil d'élévation n'a été dessiné.")

if __name__ == "__main__":
    main()

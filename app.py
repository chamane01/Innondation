import streamlit as st
import os
import rasterio
import rasterio.merge
import folium
import math
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
from folium.plugins import Draw

def load_tiff_files(folder_path):
    """
    Charge les fichiers TIFF contenus dans un dossier.
    """
    try:
        tiff_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.tif')]
    except Exception as e:
        st.error(f"Erreur lors de la lecture du dossier {folder_path}: {e}")
        return []
    
    if not tiff_files:
        st.error("Aucun fichier TIFF trouvé dans le dossier.")
        return []
    
    valid_files = []
    for file in tiff_files:
        if os.path.exists(file):
            valid_files.append(file)
        else:
            st.error(f"Le fichier {file} n'existe pas.")
    return valid_files

def build_mosaic(tiff_files, mosaic_path="mosaic.tif"):
    """
    Construit une mosaïque à partir d'une liste de fichiers TIFF en utilisant rasterio.merge.
    La mosaïque est sauvegardée dans un fichier GeoTIFF.
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
      - L'outil de dessin pour tracer une ou plusieurs lignes.
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
    
    # Ajout de l'outil de dessin pour tracer des lignes
    draw = Draw(
        draw_options={
            'polyline': True,
            'polygon': False,
            'circle': False,
            'marker': False,
            'circlemarker': False,
            'rectangle': False,
        },
        edit_options={'edit': False}
    )
    draw.add_to(m)
    return m

def haversine(lon1, lat1, lon2, lat2):
    """
    Calcule la distance (en mètres) entre deux points (lon, lat)
    en utilisant la formule de Haversine.
    """
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371000  # Rayon de la Terre en mètres
    return c * r

def interpolate_line(coords, step=50):
    """
    Interpole les points le long d'une ligne pour obtenir un échantillonnage régulier.
    
    :param coords: liste de points [lon, lat]
    :param step: distance en mètres entre deux points interpolés
    :return: (sampled_points, cumulative_dist)
             - sampled_points: liste de points [lon, lat]
             - cumulative_dist: distance cumulée à chaque point
    """
    if len(coords) < 2:
        return coords, [0]
    sampled_points = [coords[0]]
    cumulative_dist = [0]
    for i in range(len(coords) - 1):
        start = coords[i]
        end = coords[i + 1]
        seg_distance = haversine(start[0], start[1], end[0], end[1])
        num_steps = max(int(seg_distance // step), 1)
        for j in range(1, num_steps + 1):
            fraction = j / num_steps
            lon = start[0] + fraction * (end[0] - start[0])
            lat = start[1] + fraction * (end[1] - start[1])
            dist = haversine(sampled_points[-1][0], sampled_points[-1][1], lon, lat)
            sampled_points.append([lon, lat])
            cumulative_dist.append(cumulative_dist[-1] + dist)
    return sampled_points, cumulative_dist

def main():
    st.title("Carte Dynamique avec Profils d'Élévation")
    folder_path = "TIFF"  # Modifiez si nécessaire

    if not os.path.exists(folder_path):
        st.error("Le dossier TIFF n'existe pas.")
        return

    st.write("Chargement des fichiers TIFF...")
    tiff_files = load_tiff_files(folder_path)
    if not tiff_files:
        st.error("Aucun fichier TIFF trouvé.")
        return

    st.write("Création de la mosaïque...")
    mosaic_path = build_mosaic(tiff_files, mosaic_path="mosaic.tif")
    if mosaic_path is None:
        st.error("Erreur lors de la création de la mosaïque.")
        return

    st.write("Création de la carte...")
    map_object = create_map(mosaic_path)
    st.write("**Dessinez une ou plusieurs lignes sur la carte pour tracer les profils d'élévation.**")
    
    # Récupérer les dessins réalisés sur la carte
    map_data = st_folium(map_object, width=700, height=500)

    if not map_data or not isinstance(map_data, dict):
        st.info("Aucune donnée de dessin récupérée.")
        return

    drawings = map_data.get("all_drawings", [])
    if not drawings or len(drawings) == 0:
        st.info("Dessinez au moins une ligne pour afficher les profils d'élévation.")
        return

    # Préparation du graphique pour les profils
    fig, ax = plt.subplots(figsize=(8, 4))
    profile_found = False

    # Traitement de chaque dessin (profil) de type LineString
    for i, drawing in enumerate(drawings):
        geom = drawing.get("geometry", {})
        if geom.get("type") == "LineString":
            coords = geom.get("coordinates", [])
            if not coords:
                st.info(f"Dessin {i+1} : aucune coordonnée trouvée.")
                continue

            # Interpolation des points le long de la ligne
            sampled_points, distances = interpolate_line(coords, step=50)
            try:
                with rasterio.open(mosaic_path) as src:
                    elevations = []
                    for point in sampled_points:
                        for val in src.sample([point]):
                            elevations.append(val[0])
                # Tracé du profil avec une épaisseur réduite
                ax.plot(distances, elevations, marker='o', linewidth=1, label=f"Profil {i+1}")
                profile_found = True
            except Exception as e:
                st.error(f"Erreur lors de l'extraction du profil {i+1}: {e}")
        else:
            st.info(f"Dessin {i+1} n'est pas une ligne (type: {geom.get('type')}).")
    
    if profile_found:
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Altitude")
        ax.set_title("Profils d'Élévation")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Aucun profil d'élévation valide n'a été dessiné.")

if __name__ == "__main__":
    main()

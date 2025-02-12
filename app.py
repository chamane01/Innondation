import streamlit as st
import os
import rasterio
import folium
import math
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
from folium.plugins import Draw
from osgeo import gdal

def load_tiff_files(folder_path):
    """
    Charge les fichiers TIFF contenus dans un dossier.
    """
    try:
        tiff_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tif')]
    except Exception as e:
        st.error(f"Erreur lors de la lecture du dossier {folder_path}: {e}")
        return []
    
    if not tiff_files:
        st.error("Aucun fichier TIFF trouvé dans le dossier.")
        return []
    
    # Vérifier l'existence de chaque fichier
    valid_files = []
    for file in tiff_files:
        if os.path.exists(file):
            valid_files.append(file)
        else:
            st.error(f"Le fichier {file} n'existe pas.")
    return valid_files

def build_vrt(tiff_files, vrt_path="mosaic.vrt"):
    """
    Construit un VRT (Virtual Raster) à partir d'une liste de fichiers TIFF.
    """
    try:
        vrt = gdal.BuildVRT(vrt_path, tiff_files)
        if vrt is None:
            raise Exception("La création du VRT a échoué.")
        vrt = None  # Libération du dataset VRT
        return vrt_path
    except Exception as e:
        st.error(f"Erreur lors de la création du VRT: {e}")
        return None

def create_map(mosaic_vrt):
    """
    Crée une carte avec une couche OSM, affiche la zone de la mosaïque VRT,
    et ajoute l'outil de dessin pour tracer une ou plusieurs lignes.
    """
    m = folium.Map(location=[0, 0], zoom_start=2)
    try:
        with rasterio.open(mosaic_vrt) as src:
            bounds = src.bounds
            # Affichage d'un rectangle représentant l'étendu de la mosaïque
            folium.Rectangle(
                bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                color='blue', fill=True, fill_opacity=0.4, tooltip="Mosaic VRT"
            ).add_to(m)
    except Exception as e:
        st.error(f"Erreur lors de l'ouverture du VRT {mosaic_vrt}: {e}")
    
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
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
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

def main():
    st.title("Carte Dynamique avec Profils d'Élévation")
    folder_path = "TIFF"  # Modifier si nécessaire
    
    if not os.path.exists(folder_path):
        st.error("Le dossier TIFF n'existe pas.")
        return

    st.write("Chargement des fichiers TIFF...")
    tiff_files = load_tiff_files(folder_path)
    if not tiff_files:
        st.warning("Aucun fichier TIFF n'a été trouvé.")
        return

    st.write("Création du VRT (mosaïque)...")
    vrt_path = build_vrt(tiff_files, vrt_path="mosaic.vrt")
    if vrt_path is None:
        st.error("Erreur lors de la création du VRT.")
        return

    st.write("Création de la carte...")
    map_object = create_map(vrt_path)
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

    # Extraction des profils pour chaque dessin de type LineString
    fig, ax = plt.subplots(figsize=(8, 4))
    profile_found = False

    for i, drawing in enumerate(drawings):
        geom = drawing.get("geometry", {})
        if geom.get("type") == "LineString":
            coords = geom.get("coordinates", [])
            if not coords:
                st.info(f"Dessin {i+1} : aucune coordonnée trouvée.")
                continue

            # Interpolation des points le long du trait
            sampled_points, distances = interpolate_line(coords, step=50)
            try:
                with rasterio.open(vrt_path) as src:
                    elevations = []
                    for point in sampled_points:
                        # Extraire la valeur d'altitude pour le point donné
                        for val in src.sample([point]):
                            elevations.append(val[0])
                # Tracer le profil avec un trait de faible épaisseur
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

import streamlit as st
import os
import rasterio
import folium
import math
import matplotlib.pyplot as plt
from rasterio.warp import calculate_default_transform, reproject, Resampling
from streamlit_folium import st_folium
from folium.plugins import Draw

def reproject_tiff(input_path, output_path, dst_crs):
    """Reprojette un fichier TIFF vers le système de coordonnées spécifié."""
    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )

def load_tiff_files(folder_path):
    """Charge et reprojette les fichiers TIFF contenus dans un dossier."""
    tiff_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    reproj_files = []
    for file in tiff_files:
        input_path = os.path.join(folder_path, file)
        output_path_4326 = os.path.join(folder_path, f"reproj_{file}")
        reproject_tiff(input_path, output_path_4326, 'EPSG:4326')
        reproj_files.append(output_path_4326)
    return reproj_files

def create_map(tiff_files):
    """Crée une carte avec une couche OSM, affiche les TIFF reprojectés et ajoute l'outil de dessin."""
    m = folium.Map(location=[0, 0], zoom_start=2)
    # Affichage des zones couvertes par chaque TIFF
    for tiff in tiff_files:
        with rasterio.open(tiff) as src:
            bounds = src.bounds
            folium.Rectangle(
                bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                color='blue', fill=True, fill_opacity=0.4, tooltip=tiff
            ).add_to(m)
    # Ajouter l'outil de dessin (uniquement pour tracer une ligne)
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
    """Calcule la distance (en mètres) entre deux points (lon, lat) à l'aide de la formule de haversine."""
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
             sampled_points: liste de points [lon, lat]
             cumulative_dist: distance cumulée à chaque point
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
    st.title("Carte Dynamique avec Données d'Élévation")
    folder_path = "TIFF"  # Modifier selon l'emplacement réel du dossier

    if not os.path.exists(folder_path):
        st.error("Le dossier TIFF n'existe pas.")
        return

    st.write("Chargement et reprojection des fichiers TIFF...")
    reproj_files = load_tiff_files(folder_path)

    if not reproj_files:
        st.warning("Aucun fichier TIFF trouvé.")
        return

    # Permettre à l'utilisateur de choisir le fichier TIFF à utiliser pour le profil d'élévation
    selected_tiff = st.selectbox("Sélectionnez le fichier TIFF pour le profil d'élévation", reproj_files)

    st.write("Création de la carte...")
    map_object = create_map(reproj_files)
    st.write("Dessinez une ligne sur la carte pour tracer le profil d'élévation.")
    
    # Utiliser st_folium pour rendre la carte interactive et récupérer les dessins
    map_data = st_folium(map_object, width=700, height=500)

    # Vérifier si une ligne a été dessinée
    if map_data and "all_drawings" in map_data and len(map_data["all_drawings"]) > 0:
        # On récupère le dernier dessin effectué
        drawing = map_data["all_drawings"][-1]
        if drawing["geometry"]["type"] == "LineString":
            coords = drawing["geometry"]["coordinates"]
            # Interpolation des points pour obtenir un échantillonnage régulier
            sampled_points, distances = interpolate_line(coords, step=50)
            try:
                with rasterio.open(selected_tiff) as src:
                    # Extraire la valeur d'altitude à chaque point interpolé
                    elevations = []
                    for point in sampled_points:
                        for val in src.sample([point]):
                            elevations.append(val[0])
                # Tracer le profil d'élévation
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(distances, elevations, marker='o')
                ax.set_xlabel("Distance (m)")
                ax.set_ylabel("Altitude")
                ax.set_title("Profil d'Élévation")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Erreur lors de l'extraction du profil : {e}")
        else:
            st.info("Veuillez dessiner une ligne (polyline) pour obtenir le profil d'élévation.")
    else:
        st.info("Dessinez une ligne sur la carte pour afficher le profil d'élévation.")

if __name__ == "__main__":
    main()

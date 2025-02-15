import streamlit as st
import os
import rasterio
import folium
import math
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
from folium.plugins import Draw

def load_tiff_files(folder_path):
    """Charge les fichiers TIFF contenus dans un dossier sans reprojection."""
    try:
        tiff_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    except Exception as e:
        st.error(f"Erreur lors de la lecture du dossier {folder_path}: {e}")
        return []
    
    if not tiff_files:
        st.error("Aucun fichier TIFF trouvé dans le dossier.")
        return []
    
    tiff_paths = []
    for file in tiff_files:
        input_path = os.path.join(folder_path, file)
        if not os.path.exists(input_path):
            st.error(f"Le fichier {input_path} n'existe pas.")
            continue
        tiff_paths.append(input_path)
    return tiff_paths

def create_map(tiff_files):
    """Crée une carte avec une couche OSM, affiche les fichiers TIFF et ajoute l'outil de dessin."""
    m = folium.Map(location=[0, 0], zoom_start=2)
    # Affichage des zones couvertes par chaque TIFF
    for tiff in tiff_files:
        try:
            with rasterio.open(tiff) as src:
                bounds = src.bounds
                folium.Rectangle(
                    bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                    color='blue', fill=True, fill_opacity=0.4, tooltip=tiff
                ).add_to(m)
        except Exception as e:
            st.error(f"Erreur lors de l'ouverture du fichier {tiff}: {e}")
            continue

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

    st.write("Chargement des fichiers TIFF...")
    tiff_files = load_tiff_files(folder_path)

    if not tiff_files:
        st.warning("Aucun fichier TIFF n'a été trouvé.")
        return

    # Permettre à l'utilisateur de choisir le fichier TIFF à utiliser pour le profil d'élévation
    selected_tiff = st.selectbox("Sélectionnez le fichier TIFF pour le profil d'élévation", tiff_files)

    st.write("Création de la carte...")
    map_object = create_map(tiff_files)
    st.write("Dessinez une ligne sur la carte pour tracer le profil d'élévation.")
    
    # Utiliser st_folium pour rendre la carte interactive et récupérer les dessins
    map_data = st_folium(map_object, width=700, height=500)

    if not map_data or not isinstance(map_data, dict):
        st.info("Aucune donnée de dessin récupérée.")
        return

    # Utiliser .get() pour récupérer la liste des dessins
    drawings = map_data.get("all_drawings", [])
    if drawings and len(drawings) > 0:
        drawing = drawings[-1]
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

import streamlit as st
import os
import rasterio
import rasterio.merge
import rasterio.mask
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
        st.error(f"Erreur lors de la lecture du dossier {folder_path} : {e}")
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
    """
    Crée une carte Folium en affichant l'emprise de la mosaïque 
    et en ajoutant l'outil de dessin pour pouvoir tracer à la fois
    des rectangles (pour sélectionner une emprise et générer des contours)
    et des lignes (pour tracer des profils d'élévation).
    """
    m = folium.Map(location=[0, 0], zoom_start=2)
    
    # Calque indiquant l'emprise de la mosaïque
    mosaic_group = folium.FeatureGroup(name="Mosaïque")
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
    
    # Ajout de l'outil de dessin avec 2 options : rectangle et polyline
    Draw(
        draw_options={
            'rectangle': True,
            'polyline': {'allowIntersection': False},
            'polygon': False,
            'circle': False,
            'marker': False,
            'circlemarker': False
        },
        edit_options={'edit': True, 'remove': True}
    ).add_to(m)
    
    folium.LayerControl().add_to(m)
    
    return m

def generate_contours(mosaic_file, drawing_geometry=None):
    """
    Génère et affiche les courbes de niveau (contours) à partir du fichier TIFF.
    Si drawing_geometry est fourni (GeoJSON d'un rectangle dessiné), on ne
    génère les contours que sur cette zone.
    """
    try:
        with rasterio.open(mosaic_file) as src:
            if drawing_geometry is not None:
                # Découpage de la mosaïque selon l'emprise dessinée
                out_image, out_transform = rasterio.mask.mask(src, [drawing_geometry], crop=True)
                data = out_image[0]
            else:
                data = src.read(1)
                out_transform = src.transform
            nodata = src.nodata
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier TIFF : {e}")
        return
    
    # Remplacer les valeurs nodata par NaN
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)
    
    nrows, ncols = data.shape
    # Calcul des coordonnées centrales de chaque pixel
    x_coords = np.arange(ncols) * out_transform.a + out_transform.c + out_transform.a/2
    y_coords = np.arange(nrows) * out_transform.e + out_transform.f + out_transform.e/2
    X, Y = np.meshgrid(x_coords, y_coords)
    
    try:
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
    except Exception as e:
        st.error(f"Erreur lors du calcul des valeurs min et max : {e}")
        return
    
    levels = np.linspace(vmin, vmax, 15)
    
    # Tracé des contours
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contour(X, Y, data, levels=levels, cmap='terrain')
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_title("Contours d'élévation")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    st.pyplot(fig)

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
    """
    Interpole des points le long d'une ligne pour obtenir des échantillons réguliers.
    'coords' est une liste de [lon, lat].
    """
    if len(coords) < 2:
        return coords, [0]
    sampled_points = [coords[0]]
    cumulative_dist = [0]
    for i in range(len(coords) - 1):
        start = coords[i]
        end = coords[i+1]
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

##################
# Fonction main  #
##################

def main():
    st.title("Carte Interactive : Contours et Profils d'Élévation")
    
    # Initialisation du mode dans la session :
    # "none"     -> aucun mode choisi (affichage des 2 boutons)
    # "contours" -> menu de génération de contours actif
    # "profiles" -> menu de tracé de profils actif
    if "mode" not in st.session_state:
        st.session_state.mode = "none"
    
    # Saisie du nom de la carte (sera utilisé pour nommer les profils)
    map_name = st.text_input("Nom de votre carte", value="Ma Carte")
    
    # Chargement et construction de la mosaïque
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

    # Création de la carte interactive avec outils de dessin pour rectangle et ligne
    m = create_map(mosaic_path)
    st.write("**Utilisez l'outil de dessin sur la carte ci-dessus.**")
    map_data = st_folium(m, width=700, height=500)
    
    #############################
    # Espace des options sous la carte
    #############################
    options_container = st.container()
    
    # Si aucun mode n'est sélectionné, afficher les 2 boutons
    if st.session_state.mode == "none":
        col1, col2 = options_container.columns(2)
        if col1.button("Tracer des profils"):
            st.session_state.mode = "profiles"
        if col2.button("Générer des contours"):
            st.session_state.mode = "contours"
    
    # Mode "Générer des contours" (à partir d'un rectangle dessiné)
    if st.session_state.mode == "contours":
        st.subheader("Générer des contours")
        drawing_geometries = []
        if isinstance(map_data, dict):
            # Les rectangles dessinés sont renvoyés comme des polygones
            raw_drawings = map_data.get("all_drawings", [])
            for drawing in raw_drawings:
                if isinstance(drawing, dict) and drawing.get("geometry", {}).get("type") == "Polygon":
                    drawing_geometries.append(drawing.get("geometry"))
        if not drawing_geometries:
            st.warning("Veuillez dessiner une emprise (rectangle) sur la carte.")
        else:
            for i, geom in enumerate(drawing_geometries, start=1):
                st.markdown(f"**Contours pour l'emprise {i}**")
                generate_contours(mosaic_path, geom)
        if st.button("Retour", key="retour_contours"):
            st.session_state.mode = "none"
    
    # Mode "Tracer des profils" (à partir d'une ligne tracée)
    if st.session_state.mode == "profiles":
        st.subheader("Tracer des profils")
        current_drawings = []
        if isinstance(map_data, dict):
            raw_drawings = map_data.get("all_drawings") or []
            # On filtre uniquement les lignes dessinées
            current_drawings = [
                d for d in raw_drawings 
                if isinstance(d, dict) and d.get("geometry", {}).get("type") == "LineString"
            ]
        if not current_drawings:
            st.info("Aucun profil dessiné pour le moment. Tracez une ligne sur la carte.")
        else:
            for i, drawing in enumerate(current_drawings):
                profile = {
                    "coords": drawing["geometry"]["coordinates"],
                    "name": f"{map_name} - Profil {i+1}"
                }
                st.markdown(f"#### {profile['name']}")
                col_a, col_b = st.columns([1, 4])
                with col_a:
                    # Possibilité de renommer le profil
                    new_name = st.text_input("Nom du profil", value=profile["name"], key=f"profile_name_{i}")
                    profile["name"] = new_name
                    # Choix du mode de présentation
                    presentation_mode = st.radio(
                        "Mode de présentation",
                        ("Automatique", "Manuel"),
                        key=f"presentation_mode_{i}"
                    )
                    manual_options = {}
                    if presentation_mode == "Manuel":
                        manual_options["ecart_distance"] = st.number_input(
                            "Ecart distance (m)",
                            min_value=1.0,
                            value=50.0,
                            step=1.0,
                            key=f"ecart_distance_{i}"
                        )
                        manual_options["ecart_altitude"] = st.number_input(
                            "Ecart altitude (m)",
                            min_value=1.0,
                            value=10.0,
                            step=1.0,
                            key=f"ecart_altitude_{i}"
                        )
                with col_b:
                    try:
                        # Interpolation de la ligne pour obtenir des points régulièrement espacés
                        points, distances = interpolate_line(profile["coords"])
                        # Extraction des altitudes depuis la mosaïque
                        with rasterio.open(mosaic_path) as src:
                            elevations = [list(src.sample([p]))[0][0] for p in points]
                        
                        fig, ax = plt.subplots(figsize=(8, 3))
                        ax.plot(distances, elevations, 'b-', linewidth=1.5)
                        ax.set_title(profile["name"])
                        ax.set_xlabel("Distance (m)")
                        ax.set_ylabel("Altitude (m)")
                        
                        # En mode manuel, ajuster les graduations
                        if presentation_mode == "Manuel":
                            ecart_distance = manual_options.get("ecart_distance", 50.0)
                            ecart_altitude = manual_options.get("ecart_altitude", 10.0)
                            xticks = np.arange(0, max(distances) + ecart_distance, ecart_distance)
                            yticks = np.arange(min(elevations), max(elevations) + ecart_altitude, ecart_altitude)
                            ax.set_xticks(xticks)
                            ax.set_yticks(yticks)
                        
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Erreur de traitement : {e}")
        if st.button("Retour", key="retour_profiles"):
            st.session_state.mode = "none"

if __name__ == "__main__":
    main()

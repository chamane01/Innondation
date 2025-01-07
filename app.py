import rasterio
from rasterio.warp import transform_bounds
import geopandas as gpd
import numpy as np
import folium
from folium.plugins import MeasureControl, Draw
import streamlit as st
from shapely.geometry import Point
from streamlit_folium import folium_static

# Fonction pour charger un fichier TIFF
def load_tiff(file_path, target_crs="EPSG:4326"):
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)  # Lire la première bande
            src_crs = src.crs  # CRS source
            bounds = src.bounds  # Bornes source

            # Reprojeter les bornes vers le CRS cible
            target_bounds = transform_bounds(src_crs, target_crs, *bounds)

        return data, target_bounds
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier GeoTIFF : {e}")
        return None, None

# Fonction pour charger un fichier GeoJSON ou Shapefile et le projeter
def load_and_reproject_shapefile(file_path, target_crs="EPSG:4326"):
    try:
        gdf = gpd.read_file(file_path)
        gdf = gdf.to_crs(target_crs)  # Reprojection au CRS cible
        return gdf
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier Shapefile/GeoJSON : {e}")
        return None

# Calcul de hauteur relative (pour la méthode 1 : MNS - MNT)
def calculate_heights(mns, mnt):
    return np.maximum(0, mns - mnt)

# Fonction pour calculer le volume dans l'emprise de la polygonale (Méthode 1 : MNS - MNT)
def calculate_volume_in_polygon(mns, mnt, bounds, polygon_gdf):
    # Calculer les hauteurs relatives
    heights = calculate_heights(mns, mnt)

    # Extraire les coordonnées de la polygonale
    polygon_mask = np.zeros_like(heights, dtype=bool)
    height = bounds[3] - bounds[1]
    width = bounds[2] - bounds[0]
    img_height, img_width = heights.shape

    for _, row in polygon_gdf.iterrows():
        polygon = row.geometry
        for x in range(img_width):
            for y in range(img_height):
                lat = bounds[3] - height * (y / img_height)
                lon = bounds[0] + width * (x / img_width)
                if polygon.contains(Point(lon, lat)):
                    polygon_mask[y, x] = True

    # Calculer le volume (somme des hauteurs dans la polygonale)
    volume = np.sum(heights[polygon_mask])  # Volume en m³ (si les données sont en mètres)
    return volume

# Fonction pour calculer le volume avec MNS seul (Méthode 2 : MNS seul)
def calculate_volume_without_mnt(mns, bounds, polygon_gdf, reference_altitude):
    # Extraire les coordonnées de la polygonale
    polygon_mask = np.zeros_like(mns, dtype=bool)
    height = bounds[3] - bounds[1]
    width = bounds[2] - bounds[0]
    img_height, img_width = mns.shape

    for _, row in polygon_gdf.iterrows():
        polygon = row.geometry
        for x in range(img_width):
            for y in range(img_height):
                lat = bounds[3] - height * (y / img_height)
                lon = bounds[0] + width * (x / img_width)
                if polygon.contains(Point(lon, lat)):
                    polygon_mask[y, x] = True

    # Calculer les différences par rapport à l'altitude de référence
    diff = mns - reference_altitude

    # Calculer les volumes positifs et négatifs
    positive_volume = np.sum(diff[polygon_mask & (diff > 0)])  # Volume positif
    negative_volume = np.sum(diff[polygon_mask & (diff < 0)])  # Volume négatif
    real_volume = positive_volume + negative_volume  # Volume réel

    return positive_volume, negative_volume, real_volume

# Streamlit app
st.title("Calcul de volume avec MNS et MNT")

# Boutons sous la carte
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Faire une carte"):
        st.info("Fonctionnalité 'Faire une carte' en cours de développement.")
with col2:
    if st.button("Calculer des volumes"):
        st.session_state.show_volume_sidebar = True
with col3:
    if st.button("Détecter les arbres"):
        st.info("Fonctionnalité 'Détecter les arbres' en cours de développement.")

# Affichage des paramètres pour le calcul des volumes
if st.session_state.get("show_volume_sidebar", False):
    st.sidebar.title("Paramètres de calcul des volumes")

    # Choix de la méthode
    method = st.sidebar.radio(
        "Choisissez la méthode de calcul :",
        ("Méthode 1 : MNS - MNT", "Méthode 2 : MNS seul")
    )

    # Téléversement des fichiers
    mns_file = st.sidebar.file_uploader("Téléchargez le fichier MNS (TIFF)", type=["tif", "tiff"])
    polygon_file = st.sidebar.file_uploader("Téléchargez un fichier de polygone (obligatoire)", type=["geojson", "shp"])

    if method == "Méthode 1 : MNS - MNT":
        mnt_file = st.sidebar.file_uploader("Téléchargez le fichier MNT (TIFF)", type=["tif", "tiff"])
    else:
        mnt_file = None

    if mns_file and polygon_file and (method == "Méthode 2 : MNS seul" or mnt_file):
        mns, mns_bounds = load_tiff(mns_file)
        polygons_gdf = load_and_reproject_shapefile(polygon_file)

        if mns is None or polygons_gdf is None:
            st.sidebar.error("Erreur lors du chargement des fichiers.")
        else:
            try:
                if method == "Méthode 1 : MNS - MNT":
                    mnt, mnt_bounds = load_tiff(mnt_file)
                    if mnt is None or mnt_bounds != mns_bounds:
                        st.sidebar.error("Les fichiers doivent avoir les mêmes bornes géographiques.")
                    else:
                        volume = calculate_volume_in_polygon(mns, mnt, mnt_bounds, polygons_gdf)
                        st.sidebar.write(f"Volume calculé dans la polygonale : {volume:.2f} m³")
                else:
                    # Saisie de l'altitude de référence pour la méthode 2
                    reference_altitude = st.sidebar.number_input(
                        "Entrez l'altitude de référence (en mètres) :",
                        value=0.0,
                        step=0.1
                    )
                    positive_volume, negative_volume, real_volume = calculate_volume_without_mnt(
                        mns, mns_bounds, polygons_gdf, reference_altitude
                    )
                    st.sidebar.write(f"Volume positif (au-dessus de la référence) : {positive_volume:.2f} m³")
                    st.sidebar.write(f"Volume négatif (en dessous de la référence) : {negative_volume:.2f} m³")
                    st.sidebar.write(f"Volume réel (différence) : {real_volume:.2f} m³")

                # Afficher la carte
                center_lat = (mns_bounds[1] + mns_bounds[3]) / 2
                center_lon = (mns_bounds[0] + mns_bounds[2]) / 2
                fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12)

                # Ajouter le MNS
                folium.raster_layers.ImageOverlay(
                    image=mns,
                    bounds=[[mns_bounds[1], mns_bounds[0]], [mns_bounds[3], mns_bounds[2]]],
                    opacity=0.7,
                    name="MNS"
                ).add_to(fmap)

                # Ajouter la polygonale
                folium.GeoJson(
                    polygons_gdf,
                    name="Polygone",
                    style_function=lambda x: {'fillOpacity': 0, 'color': 'red', 'weight': 2}
                ).add_to(fmap)

                # Ajouter les contrôles de carte
                fmap.add_child(MeasureControl(position='topleft'))
                fmap.add_child(Draw(position='topleft', export=True))
                fmap.add_child(folium.LayerControl(position='topright'))

                # Afficher la carte
                folium_static(fmap, width=700, height=500)

            except Exception as e:
                st.sidebar.error(f"Erreur lors du calcul du volume : {e}")

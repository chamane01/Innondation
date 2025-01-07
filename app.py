import rasterio
from rasterio.warp import transform_bounds
from rasterio.features import rasterize
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
            transform = src.transform  # Transformation affine
            bounds = src.bounds  # Bornes source

            # Reprojeter les bornes vers le CRS cible
            target_bounds = transform_bounds(src_crs, target_crs, *bounds)

        return data, transform, src_crs, bounds, target_bounds
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier GeoTIFF : {e}")
        return None, None, None, None, None

# Fonction pour charger un fichier GeoJSON ou Shapefile et le projeter
def load_and_reproject_shapefile(file_path, target_crs="EPSG:4326"):
    try:
        gdf = gpd.read_file(file_path)
        gdf = gdf.to_crs(target_crs)  # Reprojection au CRS cible
        return gdf
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier Shapefile/GeoJSON : {e}")
        return None

# Calcul de hauteur relative
def calculate_heights(mns, mnt):
    return np.maximum(0, mns - mnt)

# Fonction pour calculer le volume dans l'emprise de la polygonale (MNS - MNT)
def calculate_volume_with_mnt(mns, mnt, transform, polygon_gdf):
    # Calculer les hauteurs relatives
    heights = calculate_heights(mns, mnt)

    # Créer un masque à partir du polygone
    mask = rasterize(
        shapes=polygon_gdf.geometry,
        out_shape=mns.shape,
        transform=transform,
        fill=0,
        default_value=1,
        dtype=np.uint8
    ).astype(bool)

    # Calculer la surface d'un pixel
    pixel_width = transform[0]
    pixel_height = -transform[4]  # Négatif car la transformation est en y inversé
    cell_area = pixel_width * pixel_height  # Surface en m²

    # Calculer le volume
    volume = np.sum(heights[mask]) * cell_area  # Volume en m³
    return volume

# Fonction pour calculer le volume avec MNS seul
def calculate_volume_without_mnt(mns, transform, polygon_gdf):
    # Créer un masque à partir du polygone
    mask = rasterize(
        shapes=polygon_gdf.geometry,
        out_shape=mns.shape,
        transform=transform,
        fill=0,
        default_value=1,
        dtype=np.uint8
    ).astype(bool)

    # Calculer la surface d'un pixel
    pixel_width = transform[0]
    pixel_height = -transform[4]  # Négatif car la transformation est en y inversé
    cell_area = pixel_width * pixel_height  # Surface en m²

    # Calculer le volume
    volume = np.sum(mns[mask]) * cell_area  # Volume en m³
    return volume

# Streamlit app
st.title("Calcul de volume avec MNS et MNT")

# Boutons sous la carte
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Créer une carte"):
        st.info("Fonctionnalité 'Créer une carte' en cours de développement.")
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
        mns, mns_transform, mns_crs, mns_bounds, mns_target_bounds = load_tiff(mns_file)
        polygons_gdf = load_and_reproject_shapefile(polygon_file, mns_crs)

        if mns is None or polygons_gdf is None:
            st.sidebar.error("Erreur lors du chargement des fichiers.")
        else:
            try:
                if method == "Méthode 1 : MNS - MNT":
                    mnt, mnt_transform, mnt_crs, mnt_bounds, mnt_target_bounds = load_tiff(mnt_file)
                    if mnt is None or mnt_bounds != mns_bounds:
                        st.sidebar.error("Les fichiers doivent avoir les mêmes bornes géographiques.")
                    else:
                        volume = calculate_volume_with_mnt(mns, mnt, mns_transform, polygons_gdf)
                        st.sidebar.write(f"Volume calculé dans la polygonale : {volume:.2f} m³")
                else:
                    volume = calculate_volume_without_mnt(mns, mns_transform, polygons_gdf)
                    st.sidebar.write(f"Volume calculé dans la polygonale : {volume:.2f} m³")

                # Afficher la carte
                center_lat = (mns_target_bounds[1] + mns_target_bounds[3]) / 2
                center_lon = (mns_target_bounds[0] + mns_target_bounds[2]) / 2
                fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12)

                # Ajouter le MNS
                folium.raster_layers.ImageOverlay(
                    image=mns,
                    bounds=[[mns_target_bounds[1], mns_target_bounds[0]], [mns_target_bounds[3], mns_target_bounds[2]]],
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

import streamlit as st
import folium
from folium.plugins import MeasureControl, Draw
from streamlit_folium import folium_static
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from PIL import Image
from folium import plugins
from rasterio.plot import reshape_as_image
import rasterio.warp

# Fonction pour reprojeter un fichier TIFF
def reproject_tiff(uploaded_file, target_crs="EPSG:4326"):
    try:
        with rasterio.open(uploaded_file) as src:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': target_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            reprojected_tiff = "reprojected.tiff"
            with rasterio.open(reprojected_tiff, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.nearest
                    )
            return reprojected_tiff, src.bounds
    except Exception as e:
        st.error(f"Erreur lors de la reprojection du fichier TIFF : {e}")
        return None, None

# Fonction pour ajouter une couche d'image sur une carte Folium
def add_image_overlay(map_object, tiff_path, bounds, layer_name="TIFF Layer"):
    try:
        with rasterio.open(tiff_path) as src:
            image = src.read(1)  # Lire la première bande
            folium.raster_layers.ImageOverlay(
                image=image,
                bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                name=layer_name,
                opacity=0.6
            ).add_to(map_object)
    except Exception as e:
        st.error(f"Erreur lors de l'ajout de la couche d'image : {e}")

# Interface Streamlit
st.title("AFRIQUE CARTOGRAPHIE")
uploaded_file = st.file_uploader("Téléversez un fichier TIFF (orthophoto ou orthomosaïque)", type=["tiff", "tif"])

# Carte initiale
center_lat, center_lon = 7.0, -5.0
zoom_start = 6
fmap = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)

# Ajouter l'outil de mesure
fmap.add_child(MeasureControl(position='topleft', primary_length_unit='meters', secondary_length_unit='kilometers'))

# Ajouter l'outil de dessin
draw = Draw(position='topleft', export=True,
            draw_options={
                'polyline': {'shapeOptions': {'color': 'blue', 'weight': 4, 'opacity': 0.7}},
                'polygon': {'shapeOptions': {'color': 'green', 'weight': 4, 'opacity': 0.7}},
                'rectangle': {'shapeOptions': {'color': 'red', 'weight': 4, 'opacity': 0.7}},
                'circle': {'shapeOptions': {'color': 'purple', 'weight': 4, 'opacity': 0.7}}},
            edit_options={'edit': True})
fmap.add_child(draw)

# Traitement du fichier téléversé
if uploaded_file:
    with open("uploaded_file.tif", "wb") as f:
        f.write(uploaded_file.read())

    st.write("Reprojection du fichier TIFF...")
    reprojected_file, bounds = reproject_tiff("uploaded_file.tif", target_crs="EPSG:4326")

    if reprojected_file and bounds:
        st.write("Ajout de la couche TIFF à la carte...")
        add_image_overlay(fmap, reprojected_file, bounds)

# Ajouter le contrôle des couches
fmap.add_child(folium.LayerControl(position='topright'))

# Afficher la carte
folium_static(fmap)

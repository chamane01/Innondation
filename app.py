import streamlit as st
import rasterio
import rasterio.warp
import folium
from folium import plugins
from folium.plugins import MeasureControl, Draw
from rasterio.plot import reshape_as_image
from PIL import Image
from streamlit_folium import folium_static
from rasterio.warp import transform_bounds
import numpy as np
from sklearn.cluster import DBSCAN
import geopandas as gpd
from shapely.geometry import Polygon, Point, LineString
from folium import IFrame
from streamlit_folium import st_folium
import json
from io import BytesIO
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import matplotlib.pyplot as plt
import os

# Reprojection function
def reproject_tiff(input_tiff, target_crs):
    """Reproject a TIFF file to a target CRS."""
    with rasterio.open(input_tiff) as src:
        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height,
        })

        reprojected_tiff = "reprojected.tiff"
        with rasterio.open(reprojected_tiff, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=rasterio.warp.Resampling.nearest,
                )
    return reprojected_tiff

# Function to apply color gradient to a DEM TIFF
def apply_color_gradient(tiff_path, output_path):
    """Apply a color gradient to the DEM TIFF and save it as a PNG."""
    with rasterio.open(tiff_path) as src:
        # Read the DEM data
        dem_data = src.read(1)
        
        # Create a color map using matplotlib
        cmap = plt.get_cmap("terrain")
        norm = plt.Normalize(vmin=dem_data.min(), vmax=dem_data.max())
        
        # Apply the colormap
        colored_image = cmap(norm(dem_data))
        
        # Save the colored image as PNG
        plt.imsave(output_path, colored_image)
        plt.close()

# Overlay function for TIFF images
def add_image_overlay(map_object, tiff_path, bounds, name):
    """Add a TIFF image overlay to a Folium map."""
    with rasterio.open(tiff_path) as src:
        image = reshape_as_image(src.read())
        folium.raster_layers.ImageOverlay(
            image=image,
            bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
            name=name,
            opacity=0.6,
        ).add_to(map_object)

# Main application
def main():
    st.title("DESSINER une CARTE")

    # Initialize session state for layers
    if "map_layers" not in st.session_state:
        st.session_state["map_layers"] = []

    # Initialize map
    fmap = folium.Map(location=[0, 0], zoom_start=2)
    fmap.add_child(MeasureControl(position="topleft"))
    draw = Draw(
        position="topleft",
        export=True,
        draw_options={
            "polyline": {"shapeOptions": {"color": "orange", "weight": 4, "opacity": 0.7}},
            "polygon": {"shapeOptions": {"color": "green", "weight": 4, "opacity": 0.7}},
            "rectangle": {"shapeOptions": {"color": "red", "weight": 4, "opacity": 0.7}},
            "circle": {"shapeOptions": {"color": "purple", "weight": 4, "opacity": 0.7}},
        },
        edit_options={"edit": True},
    )
    fmap.add_child(draw)

    # Téléversement des fichiers TIFF
    st.subheader("Téléverser des fichiers TIFF")
    uploaded_tiff = st.file_uploader("Choisir un fichier TIFF (Orthophoto, MNT, MNS)", type=["tif", "tiff"], key="tiff_uploader")
    tiff_type = st.selectbox("Sélectionnez le type de fichier TIFF", ["Orthophoto", "MNT", "MNS"], key="tiff_type")

    # Téléversement des fichiers GeoJSON
    st.subheader("Téléverser des fichiers GeoJSON")
    uploaded_geojson = st.file_uploader("Choisir un fichier GeoJSON (Routes, Polygonale)", type=["geojson"], key="geojson_uploader")
    geojson_type = st.selectbox("Sélectionnez le type de fichier GeoJSON", ["Routes", "Polygonale"], key="geojson_type")
    color = st.color_picker(f"Choisir la couleur pour {geojson_type}", "#FFA500" if geojson_type == "Routes" else "#FF0000", key="geojson_color")
    weight = st.slider(f"Choisir l'épaisseur pour {geojson_type}", 1, 10, 4, key="geojson_weight")
    opacity = st.slider(f"Choisir l'opacité pour {geojson_type}", 0.1, 1.0, 0.7, key="geojson_opacity")

    # Bouton pour ajouter une couche
    if st.button("Ajouter une couche"):
        if uploaded_tiff:
            tiff_path = uploaded_tiff.name
            with open(tiff_path, "wb") as f:
                f.write(uploaded_tiff.read())

            st.write(f"Reprojection du fichier {tiff_type}...")
            try:
                reprojected_tiff = reproject_tiff(tiff_path, "EPSG:4326")
                st.session_state["map_layers"].append(("tiff", reprojected_tiff, tiff_type))
                st.success(f"Fichier {tiff_type} ajouté à la liste des couches.")
            except Exception as e:
                st.error(f"Erreur lors de la reprojection : {e}")

        if uploaded_geojson:
            try:
                geojson_data = json.load(uploaded_geojson)
                st.session_state["map_layers"].append(("geojson", geojson_data, geojson_type, color, weight, opacity))
                st.success(f"Fichier {geojson_type} ajouté à la liste des couches.")
            except Exception as e:
                st.error(f"Erreur lors du chargement du GeoJSON : {e}")

    # Afficher la liste des couches disponibles
    if st.session_state["map_layers"]:
        st.subheader("Liste des couches disponibles")
        layer_names = [f"{layer[2]} ({layer[0]})" for layer in st.session_state["map_layers"]]
        selected_layers = st.multiselect("Sélectionnez les couches à afficher", layer_names)

        # Bouton pour afficher les couches sélectionnées sur la carte
        if st.button("Afficher sur la carte"):
            for layer in st.session_state["map_layers"]:
                layer_name = f"{layer[2]} ({layer[0]})"
                if layer_name in selected_layers:
                    if layer[0] == "tiff":
                        tiff_path, tiff_type = layer[1], layer[2]
                        try:
                            with rasterio.open(tiff_path) as src:
                                bounds = src.bounds
                                if tiff_type == "MNT" or tiff_type == "MNS":
                                    # Create a temporary PNG file for the colorized DEM
                                    temp_png_path = f"{tiff_type.lower()}_colored.png"
                                    apply_color_gradient(tiff_path, temp_png_path)
                                    add_image_overlay(fmap, temp_png_path, bounds, tiff_type)
                                    os.remove(temp_png_path)
                                else:
                                    add_image_overlay(fmap, tiff_path, bounds, tiff_type)
                        except Exception as e:
                            st.error(f"Erreur lors de l'affichage du fichier {tiff_type} : {e}")

                    elif layer[0] == "geojson":
                        geojson_data, geojson_type, color, weight, opacity = layer[1], layer[2], layer[3], layer[4], layer[5]
                        try:
                            folium.GeoJson(
                                geojson_data,
                                name=geojson_type,
                                style_function=lambda x, color=color, weight=weight, opacity=opacity: {
                                    "color": color,
                                    "weight": weight,
                                    "opacity": opacity,
                                    "fillColor": "transparent" if geojson_type == "Polygonale" else color,
                                    "fillOpacity": 0.1 if geojson_type == "Polygonale" else opacity,
                                }
                            ).add_to(fmap)
                        except Exception as e:
                            st.error(f"Erreur lors de l'affichage du fichier {geojson_type} : {e}")

    # Ajout des contrôles de calques
    folium.LayerControl().add_to(fmap)

    # Affichage de la carte
    folium_static(fmap, width=700, height=500)

if __name__ == "__main__":
    main()

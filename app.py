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
    st.title("DESSINER une CARTE ")

    # Initialize session state for drawings
    if "drawings" not in st.session_state:
        st.session_state["drawings"] = {
            "type": "FeatureCollection",
            "features": [],
        }

    # Initialize map
    fmap = folium.Map(location=[0, 0], zoom_start=2)
    fmap.add_child(MeasureControl(position="topleft"))
    draw = Draw(
        position="topleft",
        export=True,
        draw_options={
            "polyline": {"shapeOptions": {"color": "orange", "weight": 4, "opacity": 0.7}},  # Change color to orange
            "polygon": {"shapeOptions": {"color": "green", "weight": 4, "opacity": 0.7}},
            "rectangle": {"shapeOptions": {"color": "red", "weight": 4, "opacity": 0.7}},
            "circle": {"shapeOptions": {"color": "purple", "weight": 4, "opacity": 0.7}},
        },
        edit_options={"edit": True},
    )
    fmap.add_child(draw)

    # Téléversement d'une orthophoto (TIFF)
    uploaded_tiff = st.file_uploader("Téléverser une orthophoto (TIFF)", type=["tif", "tiff"])
    if uploaded_tiff:
        tiff_path = uploaded_tiff.name
        with open(tiff_path, "wb") as f:
            f.write(uploaded_tiff.read())

        st.write("Reprojection du fichier TIFF...")
        try:
            reprojected_tiff = reproject_tiff(tiff_path, "EPSG:4326")
            with rasterio.open(reprojected_tiff) as src:
                bounds = src.bounds
                center_lat = (bounds.top + bounds.bottom) / 2
                center_lon = (bounds.left + bounds.right) / 2
                fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12)

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
                add_image_overlay(fmap, reprojected_tiff, bounds, "Orthophoto")
        except Exception as e:
            st.error(f"Erreur lors de la reprojection : {e}")

    # Téléversement du fichier MNT (Modèle Numérique de Terrain)
    uploaded_mnt = st.file_uploader("Téléverser un fichier MNT (TIFF)", type=["tif", "tiff"])
    if uploaded_mnt:
        mnt_path = uploaded_mnt.name
        with open(mnt_path, "wb") as f:
            f.write(uploaded_mnt.read())

        st.write("Reprojection du fichier MNT...")
        try:
            reprojected_mnt = reproject_tiff(mnt_path, "EPSG:4326")
            
            # Create a temporary PNG file for the colorized DEM
            temp_png_path = "mnt_colored.png"
            apply_color_gradient(reprojected_mnt, temp_png_path)
            
            with rasterio.open(reprojected_mnt) as src:
                bounds = src.bounds
                center_lat = (bounds.top + bounds.bottom) / 2
                center_lon = (bounds.left + bounds.right) / 2
                fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12)
                add_image_overlay(fmap, temp_png_path, bounds, "MNT")
                
            # Remove the temporary PNG file
            os.remove(temp_png_path)
        except Exception as e:
            st.error(f"Erreur lors de la reprojection du MNT : {e}")

    # Téléversement du fichier MNS (Modèle Numérique de Surface)
    uploaded_mns = st.file_uploader("Téléverser un fichier MNS (TIFF)", type=["tif", "tiff"])
    if uploaded_mns:
        mns_path = uploaded_mns.name
        with open(mns_path, "wb") as f:
            f.write(uploaded_mns.read())

        st.write("Reprojection du fichier MNS...")
        try:
            reprojected_mns = reproject_tiff(mns_path, "EPSG:4326")
            
            # Create a temporary PNG file for the colorized MNS
            temp_png_path = "mns_colored.png"
            apply_color_gradient(reprojected_mns, temp_png_path)
            
            with rasterio.open(reprojected_mns) as src:
                bounds = src.bounds
                center_lat = (bounds.top + bounds.bottom) / 2
                center_lon = (bounds.left + bounds.right) / 2
                fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12)
                add_image_overlay(fmap, temp_png_path, bounds, "MNS")
                
            # Remove the temporary PNG file
            os.remove(temp_png_path)
        except Exception as e:
            st.error(f"Erreur lors de la reprojection du MNS : {e}")


    # Téléversement d'un fichier GeoJSON pour les routes
    geojson_file = st.file_uploader("Téléverser un fichier GeoJSON de routes", type=["geojson"])
    if geojson_file:
        try:
            geojson_data = json.load(geojson_file)
            folium.GeoJson(
                geojson_data,
                name="Routes",
                style_function=lambda x: {
                    "color": "orange",  # Change color to orange
                    "weight": 4,
                    "opacity": 0.7
                }
            ).add_to(fmap)
        except Exception as e:
            st.error(f"Erreur lors du chargement du GeoJSON : {e}")

    # Téléversement d'un fichier GeoJSON pour la polygonale
    geojson_polygon = st.file_uploader("Téléverser un fichier GeoJSON de polygonale", type=["geojson"])
    if geojson_polygon:
        try:
            polygon_data = json.load(geojson_polygon)
            folium.GeoJson(
                polygon_data,
                name="Polygonale",
                style_function=lambda x: {
                    "color": "red",  # Border color red
                    "weight": 2,
                    "opacity": 1,
                    "fillColor": "transparent",  # Transparent fill color
                    "fillOpacity": 0.1
                }
            ).add_to(fmap)
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier polygonal : {e}")

    # Ajout des contrôles de calques
    folium.LayerControl().add_to(fmap)

    # Affichage de la carte
    folium_static(fmap, width=700, height=500)


if __name__ == "__main__":
    main()
    


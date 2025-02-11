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
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height,
        })

        # Generate unique output filename
        base_name = os.path.basename(input_tiff)
        reprojected_tiff = os.path.splitext(base_name)[0] + "_reprojected.tif"
        reprojected_path = os.path.join(os.path.dirname(input_tiff), reprojected_tiff)

        with rasterio.open(reprojected_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest,
                )
    return reprojected_path

# Function to apply color gradient to a DEM TIFF
def apply_color_gradient(tiff_path, output_path):
    """Apply a color gradient to the DEM TIFF and save it as a PNG."""
    with rasterio.open(tiff_path) as src:
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

# Function to calculate bounds from GeoJSON
def calculate_geojson_bounds(geojson_data):
    """Calculate bounds from a GeoJSON object."""
    geometries = [feature["geometry"] for feature in geojson_data["features"]]
    gdf = gpd.GeoDataFrame.from_features(geojson_data)
    return gdf.total_bounds  # Returns [minx, miny, maxx, maxy]

# Main application
def main():
    st.title("DESSINER une CARTE")

    # Initialize session state
    if "drawings" not in st.session_state:
        st.session_state["drawings"] = {
            "type": "FeatureCollection",
            "features": [],
        }
    if "uploaded_layers" not in st.session_state:
        st.session_state["uploaded_layers"] = []
    if "auto_bounds" not in st.session_state:
        st.session_state["auto_bounds"] = []

    # Initialize map
    fmap = folium.Map(location=[0, 0], zoom_start=2)
    
    # Chargement automatique des TIFF du dossier TIFF
    tiff_dir = "TIFF"
    if os.path.exists(tiff_dir) and os.path.isdir(tiff_dir):
        tiff_files = [f for f in os.listdir(tiff_dir) 
                     if f.lower().endswith(('.tif', '.tiff')) 
                     and '_reprojected' not in f]
        
        for tiff_file in tiff_files:
            try:
                tiff_path = os.path.join(tiff_dir, tiff_file)
                reprojected_tiff = reproject_tiff(tiff_path, "EPSG:4326")
                
                # Apply color gradient
                temp_png_path = f"elevation_colored_{tiff_file}.png"
                apply_color_gradient(reprojected_tiff, temp_png_path)
                
                # Get bounds
                with rasterio.open(reprojected_tiff) as src:
                    bounds = src.bounds
                    image_bounds = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
                
                # Add to map
                folium.raster_layers.ImageOverlay(
                    image=temp_png_path,
                    bounds=image_bounds,
                    name="Élévation",
                    opacity=0.6,
                ).add_to(fmap)
                
                st.session_state["auto_bounds"].append(image_bounds)
                
                # Cleanup
                os.remove(temp_png_path)
                os.remove(reprojected_tiff)
                
            except Exception as e:
                st.error(f"Erreur lors du traitement du fichier {tiff_file}: {e}")

    # Adjust map view for auto layers
    if st.session_state["auto_bounds"]:
        fmap.fit_bounds(st.session_state["auto_bounds"])

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

    # Interface de téléversement manuel
    tiff_type = st.selectbox("Sélectionnez le type de fichier TIFF", ["MNT", "MNS", "Orthophoto"], key="tiff_selectbox")
    uploaded_tiff = st.file_uploader(f"Téléverser un fichier TIFF ({tiff_type})", type=["tif", "tiff"], key="tiff_uploader")
    
    if uploaded_tiff:
        tiff_path = uploaded_tiff.name
        with open(tiff_path, "wb") as f:
            f.write(uploaded_tiff.read())

        try:
            reprojected_tiff = reproject_tiff(tiff_path, "EPSG:4326")
            with rasterio.open(reprojected_tiff) as src:
                bounds = src.bounds
                
                # Update map location
                center_lat = (bounds.top + bounds.bottom) / 2
                center_lon = (bounds.left + bounds.right) / 2
                fmap.location = [center_lat, center_lon]

                # Check existing layers
                layer_exists = any(
                    layer["type"] == "TIFF" and layer["name"] == tiff_type
                    for layer in st.session_state["uploaded_layers"]
                )

                if not layer_exists:
                    st.session_state["uploaded_layers"].append({
                        "type": "TIFF",
                        "name": tiff_type,
                        "path": reprojected_tiff,
                        "bounds": bounds
                    })
                    st.success(f"Couche {tiff_type} ajoutée!")
                else:
                    st.warning("Couche déjà existante!")

        except Exception as e:
            st.error(f"Erreur de traitement: {e}")

    # Gestion des GeoJSON
    geojson_type = st.selectbox("Sélectionnez le type de fichier GeoJSON", ["Routes", "Polygonale", "Cours d'eau"], key="geojson_selectbox")
    uploaded_geojson = st.file_uploader(f"Téléverser un fichier GeoJSON ({geojson_type})", type=["geojson"], key="geojson_uploader")
    
    if uploaded_geojson:
        try:
            geojson_data = json.load(uploaded_geojson)
            layer_exists = any(
                layer["type"] == "GeoJSON" and layer["name"] == geojson_type
                for layer in st.session_state["uploaded_layers"]
            )

            if not layer_exists:
                st.session_state["uploaded_layers"].append({
                    "type": "GeoJSON",
                    "name": geojson_type,
                    "data": geojson_data
                })
                st.success(f"Couche {geojson_type} ajoutée!")
            else:
                st.warning("Couche déjà existante!")

        except Exception as e:
            st.error(f"Erreur de chargement: {e}")

    # Affichage des couches
    if st.button("Afficher toutes les couches"):
        all_bounds = st.session_state["auto_bounds"].copy()
        
        # Add uploaded layers
        for layer in st.session_state["uploaded_layers"]:
            if layer["type"] == "TIFF":
                if layer["name"] in ["MNT", "MNS"]:
                    temp_png_path = f"{layer['name']}_colored.png"
                    apply_color_gradient(layer["path"], temp_png_path)
                    add_image_overlay(fmap, temp_png_path, layer["bounds"], layer["name"])
                    os.remove(temp_png_path)
                else:
                    add_image_overlay(fmap, layer["path"], layer["bounds"], layer["name"])
                
                all_bounds.append([
                    [layer["bounds"].bottom, layer["bounds"].left],
                    [layer["bounds"].top, layer["bounds"].right]
                ])
                
            elif layer["type"] == "GeoJSON":
                color = "orange" if layer["name"] == "Routes" else "blue"
                folium.GeoJson(
                    layer["data"],
                    name=layer["name"],
                    style_function=lambda x, color=color: {"color": color, "weight": 4}
                ).add_to(fmap)
                
                geojson_bounds = calculate_geojson_bounds(layer["data"])
                all_bounds.append([
                    [geojson_bounds[1], geojson_bounds[0]],
                    [geojson_bounds[3], geojson_bounds[2]]
                ])

        if all_bounds:
            fmap.fit_bounds(all_bounds)

    # Contrôle des calques
    folium.LayerControl().add_to(fmap)
    folium_static(fmap, width=700, height=500)

if __name__ == "__main__":
    main()

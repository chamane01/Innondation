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

# Function to calculate bounds from GeoJSON
def calculate_geojson_bounds(geojson_data):
    """Calculate bounds from a GeoJSON object."""
    geometries = [feature["geometry"] for feature in geojson_data["features"]]
    gdf = gpd.GeoDataFrame.from_features(geojson_data)
    return gdf.total_bounds  # Returns [minx, miny, maxx, maxy]

# Dictionnaire des couleurs pour les types de fichiers GeoJSON
geojson_colors = {
    "Routes": "orange",
    "Pistes": "brown",
    "Plantations": "green",
    "Bâtiments": "gray",
    "Électricité": "yellow",
    "Assainissements": "blue",
    "Villages": "purple",
    "Villes": "red",
    "Chemin de fer": "black",
    "Parc et réserves": "darkgreen",
    "Cours d'eau": "lightblue",
    "Polygonale": "pink"
}

# Main application
def main():
    st.title("DESSINER une CARTE ")

    # Initialize session state for drawings and uploaded layers
    if "drawings" not in st.session_state:
        st.session_state["drawings"] = {
            "type": "FeatureCollection",
            "features": [],
        }
    if "uploaded_layers" not in st.session_state:
        st.session_state["uploaded_layers"] = []

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

    # Single button for TIFF upload with type selection
    tiff_type = st.selectbox(
        "Sélectionnez le type de fichier TIFF",
        options=["MNT", "MNS", "Orthophoto"],
        index=None,  # Aucune option sélectionnée par défaut
        placeholder="Veuillez sélectionner",
        key="tiff_selectbox"
    )

    if tiff_type:
        uploaded_tiff = st.file_uploader(f"Téléverser un fichier TIFF ({tiff_type})", type=["tif", "tiff"], key="tiff_uploader")

        if uploaded_tiff:
            tiff_path = uploaded_tiff.name
            with open(tiff_path, "wb") as f:
                f.write(uploaded_tiff.read())

            st.write(f"Reprojection du fichier TIFF ({tiff_type})...")
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

                    # Bouton pour ajouter le fichier TIFF à la liste des couches
                    if st.button(f"Ajouter {tiff_type} à la liste de couches", key=f"add_tiff_{tiff_type}"):
                        # Check if the layer already exists in the list
                        layer_exists = any(
                            layer["type"] == "TIFF" and layer["name"] == tiff_type and layer["path"] == reprojected_tiff
                            for layer in st.session_state["uploaded_layers"]
                        )

                        if not layer_exists:
                            # Store the layer in the uploaded_layers list
                            st.session_state["uploaded_layers"].append({"type": "TIFF", "name": tiff_type, "path": reprojected_tiff, "bounds": bounds})
                            st.success(f"Couche {tiff_type} ajoutée à la liste des couches.")
                        else:
                            st.warning(f"La couche {tiff_type} existe déjà dans la liste.")
            except Exception as e:
                st.error(f"Erreur lors de la reprojection : {e}")

    # Single button for GeoJSON upload with type selection
    geojson_type = st.selectbox(
        "Sélectionnez le type de fichier GeoJSON",
        options=[
            "Polygonale",
            "Routes",
            "Cours d'eau",
            "Bâtiments",
            "Pistes",
            "Plantations",
            "Électricité",
            "Assainissements",
            "Villages",
            "Villes",
            "Chemin de fer",
            "Parc et réserves" 
        ],
        index=None,  # Aucune option sélectionnée par défaut
        placeholder="Veuillez sélectionner",
        key="geojson_selectbox"
    )

    if geojson_type:
        uploaded_geojson = st.file_uploader(f"Téléverser un fichier GeoJSON ({geojson_type})", type=["geojson"], key="geojson_uploader")

        if uploaded_geojson:
            try:
                geojson_data = json.load(uploaded_geojson)
                # Bouton pour ajouter le fichier GeoJSON à la liste des couches
                if st.button(f"Ajouter {geojson_type} à la liste de couches", key=f"add_geojson_{geojson_type}"):
                    # Check if the layer already exists in the list
                    layer_exists = any(
                        layer["type"] == "GeoJSON" and layer["name"] == geojson_type and layer["data"] == geojson_data
                        for layer in st.session_state["uploaded_layers"]
                    )

                    if not layer_exists:
                        # Store the layer in the uploaded_layers list
                        st.session_state["uploaded_layers"].append({"type": "GeoJSON", "name": geojson_type, "data": geojson_data})
                        st.success(f"Couche {geojson_type} ajoutée à la liste des couches.")
                    else:
                        st.warning(f"La couche {geojson_type} existe déjà dans la liste.")
            except Exception as e:
                st.error(f"Erreur lors du chargement du GeoJSON : {e}")

    # Display the list of uploaded layers with delete buttons
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader("Liste des couches téléversées")
    with col2:
        if st.button("Rafraîchir la liste", key="refresh_list"):
            # No need to call st.experimental_rerun(), Streamlit will automatically re-run the script
            pass

    if st.session_state["uploaded_layers"]:
        for i, layer in enumerate(st.session_state["uploaded_layers"]):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"{i + 1}. {layer['name']} ({layer['type']})")
            with col2:
                if st.button(f"Supprimer {layer['name']}", key=f"delete_{i}"):
                    # Remove the layer from the list
                    st.session_state["uploaded_layers"].pop(i)
                    st.success(f"Couche {layer['name']} supprimée.")
                    # Streamlit will automatically re-run the script
    else:
        st.write("Aucune couche téléversée pour le moment.")

    # Button to add all uploaded layers to the map
    if st.button("Ajouter la liste de couches à la carte", key="add_layers_button"):
        # Use a set to track added layers and avoid duplicates
        added_layers = set()
        all_bounds = []  # To store bounds of all layers

        for layer in st.session_state["uploaded_layers"]:
            if layer["name"] not in added_layers:
                if layer["type"] == "TIFF":
                    if layer["name"] in ["MNT", "MNS"]:
                        temp_png_path = f"{layer['name'].lower()}_colored.png"
                        apply_color_gradient(layer["path"], temp_png_path)
                        add_image_overlay(fmap, temp_png_path, layer["bounds"], layer["name"])
                        os.remove(temp_png_path)
                    else:
                        add_image_overlay(fmap, layer["path"], layer["bounds"], layer["name"])
                    # Add bounds to the list
                    all_bounds.append([[layer["bounds"].bottom, layer["bounds"].left], [layer["bounds"].top, layer["bounds"].right]])
                elif layer["type"] == "GeoJSON":
                    # Get the color for the GeoJSON layer
                    color = geojson_colors.get(layer["name"], "blue")  # Default to blue if not found
                    folium.GeoJson(
                        layer["data"],
                        name=layer["name"],
                        style_function=lambda x, color=color: {
                            "color": color,
                            "weight": 4,
                            "opacity": 0.7
                        }
                    ).add_to(fmap)
                    # Calculate bounds for GeoJSON and add to the list
                    geojson_bounds = calculate_geojson_bounds(layer["data"])
                    all_bounds.append([[geojson_bounds[1], geojson_bounds[0]], [geojson_bounds[3], geojson_bounds[2]]])
                added_layers.add(layer["name"])

        # Adjust the map view to fit all bounds
        if all_bounds:
            fmap.fit_bounds(all_bounds)
        st.success("Toutes les couches ont été ajoutées à la carte.")

    # Ajout des contrôles de calques
    folium.LayerControl().add_to(fmap)

    # Affichage de la carte
    folium_static(fmap, width=700, height=500)

if __name__ == "__main__":
    main()

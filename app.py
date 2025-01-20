import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw, MeasureControl
from folium import LayerControl
import rasterio
import rasterio.warp
from rasterio.plot import reshape_as_image
from PIL import Image
from rasterio.warp import transform_bounds
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point, LineString
import json
from io import BytesIO
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import matplotlib.pyplot as plt
import os

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

# Fonction pour reprojeter un fichier TIFF
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

# Fonction pour appliquer un gradient de couleur à un MNT/MNS
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

# Fonction pour ajouter une image TIFF à la carte
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

# Fonction pour calculer les limites d'un GeoJSON
def calculate_geojson_bounds(geojson_data):
    """Calculate bounds from a GeoJSON object."""
    geometries = [feature["geometry"] for feature in geojson_data["features"]]
    gdf = gpd.GeoDataFrame.from_features(geojson_data)
    return gdf.total_bounds  # Returns [minx, miny, maxx, maxy]

# Initialisation des couches et des entités dans la session Streamlit
if "layers" not in st.session_state:
    st.session_state["layers"] = {"Routes": [], "Bâtiments": [], "Polygonale": [], "MNT": [], "MNS": [], "Orthophotos": []}

if "new_features" not in st.session_state:
    st.session_state["new_features"] = []

if "uploaded_layers" not in st.session_state:
    st.session_state["uploaded_layers"] = []

# Titre de l'application
st.title("Carte Dynamique avec Gestion Avancée des Couches")

# Description
st.markdown("""
Créez des entités géographiques (points, lignes, polygones) en les dessinant sur la carte et ajoutez-les à des couches spécifiques. 
Vous pouvez également activer ou désactiver des couches grâce au gestionnaire de couches.
""")

# Carte de base
m = folium.Map(location=[5.5, -4.0], zoom_start=8)

# Sidebar pour la gestion des couches
with st.sidebar:
    st.header("Gestion des Couches")

    # Sous-titre 1 : Téléverser des fichiers
    st.subheader("1. Téléverser des fichiers")
    tiff_type = st.selectbox(
        "Sélectionnez le type de fichier TIFF",
        options=["MNT", "MNS", "Orthophoto"],
        index=None,
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
                    m.location = [center_lat, center_lon]
                    m.zoom_start = 12

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
        index=None,
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

    # Liste des couches téléversées
    st.markdown("### Liste des couches téléversées")
    
    if st.session_state["uploaded_layers"]:
        for i, layer in enumerate(st.session_state["uploaded_layers"]):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"{i + 1}. {layer['name']} ({layer['type']})")
            with col2:
                # Bouton de suppression en rouge
                if st.button("🗑️", key=f"delete_{i}_{layer['name']}", help="Supprimer cette couche"):
                    st.session_state["uploaded_layers"].pop(i)
                    st.success(f"Couche {layer['name']} supprimée.")
    else:
        st.write("Aucune couche téléversée pour le moment.")

    # Bouton pour ajouter toutes les couches à la carte
    if st.button("Ajouter la liste de couches à la carte", key="add_layers_button"):
        added_layers = set()
        all_bounds = []  # Pour stocker les limites de toutes les couches

        for layer in st.session_state["uploaded_layers"]:
            if layer["name"] not in added_layers:
                if layer["type"] == "TIFF":
                    if layer["name"] in ["MNT", "MNS"]:
                        temp_png_path = f"{layer['name'].lower()}_colored.png"
                        apply_color_gradient(layer["path"], temp_png_path)
                        add_image_overlay(m, temp_png_path, layer["bounds"], layer["name"])
                        os.remove(temp_png_path)
                    else:
                        add_image_overlay(m, layer["path"], layer["bounds"], layer["name"])
                    all_bounds.append([[layer["bounds"].bottom, layer["bounds"].left], [layer["bounds"].top, layer["bounds"].right]])
                elif layer["type"] == "GeoJSON":
                    color = geojson_colors.get(layer["name"], "blue")
                    folium.GeoJson(
                        layer["data"],
                        name=layer["name"],
                        style_function=lambda x, color=color: {
                            "color": color,
                            "weight": 4,
                            "opacity": 0.7
                        }
                    ).add_to(m)
                    geojson_bounds = calculate_geojson_bounds(layer["data"])
                    all_bounds.append([[geojson_bounds[1], geojson_bounds[0]], [geojson_bounds[3], geojson_bounds[2]]])
                added_layers.add(layer["name"])

        # Ajuster la vue de la carte pour inclure toutes les limites
        if all_bounds:
            m.fit_bounds(all_bounds)
        st.success("Toutes les couches ont été ajoutées à la carte.")

    # Espacement entre les sections
    st.markdown("---")

    # Sous-titre 2 : Ajouter une nouvelle couche
    st.subheader("2. Ajouter une nouvelle couche")
    new_layer_name = st.text_input("Nom de la nouvelle couche à ajouter", "")
    if st.button("Ajouter la couche", key="add_new_layer_button") and new_layer_name:
        if new_layer_name not in st.session_state["layers"]:
            st.session_state["layers"][new_layer_name] = []
            st.success(f"La couche '{new_layer_name}' a été ajoutée.")
        else:
            st.warning(f"La couche '{new_layer_name}' existe déjà.")

    # Sélection de la couche active pour ajouter les nouvelles entités
    st.subheader("Sélectionner une couche active")
    layer_name = st.selectbox(
        "Choisissez la couche à laquelle ajouter les entités",
        list(st.session_state["layers"].keys()),
        key="layer_selectbox"
    )

    # Affichage des entités temporairement dessinées
    if st.session_state["new_features"]:
        st.write(f"**Entités dessinées temporairement ({len(st.session_state['new_features'])}) :**")
        for idx, feature in enumerate(st.session_state["new_features"]):
            st.write(f"- Entité {idx + 1}: {feature['geometry']['type']}")

    # Bouton pour enregistrer les nouvelles entités dans la couche active
    if st.button("Enregistrer les entités", key="save_features_button"):
        # Ajouter les entités non dupliquées à la couche sélectionnée
        current_layer = st.session_state["layers"][layer_name]
        for feature in st.session_state["new_features"]:
            if feature not in current_layer:
                current_layer.append(feature)
        st.session_state["new_features"] = []  # Réinitialisation des entités temporaires
        st.success(f"Toutes les nouvelles entités ont été enregistrées dans la couche '{layer_name}'.")

    # Suppression et modification d'une entité dans une couche
    st.subheader("Gestion des entités dans les couches")
    selected_layer = st.selectbox("Choisissez une couche pour voir ses entités", list(st.session_state["layers"].keys()), key="selected_layer_selectbox")
    if st.session_state["layers"][selected_layer]:
        entity_idx = st.selectbox(
            "Sélectionnez une entité à gérer",
            range(len(st.session_state["layers"][selected_layer])),
            format_func=lambda idx: f"Entité {idx + 1}: {st.session_state['layers'][selected_layer][idx]['geometry']['type']}",
            key="entity_selectbox"
        )
        selected_entity = st.session_state["layers"][selected_layer][entity_idx]
        current_name = selected_entity.get("properties", {}).get("name", "")
        new_name = st.text_input("Nom de l'entité", current_name, key="entity_name_input")

        if st.button("Modifier le nom", key=f"edit_{entity_idx}"):
            if "properties" not in selected_entity:
                selected_entity["properties"] = {}
            selected_entity["properties"]["name"] = new_name
            st.success(f"Le nom de l'entité a été mis à jour en '{new_name}'.")

        if st.button("Supprimer l'entité sélectionnée", key=f"delete_{entity_idx}"):
            st.session_state["layers"][selected_layer].pop(entity_idx)
            st.success(f"L'entité sélectionnée a été supprimée de la couche '{selected_layer}'.")
    else:
        st.write("Aucune entité dans cette couche pour le moment.")

# Gestionnaire de dessin
draw = Draw(
    draw_options={
        "polyline": True,
        "polygon": True,
        "circle": False,
        "rectangle": True,
        "marker": True,
        "circlemarker": False,
    },
    edit_options={"edit": True, "remove": True},
)
draw.add_to(m)

# Ajout du gestionnaire de couches en mode plié
LayerControl(position="topleft", collapsed=True).add_to(m)

# Affichage interactif de la carte
output = st_folium(m, width=800, height=600, returned_objects=["last_active_drawing", "all_drawings"])

# Gestion des nouveaux dessins
if output and "last_active_drawing" in output and output["last_active_drawing"]:
    new_feature = output["last_active_drawing"]
    # Ajouter l'entité temporairement si elle n'existe pas déjà
    if new_feature not in st.session_state["new_features"]:
        st.session_state["new_features"].append(new_feature)
        st.info("Nouvelle entité ajoutée temporairement. Cliquez sur 'Enregistrer les entités' pour les ajouter à la couche.")

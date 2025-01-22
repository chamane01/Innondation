import streamlit as st
from streamlit_folium import st_folium, folium_static
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
from shapely.geometry import Polygon, Point, LineString, shape
import json
from io import BytesIO
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import matplotlib.pyplot as plt
import os
import uuid  # Pour générer des identifiants uniques

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

# Fonction pour reprojeter un fichier TIFF avec un nom unique
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

        # Générer un nom de fichier unique
        unique_id = str(uuid.uuid4())[:8]  # Utilisation des 8 premiers caractères d'un UUID
        reprojected_tiff = f"reprojected_{unique_id}.tiff"
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

# Fonction pour charger un fichier TIFF
def load_tiff(tiff_path):
    """Charge un fichier TIFF et retourne les données et les bornes."""
    try:
        with rasterio.open(tiff_path) as src:
            data = src.read(1)
            bounds = src.bounds
        return data, bounds
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier TIFF : {e}")
        return None, None

# Fonction pour calculer le volume avec MNS et MNT
def calculate_volume_in_polygon(mns, mnt, bounds, polygons_gdf):
    """Calcule le volume entre un MNS et un MNT dans une polygonale."""
    try:
        # Masquer les données en dehors de la polygonale
        mask = polygons_gdf.geometry
        mns_masked = np.where(mask, mns, np.nan)
        mnt_masked = np.where(mask, mnt, np.nan)

        # Calculer la différence entre MNS et MNT
        volume = np.nansum(mns_masked - mnt_masked) * (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) / (mns.shape[0] * mns.shape[1])
        return volume
    except Exception as e:
        st.error(f"Erreur lors du calcul du volume : {e}")
        return None

# Fonction pour calculer le volume avec MNS seul
def calculate_volume_without_mnt(mns, bounds, polygons_gdf, reference_altitude):
    """Calcule le volume avec un MNS seul et une altitude de référence."""
    try:
        # Masquer les données en dehors de la polygonale
        mask = polygons_gdf.geometry
        mns_masked = np.where(mask, mns, np.nan)

        # Calculer les volumes positif et négatif
        positive_volume = np.nansum(np.where(mns_masked > reference_altitude, mns_masked - reference_altitude, 0)) * (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) / (mns.shape[0] * mns.shape[1])
        negative_volume = np.nansum(np.where(mns_masked < reference_altitude, reference_altitude - mns_masked, 0)) * (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) / (mns.shape[0] * mns.shape[1])
        real_volume = positive_volume - negative_volume
        return positive_volume, negative_volume, real_volume
    except Exception as e:
        st.error(f"Erreur lors du calcul du volume : {e}")
        return None, None, None

# Fonction pour convertir les entités dessinées en GeoDataFrame
def convert_drawn_features_to_gdf(features):
    """Convertit les entités dessinées en GeoDataFrame."""
    geometries = []
    for feature in features:
        geom = shape(feature["geometry"])
        geometries.append(geom)
    gdf = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")
    return gdf

# Fonction pour vérifier si une couche contient des polygones
def find_polygons_in_layers(layers):
    """Recherche des polygones dans les couches téléversées."""
    polygons = []
    for layer in layers:
        if layer["type"] == "GeoJSON":
            geojson_data = layer["data"]
            for feature in geojson_data["features"]:
                if feature["geometry"]["type"] == "Polygon":
                    polygons.append(feature)
    return polygons

# Fonction pour convertir les polygones en GeoDataFrame
def convert_polygons_to_gdf(polygons):
    """Convertit une liste de polygones en GeoDataFrame."""
    geometries = [shape(polygon["geometry"]) for polygon in polygons]
    gdf = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")
    return gdf

# Initialisation des couches et des entités dans la session Streamlit
if "layers" not in st.session_state:
    st.session_state["layers"] = {}  # Couches créées par l'utilisateur

if "uploaded_layers" not in st.session_state:
    st.session_state["uploaded_layers"] = []  # Couches téléversées (TIFF et GeoJSON)

if "new_features" not in st.session_state:
    st.session_state["new_features"] = []  # Entités temporairement dessinées

# Titre de l'application
st.title("Carte Topographique et Analyse Spatiale")

# Description
st.markdown("""
Créez des entités géographiques (points, lignes, polygones) en les dessinant sur la carte et ajoutez-les à des couches spécifiques. 
Vous pouvez également téléverser des fichiers TIFF ou GeoJSON pour les superposer à la carte.
""")

# Sidebar pour la gestion des couches
with st.sidebar:
    st.header("Gestion des Couches")

    # Section 1: Ajout d'une nouvelle couche
    st.markdown("### 1- Ajouter une nouvelle couche")
    new_layer_name = st.text_input("Nom de la nouvelle couche à ajouter", "")
    if st.button("Ajouter la couche", key="add_layer_button", help="Ajouter une nouvelle couche", type="primary") and new_layer_name:
        if new_layer_name not in st.session_state["layers"]:
            st.session_state["layers"][new_layer_name] = []
            st.success(f"La couche '{new_layer_name}' a été ajoutée.")
        else:
            st.warning(f"La couche '{new_layer_name}' existe déjà.")

    # Sélection de la couche active pour ajouter les nouvelles entités
    st.markdown("#### Sélectionner une couche active")
    if st.session_state["layers"]:
        layer_name = st.selectbox(
            "Choisissez la couche à laquelle ajouter les entités",
            list(st.session_state["layers"].keys())
        )
    else:
        st.write("Aucune couche disponible. Ajoutez une couche pour commencer.")

    # Affichage des entités temporairement dessinées
    if st.session_state["new_features"]:
        st.write(f"**Entités dessinées temporairement ({len(st.session_state['new_features'])}) :**")
        for idx, feature in enumerate(st.session_state["new_features"]):
            st.write(f"- Entité {idx + 1}: {feature['geometry']['type']}")

    # Bouton pour enregistrer les nouvelles entités dans la couche active
    if st.button("Enregistrer les entités", key="save_features_button", type="primary") and st.session_state["layers"]:
        # Ajouter les entités non dupliquées à la couche sélectionnée
        current_layer = st.session_state["layers"][layer_name]
        for feature in st.session_state["new_features"]:
            if feature not in current_layer:
                current_layer.append(feature)
        st.session_state["new_features"] = []  # Réinitialisation des entités temporaires
        st.success(f"Toutes les nouvelles entités ont été enregistrées dans la couche '{layer_name}'.")

    # Gestion des entités dans les couches
    st.markdown("#### Gestion des entités dans les couches")
    if st.session_state["layers"]:
        selected_layer = st.selectbox("Choisissez une couche pour voir ses entités", list(st.session_state["layers"].keys()))
        if st.session_state["layers"][selected_layer]:
            entity_idx = st.selectbox(
                "Sélectionnez une entité à gérer",
                range(len(st.session_state["layers"][selected_layer])),
                format_func=lambda idx: f"Entité {idx + 1}: {st.session_state['layers'][selected_layer][idx]['geometry']['type']}"
            )
            selected_entity = st.session_state["layers"][selected_layer][entity_idx]
            current_name = selected_entity.get("properties", {}).get("name", "")
            new_name = st.text_input("Nom de l'entité", current_name)

            if st.button("Modifier le nom", key=f"edit_{entity_idx}", type="primary"):
                if "properties" not in selected_entity:
                    selected_entity["properties"] = {}
                selected_entity["properties"]["name"] = new_name
                st.success(f"Le nom de l'entité a été mis à jour en '{new_name}'.")

            if st.button("Supprimer l'entité sélectionnée", key=f"delete_{entity_idx}", type="secondary"):
                st.session_state["layers"][selected_layer].pop(entity_idx)
                st.success(f"L'entité sélectionnée a été supprimée de la couche '{selected_layer}'.")
        else:
            st.write("Aucune entité dans cette couche pour le moment.")
    else:
        st.write("Aucune couche disponible pour gérer les entités.")

    # Démarcation claire entre 1- et 2-
    st.markdown("---")

    # Section 2: Téléversement de fichiers
    st.markdown("### 2- Téléverser des fichiers")
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
            # Générer un nom de fichier unique pour le fichier téléversé
            unique_id = str(uuid.uuid4())[:8]
            tiff_path = f"uploaded_{unique_id}.tiff"
            with open(tiff_path, "wb") as f:
                f.write(uploaded_tiff.read())

            st.write(f"Reprojection du fichier TIFF ({tiff_type})...")
            try:
                reprojected_tiff = reproject_tiff(tiff_path, "EPSG:4326")
                with rasterio.open(reprojected_tiff) as src:
                    bounds = src.bounds
                    # Vérifier si la couche existe déjà
                    if not any(layer["name"] == tiff_type and layer["type"] == "TIFF" for layer in st.session_state["uploaded_layers"]):
                        st.session_state["uploaded_layers"].append({"type": "TIFF", "name": tiff_type, "path": reprojected_tiff, "bounds": bounds})
                        st.success(f"Couche {tiff_type} ajoutée à la liste des couches.")
                    else:
                        st.warning(f"La couche {tiff_type} existe déjà.")
            except Exception as e:
                st.error(f"Erreur lors de la reprojection : {e}")
            finally:
                # Supprimer le fichier temporaire après utilisation
                os.remove(tiff_path)

    geojson_type = st.selectbox(
        "Sélectionnez le type de fichier GeoJSON",
        options=[
            "Polygonale", "Routes", "Cours d'eau", "Bâtiments", "Pistes", "Plantations",
            "Électricité", "Assainissements", "Villages", "Villes", "Chemin de fer", "Parc et réserves"
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
                # Vérifier si la couche existe déjà
                if not any(layer["name"] == geojson_type and layer["type"] == "GeoJSON" for layer in st.session_state["uploaded_layers"]):
                    st.session_state["uploaded_layers"].append({"type": "GeoJSON", "name": geojson_type, "data": geojson_data})
                    st.success(f"Couche {geojson_type} ajoutée à la liste des couches.")
                else:
                    st.warning(f"La couche {geojson_type} existe déjà.")
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
                if st.button("🗑️", key=f"delete_{i}_{layer['name']}", help="Supprimer cette couche", type="secondary"):
                    st.session_state["uploaded_layers"].pop(i)
                    st.success(f"Couche {layer['name']} supprimée.")
    else:
        st.write("Aucune couche téléversée pour le moment.")

# Carte de base
m = folium.Map(location=[7.5399, -5.5471], zoom_start=6)  # Centré sur la Côte d'Ivoire avec un zoom adapté

# Ajout des fonds de carte
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="Satellite",
).add_to(m)

folium.TileLayer(
    tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
    attr="OpenTopoMap",
    name="Topographique",
).add_to(m)  # Carte topographique ajoutée en dernier pour être la carte par défaut

# Ajout des couches créées à la carte
for layer, features in st.session_state["layers"].items():
    layer_group = folium.FeatureGroup(name=layer, show=True)
    for feature in features:
        feature_type = feature["geometry"]["type"]
        coordinates = feature["geometry"]["coordinates"]
        popup = feature.get("properties", {}).get("name", f"{layer} - Entité")

        if feature_type == "Point":
            lat, lon = coordinates[1], coordinates[0]
            folium.Marker(location=[lat, lon], popup=popup).add_to(layer_group)
        elif feature_type == "LineString":
            folium.PolyLine(locations=[(lat, lon) for lon, lat in coordinates], color="blue", popup=popup).add_to(layer_group)
        elif feature_type == "Polygon":
            folium.Polygon(locations=[(lat, lon) for lon, lat in coordinates[0]], color="green", fill=True, popup=popup).add_to(layer_group)
    layer_group.add_to(m)

# Ajout des couches téléversées à la carte
for layer in st.session_state["uploaded_layers"]:
    if layer["type"] == "TIFF":
        if layer["name"] in ["MNT", "MNS"]:
            # Générer un nom de fichier unique pour l'image colorée
            unique_id = str(uuid.uuid4())[:8]
            temp_png_path = f"{layer['name'].lower()}_colored_{unique_id}.png"
            apply_color_gradient(layer["path"], temp_png_path)
            add_image_overlay(m, temp_png_path, layer["bounds"], layer["name"])
            os.remove(temp_png_path)  # Supprimer le fichier PNG temporaire
        else:
            add_image_overlay(m, layer["path"], layer["bounds"], layer["name"])
        
        # Ajuster la vue de la carte pour inclure l'image TIFF
        bounds = [[layer["bounds"].bottom, layer["bounds"].left], [layer["bounds"].top, layer["bounds"].right]]
        m.fit_bounds(bounds)
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

# Ajout du contrôle des couches pour basculer entre les fonds de carte
LayerControl(position="topleft", collapsed=True).add_to(m)

# Affichage interactif de la carte
output = st_folium(m, width=800, height=600, returned_objects=["last_active_drawing", "all_drawings"])

# Gestion des nouveaux dessins
if output and "last_active_drawing" in output and output["last_active_drawing"]:
    new_feature = output["last_active_drawing"]
    if new_feature not in st.session_state["new_features"]:
        st.session_state["new_features"].append(new_feature)
        st.info("Nouvelle entité ajoutée temporairement. Cliquez sur 'Enregistrer les entités' pour les ajouter à la couche.")

# Initialisation de l'état de session pour le bouton actif
if 'active_button' not in st.session_state:
    st.session_state['active_button'] = None

# Fonction pour afficher les paramètres en fonction du bouton cliqué
def display_parameters(button_name):
    if button_name == "Surfaces et volumes":
        st.markdown("### Calcul des volumes")
        method = st.radio(
            "Choisissez la méthode de calcul :",
            ("Méthode 1 : MNS - MNT", "Méthode 2 : MNS seul"),
            key="volume_method"
        )

        # Récupérer les couches nécessaires
        mns_layer = next((layer for layer in st.session_state["uploaded_layers"] if layer["name"] == "MNS"), None)
        mnt_layer = next((layer for layer in st.session_state["uploaded_layers"] if layer["name"] == "MNT"), None)

        if not mns_layer:
            st.error("La couche MNS est manquante. Veuillez téléverser un fichier MNS.")
            return
        if method == "Méthode 1 : MNS - MNT" and not mnt_layer:
            st.error("La couche MNT est manquante. Veuillez téléverser un fichier MNT.")
            return

        # Charger les données
        mns, mns_bounds = load_tiff(mns_layer["path"])
        if method == "Méthode 1 : MNS - MNT":
            mnt, mnt_bounds = load_tiff(mnt_layer["path"])

        # Récupérer les polygones des couches ou des dessins
        polygons = find_polygons_in_layers(st.session_state["uploaded_layers"])
        if polygons:
            polygons_gdf = convert_polygons_to_gdf(polygons)
        elif st.session_state["new_features"]:
            polygons_gdf = convert_drawn_features_to_gdf(st.session_state["new_features"])
        else:
            st.error("Aucune polygonale disponible. Veuillez dessiner ou téléverser une polygonale.")
            return

        if mns is None or polygons_gdf is None:
            st.error("Erreur lors du chargement des fichiers.")
            return

        try:
            if method == "Méthode 1 : MNS - MNT":
                if mnt is None or mnt_bounds != mns_bounds:
                    st.error("Les fichiers doivent avoir les mêmes bornes géographiques.")
                else:
                    volume = calculate_volume_in_polygon(mns, mnt, mnt_bounds, polygons_gdf)
                    st.write(f"Volume calculé dans la polygonale : {volume:.2f} m³")
            else:
                # Saisie de l'altitude de référence pour la méthode 2
                reference_altitude = st.number_input(
                    "Entrez l'altitude de référence (en mètres) :",
                    value=0.0,
                    step=0.1,
                    key="reference_altitude"
                )
                positive_volume, negative_volume, real_volume = calculate_volume_without_mnt(
                    mns, mns_bounds, polygons_gdf, reference_altitude
                )
                st.write(f"Volume positif (au-dessus de la référence) : {positive_volume:.2f} m³")
                st.write(f"Volume négatif (en dessous de la référence) : {negative_volume:.2f} m³")
                st.write(f"Volume réel (différence) : {real_volume:.2f} m³")

        except Exception as e:
            st.error(f"Erreur lors du calcul du volume : {e}")

# Ajout des boutons pour les analyses spatiales
st.markdown("### Analyse Spatiale")
col1, col2, col3 = st.columns(3)

# Boutons principaux
with col1:
    if st.button("Surfaces et volumes", key="surfaces_volumes"):
        st.session_state['active_button'] = "Surfaces et volumes"
    if st.button("Carte de contours", key="contours"):
        st.session_state['active_button'] = "Carte de contours"

with col2:
    if st.button("Trouver un point", key="trouver_point"):
        st.session_state['active_button'] = "Trouver un point"
    if st.button("Générer un rapport", key="generer_rapport"):
        st.session_state['active_button'] = "Générer un rapport"

with col3:
    if st.button("Télécharger la carte", key="telecharger_carte"):
        st.session_state['active_button'] = "Télécharger la carte"
    if st.button("Dessin automatique", key="dessin_auto"):
        st.session_state['active_button'] = "Dessin automatique"

# Création d'un espace réservé pour les paramètres
parameters_placeholder = st.empty()

# Affichage des paramètres en fonction du bouton actif
if st.session_state['active_button']:
    with parameters_placeholder.container():
        display_parameters(st.session_state['active_button'])

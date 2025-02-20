import streamlit as st
from streamlit_folium import st_folium, folium_static
import folium
from folium.plugins import Draw, MeasureControl
from folium import LayerControl
import rasterio
import rasterio.warp
from rasterio.plot import reshape_as_image
from PIL import Image
from rasterio.warp import transform_bounds, calculate_default_transform, reproject
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point, LineString, shape
import json
from io import BytesIO
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import os
import uuid  # Pour générer des identifiants uniques
from rasterio.mask import mask
from shapely.geometry import LineString as ShapelyLineString

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
        # Si le CRS source est déjà le CRS cible, on renvoie directement le fichier d'entrée
        if src.crs.to_string() == target_crs:
            return input_tiff
        # Activation de PROJ_NETWORK pour faciliter la résolution des opérations de transformation
        with rasterio.Env(PROJ_NETWORK='ON'):
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
        unique_id = str(uuid.uuid4())[:8]  # 8 premiers caractères d'un UUID
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
                    resampling=Resampling.nearest,
                )
    return reprojected_tiff

# Fonction pour appliquer un gradient de couleur à un MNT/MNS
def apply_color_gradient(tiff_path, output_path):
    """Apply a color gradient to the DEM TIFF and save it as a PNG."""
    with rasterio.open(tiff_path) as src:
        dem_data = src.read(1)
        cmap = plt.get_cmap("terrain")
        norm = plt.Normalize(vmin=dem_data.min(), vmax=dem_data.max())
        colored_image = cmap(norm(dem_data))
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
    gdf = gpd.GeoDataFrame.from_features(geojson_data)
    return gdf.total_bounds  # [minx, miny, maxx, maxy]

# Fonction pour charger un fichier TIFF
def load_tiff(tiff_path):
    """Charge un fichier TIFF et retourne les données et les bornes."""
    try:
        with rasterio.open(tiff_path) as src:
            data = src.read(1)
            bounds = src.bounds
            transform = src.transform
            if transform.is_identity:
                st.warning("La transformation est invalide. Génération d'une transformation par défaut.")
                transform, width, height = calculate_default_transform(src.crs, src.crs, src.width, src.height, *src.bounds)
        return data, bounds, transform
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier TIFF : {e}")
        return None, None, None

# Fonction de validation de projection et d'emprise
def validate_projection_and_extent(raster_path, polygons_gdf, target_crs):
    """Vérifie la projection et l'emprise des polygones par rapport au raster."""
    with rasterio.open(raster_path) as src:
        if src.crs.to_string() != target_crs:
            raise ValueError(f"Le raster {raster_path} n'est pas dans la projection {target_crs}")
        polygons_gdf = polygons_gdf.to_crs(src.crs)
        raster_bounds = src.bounds
        for idx, row in polygons_gdf.iterrows():
            if not row.geometry.intersects(Polygon.from_bounds(*raster_bounds)):
                st.warning(f"Le polygone {idx} est en dehors de l'emprise du raster")
    return polygons_gdf

# Fonctions pour calculer volumes, surfaces et autres traitements spatiaux …
def calculate_volume_and_area_for_each_polygon(mns_path, mnt_path, polygons_gdf):
    volumes = []
    areas = []
    with rasterio.open(mns_path) as src:
        polygons_gdf = polygons_gdf.to_crs(src.crs)
    for idx, polygon in polygons_gdf.iterrows():
        try:
            with rasterio.open(mns_path) as src:
                mns_clipped, mns_transform = mask(src, [polygon.geometry], crop=True, nodata=np.nan)
                mns_data = mns_clipped[0]
                cell_area = abs(mns_transform.a * mns_transform.e)
            with rasterio.open(mnt_path) as src:
                mnt_clipped, _ = mask(src, [polygon.geometry], crop=True, nodata=np.nan)
                mnt_data = mnt_clipped[0]
            valid_mask = (~np.isnan(mns_data)) & (~np.isnan(mnt_data))
            diff = np.where(valid_mask, mns_data - mnt_data, 0)
            volume = np.sum(diff) * cell_area
            area = np.count_nonzero(valid_mask) * cell_area
            volumes.append(volume)
            areas.append(area)
            polygon_name = polygon.get("properties", {}).get("name", f"Polygone {idx + 1}")
            st.write(f"{polygon_name} - Volume: {volume:.2f} m³, Surface: {area:.2f} m²")
        except Exception as e:
            st.error(f"Erreur sur le polygone {idx + 1}: {str(e)}")
    return volumes, areas

def extract_boundary_points(polygon):
    boundary = polygon.boundary
    if isinstance(boundary, ShapelyLineString):
        return list(boundary.coords)
    else:
        return list(polygon.exterior.coords)

def calculate_average_elevation_on_boundary(mns_path, polygon):
    with rasterio.open(mns_path) as src:
        boundary_points = extract_boundary_points(polygon)
        boundary_coords = [src.index(x, y) for (x, y) in boundary_points]
        elevations = [src.read(1)[int(row), int(col)] for (row, col) in boundary_coords]
        average_elevation = np.mean(elevations)
    return average_elevation

def calculate_volume_and_area_with_mns_only(mns_path, polygons_gdf, use_average_elevation=True, reference_altitude=None):
    volumes = []
    areas = []
    with rasterio.open(mns_path) as src:
        polygons_gdf = polygons_gdf.to_crs(src.crs)
    for idx, polygon in polygons_gdf.iterrows():
        try:
            with rasterio.open(mns_path) as src:
                mns_clipped, mns_transform = mask(src, [polygon.geometry], crop=True, nodata=np.nan)
                mns_data = mns_clipped[0]
                cell_area = abs(mns_transform.a * mns_transform.e)
            if use_average_elevation:
                reference_altitude = calculate_average_elevation_on_boundary(mns_path, polygon.geometry)
            elif reference_altitude is None:
                st.error("Veuillez fournir une altitude de référence.")
                return [], []
            valid_mask = ~np.isnan(mns_data)
            diff = np.where(valid_mask, mns_data - reference_altitude, 0)
            volume = np.sum(diff) * cell_area
            area = np.count_nonzero(valid_mask) * cell_area
            volumes.append(volume)
            areas.append(area)
            polygon_name = polygon.get("properties", {}).get("name", f"Polygone {idx + 1}")
            st.write(f"{polygon_name} - Volume: {volume:.2f} m³, Surface: {area:.2f} m², Cote de référence: {reference_altitude:.2f} m")
        except Exception as e:
            st.error(f"Erreur sur le polygone {idx + 1}: {str(e)}")
    return volumes, areas

def calculate_global_volume(volumes):
    return sum(volumes)

def calculate_global_area(areas):
    return sum(areas)

def find_polygons_in_layers(layers):
    polygons = []
    for layer in layers:
        if layer["type"] == "GeoJSON":
            geojson_data = layer["data"]
            for feature in geojson_data["features"]:
                if feature["geometry"]["type"] == "Polygon":
                    polygons.append(feature)
    return polygons

def find_polygons_in_user_layers(layers):
    polygons = []
    for layer_name, features in layers.items():
        for feature in features:
            if feature["geometry"]["type"] == "Polygon":
                polygons.append(feature)
    return polygons

def convert_polygons_to_gdf(polygons):
    geometries = [shape(polygon["geometry"]) for polygon in polygons]
    properties = [polygon.get("properties", {}) for polygon in polygons]
    gdf = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")
    gdf["properties"] = properties
    return gdf

def convert_drawn_features_to_gdf(features):
    geometries = []
    properties = []
    for feature in features:
        geom = shape(feature["geometry"])
        geometries.append(geom)
        properties.append(feature.get("properties", {}))
    gdf = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")
    gdf["properties"] = properties
    return gdf

# Initialisation des états de session
if "layers" not in st.session_state:
    st.session_state["layers"] = {}  # Couches créées par l'utilisateur
if "uploaded_layers" not in st.session_state:
    st.session_state["uploaded_layers"] = []  # Couches téléversées (TIFF et GeoJSON)
if "new_features" not in st.session_state:
    st.session_state["new_features"] = []  # Entités temporairement dessinées

# Choix du mode d'affichage des TIFF (pour affichage sur la carte folium)
display_mode = st.sidebar.radio(
    "Mode d'affichage des TIFF",
    options=["Reprojection en EPSG:4326", "Affichage en UTM"]
)

st.title("Carte Topographique et Analyse Spatiale")
st.markdown("""
Créez des entités géographiques (points, lignes, polygones) en les dessinant sur la carte et ajoutez-les à des couches spécifiques. 
Vous pouvez également téléverser des fichiers TIFF ou GeoJSON pour les superposer à la carte.
""")

# Sidebar pour la gestion des couches
with st.sidebar:
    st.header("Gestion des Couches")
    st.markdown("### 1- Ajouter une nouvelle couche")
    new_layer_name = st.text_input("Nom de la nouvelle couche à ajouter", "")
    if st.button("Ajouter la couche", key="add_layer_button", help="Ajouter une nouvelle couche", type="primary") and new_layer_name:
        if new_layer_name not in st.session_state["layers"]:
            st.session_state["layers"][new_layer_name] = []
            st.success(f"La couche '{new_layer_name}' a été ajoutée.")
        else:
            st.warning(f"La couche '{new_layer_name}' existe déjà.")

    st.markdown("#### Sélectionner une couche active")
    if st.session_state["layers"]:
        layer_name = st.selectbox("Choisissez la couche à laquelle ajouter les entités", list(st.session_state["layers"].keys()))
    else:
        st.write("Aucune couche disponible. Ajoutez une couche pour commencer.")

    if st.session_state["new_features"]:
        st.write(f"**Entités dessinées temporairement ({len(st.session_state['new_features'])}) :**")
        for idx, feature in enumerate(st.session_state["new_features"]):
            st.write(f"- Entité {idx + 1}: {feature['geometry']['type']}")
    if st.button("Enregistrer les entités", key="save_features_button", type="primary") and st.session_state["layers"]:
        current_layer = st.session_state["layers"][layer_name]
        for feature in st.session_state["new_features"]:
            if feature not in current_layer:
                current_layer.append(feature)
        st.session_state["new_features"] = []
        st.success(f"Toutes les nouvelles entités ont été enregistrées dans la couche '{layer_name}'.")
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
    st.markdown("---")

    # Section 2 : Téléversement de fichiers
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
            unique_id = str(uuid.uuid4())[:8]
            tiff_path = f"uploaded_{unique_id}.tiff"
            with open(tiff_path, "wb") as f:
                f.write(uploaded_tiff.read())
            st.write(f"Traitement du fichier TIFF ({tiff_type})...")
            try:
                if display_mode == "Reprojection en EPSG:4326":
                    processed_tiff = reproject_tiff(tiff_path, "EPSG:4326")
                else:
                    # Pour l'affichage en UTM, on conserve le fichier original,
                    # mais on convertit les bornes en EPSG:4326 pour l'affichage sur la carte
                    processed_tiff = tiff_path
                with rasterio.open(processed_tiff) as src:
                    if display_mode == "Affichage en UTM":
                        bounds = rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
                    else:
                        bounds = src.bounds
                if not any(layer["name"] == tiff_type and layer["type"] == "TIFF" for layer in st.session_state["uploaded_layers"]):
                    st.session_state["uploaded_layers"].append({"type": "TIFF", "name": tiff_type, "path": processed_tiff, "bounds": bounds})
                    st.success(f"Couche {tiff_type} ajoutée à la liste des couches.")
                else:
                    st.warning(f"La couche {tiff_type} existe déjà.")
            except Exception as e:
                st.error(f"Erreur lors du traitement du fichier TIFF : {e}")
            finally:
                # Pour le mode reprojection, supprimer le fichier temporaire original
                if display_mode == "Reprojection en EPSG:4326":
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
                if not any(layer["name"] == geojson_type and layer["type"] == "GeoJSON" for layer in st.session_state["uploaded_layers"]):
                    st.session_state["uploaded_layers"].append({"type": "GeoJSON", "name": geojson_type, "data": geojson_data})
                    st.success(f"Couche {geojson_type} ajoutée à la liste des couches.")
                else:
                    st.warning(f"La couche {geojson_type} existe déjà.")
            except Exception as e:
                st.error(f"Erreur lors du chargement du GeoJSON : {e}")
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
m = folium.Map(location=[7.5399, -5.5471], zoom_start=6)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="Satellite",
).add_to(m)
folium.TileLayer(
    tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
    attr="OpenTopoMap",
    name="Topographique",
).add_to(m)
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
            unique_id = str(uuid.uuid4())[:8]
            temp_png_path = f"{layer['name'].lower()}_colored_{unique_id}.png"
            apply_color_gradient(layer["path"], temp_png_path)
            add_image_overlay(m, temp_png_path, layer["bounds"], layer["name"])
            os.remove(temp_png_path)
        else:
            add_image_overlay(m, layer["path"], layer["bounds"], layer["name"])
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
LayerControl(position="topleft", collapsed=True).add_to(m)
output = st_folium(m, width=800, height=600, returned_objects=["last_active_drawing", "all_drawings"])
if output and "last_active_drawing" in output and output["last_active_drawing"]:
    new_feature = output["last_active_drawing"]
    if new_feature not in st.session_state["new_features"]:
        st.session_state["new_features"].append(new_feature)
        st.info("Nouvelle entité ajoutée temporairement. Cliquez sur 'Enregistrer les entités' pour les ajouter à la couche.")
if 'active_button' not in st.session_state:
    st.session_state['active_button'] = None

def display_parameters(button_name):
    if button_name == "Surfaces et volumes":
        st.markdown("### Calcul des volumes et des surfaces")
        method = st.radio(
            "Choisissez la méthode de calcul :",
            ("Méthode 1 : MNS - MNT", "Méthode 2 : MNS seul"),
            key="volume_method"
        )
        mns_layer = next((layer for layer in st.session_state["uploaded_layers"] if layer["name"] == "MNS"), None)
        mnt_layer = next((layer for layer in st.session_state["uploaded_layers"] if layer["name"] == "MNT"), None)
        if not mns_layer:
            st.error("La couche MNS est manquante. Veuillez téléverser un fichier MNS.")
            return
        if method == "Méthode 1 : MNS - MNT" and not mnt_layer:
            st.error("La couche MNT est manquante. Veuillez téléverser un fichier MNT.")
            return
        try:
            mns_utm_path = reproject_tiff(mns_layer["path"], "EPSG:32630")
            if method == "Méthode 1 : MNS - MNT":
                mnt_utm_path = reproject_tiff(mnt_layer["path"], "EPSG:32630")
        except Exception as e:
            st.error(f"Échec de la reprojection : {e}")
            return
        polygons_uploaded = find_polygons_in_layers(st.session_state["uploaded_layers"])
        polygons_user_layers = find_polygons_in_user_layers(st.session_state["layers"])
        polygons_drawn = st.session_state["new_features"]
        all_polygons = polygons_uploaded + polygons_user_layers + polygons_drawn
        if not all_polygons:
            st.error("Aucune polygonale disponible.")
            return
        polygons_gdf = convert_polygons_to_gdf(all_polygons)
        try:
            polygons_gdf_utm = validate_projection_and_extent(mns_utm_path, polygons_gdf, "EPSG:32630")
            if method == "Méthode 1 : MNS - MNT":
                volumes, areas = calculate_volume_and_area_for_each_polygon(
                    mns_utm_path, 
                    mnt_utm_path,
                    polygons_gdf_utm
                )
            else:
                use_average_elevation = st.checkbox(
                    "Utiliser la cote moyenne des élévations sur les bords de la polygonale comme référence",
                    value=True,
                    key="use_average_elevation"
                )
                reference_altitude = None
                if not use_average_elevation:
                    reference_altitude = st.number_input(
                        "Entrez l'altitude de référence (en mètres) :",
                        value=0.0,
                        step=0.1,
                        key="reference_altitude"
                    )
                volumes, areas = calculate_volume_and_area_with_mns_only(
                    mns_utm_path,
                    polygons_gdf_utm,
                    use_average_elevation=use_average_elevation,
                    reference_altitude=reference_altitude
                )
            global_volume = calculate_global_volume(volumes)
            global_area = calculate_global_area(areas)
            st.write(f"Volume global : {global_volume:.2f} m³")
            st.write(f"Surface globale : {global_area:.2f} m²")
            os.remove(mns_utm_path)
            if method == "Méthode 1 : MNS - MNT":
                os.remove(mnt_utm_path)
        except Exception as e:
            st.error(f"Erreur lors du calcul : {e}")
            if os.path.exists(mns_utm_path):
                os.remove(mns_utm_path)
            if method == "Méthode 1 : MNS - MNT" and os.path.exists(mnt_utm_path):
                os.remove(mnt_utm_path)

st.markdown("### Analyse Spatiale")
col1, col2, col3 = st.columns(3)
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
parameters_placeholder = st.empty()
if st.session_state['active_button']:
    with parameters_placeholder.container():
        display_parameters(st.session_state['active_button'])

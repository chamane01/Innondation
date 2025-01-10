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

    # Initialize session state for layers
    if "layers" not in st.session_state:
        st.session_state["layers"] = []

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
    uploaded_tiff = st.file_uploader("Téléverser un fichier TIFF", type=["tif", "tiff"], accept_multiple_files=True)
    if uploaded_tiff:
        for file in uploaded_tiff:
            tiff_path = file.name
            with open(tiff_path, "wb") as f:
                f.write(file.read())

            st.write(f"Reprojection du fichier TIFF {tiff_path}...")
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
                    st.session_state["layers"].append(("TIFF", reprojected_tiff, bounds))
            except Exception as e:
                st.error(f"Erreur lors de la reprojection : {e}")

    # Téléversement des fichiers GeoJSON
    uploaded_geojson = st.file_uploader("Téléverser un fichier GeoJSON", type=["geojson"], accept_multiple_files=True)
    if uploaded_geojson:
        for file in uploaded_geojson:
            try:
                geojson_data = json.load(file)
                st.session_state["layers"].append(("GeoJSON", geojson_data))
            except Exception as e:
                st.error(f"Erreur lors du chargement du GeoJSON : {e}")

    # Bouton pour ajouter les couches à la carte
    if st.button("Ajouter couches"):
        for layer in st.session_state["layers"]:
            if layer[0] == "TIFF":
                add_image_overlay(fmap, layer[1], layer[2], "Couche TIFF")
            elif layer[0] == "GeoJSON":
                folium.GeoJson(
                    layer[1],
                    name="Couche GeoJSON",
                    style_function=lambda x: {
                        "color": "orange",
                        "weight": 4,
                        "opacity": 0.7
                    }
                ).add_to(fmap)

    # Ajout des contrôles de calques
    folium.LayerControl().add_to(fmap)

    # Affichage de la carte
    folium_static(fmap, width=700, height=500)

if __name__ == "__main__":
    main()



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

# Détection des arbres avec DBSCAN
def detect_trees(heights, threshold, eps, min_samples):
    tree_mask = heights > threshold
    coords = np.column_stack(np.where(tree_mask))

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    tree_clusters = clustering.labels_

    return coords, tree_clusters

# Calcul des centroïdes des arbres
def calculate_cluster_centroids(coords, clusters):
    unique_clusters = set(clusters) - {-1}
    centroids = []

    for cluster_id in unique_clusters:
        cluster_coords = coords[clusters == cluster_id]
        centroid = cluster_coords.mean(axis=0)
        centroids.append((cluster_id, centroid))

    return centroids

# Ajout des centroïdes des arbres sur la carte
def add_tree_centroids_layer(map_object, centroids, bounds, image_shape, layer_name):
    height = bounds[3] - bounds[1]
    width = bounds[2] - bounds[0]
    img_height, img_width = image_shape[:2]

    feature_group = folium.FeatureGroup(name=layer_name)
    for _, centroid in centroids:
        lat = bounds[3] - height * (centroid[0] / img_height)
        lon = bounds[0] + width * (centroid[1] / img_width)

        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color="green",
            fill=True,
            fill_color="green",
            fill_opacity=0.8,
        ).add_to(feature_group)

    feature_group.add_to(map_object)

# Fonction pour compter les arbres à l'intérieur de la polygonale
def count_trees_in_polygon(centroids, bounds, image_shape, polygon_gdf):
    height = bounds[3] - bounds[1]
    width = bounds[2] - bounds[0]
    img_height, img_width = image_shape[:2]

    tree_count = 0
    for _, centroid in centroids:
        lat = bounds[3] - height * (centroid[0] / img_height)
        lon = bounds[0] + width * (centroid[1] / img_width)
        point = Point(lon, lat)
        if polygon_gdf.contains(point).any():
            tree_count += 1

    return tree_count

# Streamlit app
st.title("Outils et d'analyses")

# Boutons sous la carte
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Faire une carte"):
        st.info("Fonctionnalité 'Faire une carte' en cours de développement.")
with col2:
    if st.button("Calculer des volumes"):
        st.session_state.show_volume_sidebar = True
        st.session_state.show_tree_sidebar = False  # Désactiver l'autre sidebar
with col3:
    if st.button("Détecter les arbres"):
        st.session_state.show_tree_sidebar = True
        st.session_state.show_volume_sidebar = False  # Désactiver l'autre sidebar

# Affichage des paramètres pour le calcul des volumes
if st.session_state.get("show_volume_sidebar", False):
    st.sidebar.title("Paramètres de calcul des volumes")

    # Choix de la méthode
    method = st.sidebar.radio(
        "Choisissez la méthode de calcul :",
        ("Méthode 1 : MNS - MNT", "Méthode 2 : MNS seul"),
        key="volume_method"
    )

    # Téléversement des fichiers
    mns_file = st.sidebar.file_uploader("Téléchargez le fichier MNS (TIFF)", type=["tif", "tiff"], key="mns_volume")
    polygon_file = st.sidebar.file_uploader("Téléchargez un fichier de polygone (obligatoire)", type=["geojson", "shp"], key="polygon_volume")

    if method == "Méthode 1 : MNS - MNT":
        mnt_file = st.sidebar.file_uploader("Téléchargez le fichier MNT (TIFF)", type=["tif", "tiff"], key="mnt_volume")
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
                        step=0.1,
                        key="reference_altitude"
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

# Affichage des paramètres pour la détection des arbres
if st.session_state.get("show_tree_sidebar", False):
    st.sidebar.title("Paramètres de détection des arbres")

    # Téléversement des fichiers
    mnt_file = st.sidebar.file_uploader("Téléchargez le fichier MNT (TIFF)", type=["tif", "tiff"], key="mnt_tree")
    mns_file = st.sidebar.file_uploader("Téléchargez le fichier MNS (TIFF)", type=["tif", "tiff"], key="mns_tree")
    polygon_file = st.sidebar.file_uploader("Téléchargez un fichier de polygone (optionnel)", type=["geojson", "shp"], key="polygon_tree")

    if mnt_file and mns_file:
        mnt, mnt_bounds = load_tiff(mnt_file)
        mns, mns_bounds = load_tiff(mns_file)

        if mnt is None or mns is None:
            st.sidebar.error("Erreur lors du chargement des fichiers.")
        elif mnt_bounds != mns_bounds:
            st.sidebar.error("Les fichiers doivent avoir les mêmes bornes géographiques.")
        else:
            heights = calculate_heights(mns, mnt)

            # Paramètres de détection
            height_threshold = st.sidebar.slider("Seuil de hauteur", 0.1, 20.0, 2.0, 0.1, key="height_threshold")
            eps = st.sidebar.slider("Rayon de voisinage", 0.1, 10.0, 2.0, 0.1, key="eps")
            min_samples = st.sidebar.slider("Min. points pour un cluster", 1, 10, 5, 1, key="min_samples")

            # Détection et visualisation
            if st.sidebar.button("Lancer la détection"):
                coords, tree_clusters = detect_trees(heights, height_threshold, eps, min_samples)
                num_trees = len(set(tree_clusters)) - (1 if -1 in tree_clusters else 0)
                st.sidebar.write(f"Nombre d'arbres détectés : {num_trees}")

                centroids = calculate_cluster_centroids(coords, tree_clusters)

                # Mise à jour de la carte
                center_lat = (mnt_bounds[1] + mnt_bounds[3]) / 2
                center_lon = (mnt_bounds[0] + mnt_bounds[2]) / 2
                fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12)

                folium.raster_layers.ImageOverlay(
                    image=mnt,
                    bounds=[[mnt_bounds[1], mnt_bounds[0]], [mnt_bounds[3], mnt_bounds[2]]],
                    opacity=0.5,
                    name="MNT"
                ).add_to(fmap)

                add_tree_centroids_layer(fmap, centroids, mnt_bounds, mnt.shape, "Arbres")

                # Ajout des polygones (optionnel)
                if polygon_file:
                    polygons_gdf = load_and_reproject_shapefile(polygon_file)
                    folium.GeoJson(
                        polygons_gdf,
                        name="Polygones",
                        style_function=lambda x: {'fillOpacity': 0, 'color': 'red', 'weight': 2}
                    ).add_to(fmap)

                    # Compter les arbres à l'intérieur de la polygonale
                    tree_count_in_polygon = count_trees_in_polygon(centroids, mnt_bounds, mnt.shape, polygons_gdf)
                    st.sidebar.write(f"Nombre d'arbres dans la polygonale : {tree_count_in_polygon}")

                fmap.add_child(MeasureControl(position='topleft'))
                fmap.add_child(Draw(position='topleft', export=True))
                fmap.add_child(folium.LayerControl(position='topright'))

                # Afficher la carte
                folium_static(fmap, width=700, height=500)

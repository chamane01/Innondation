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
            "polyline": {"shapeOptions": {"color": "orange", "weight": 4, "opacity": 0.7}},
            "polygon": {"shapeOptions": {"color": "green", "weight": 4, "opacity": 0.7}},
            "rectangle": {"shapeOptions": {"color": "red", "weight": 4, "opacity": 0.7}},
            "circle": {"shapeOptions": {"color": "purple", "weight": 4, "opacity": 0.7}},
        },
        edit_options={"edit": True},
    )
    fmap.add_child(draw)

    # Téléversement des fichiers
    uploaded_files = st.file_uploader(
        "Téléverser des fichiers (TIFF ou GeoJSON)", type=["tif", "tiff", "geojson"], accept_multiple_files=True
    )
    
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith((".tif", ".tiff")):
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.read())

            st.write(f"Reprojection du fichier {uploaded_file.name}...")
            try:
                reprojected_file = reproject_tiff(uploaded_file.name, "EPSG:4326")
                bounds = None
                with rasterio.open(reprojected_file) as src:
                    bounds = src.bounds

                # Appliquer un gradient si nécessaire
                temp_png_path = f"{uploaded_file.name}_colored.png"
                apply_color_gradient(reprojected_file, temp_png_path)

                # Ajouter en tant que couche
                if bounds:
                    add_image_overlay(fmap, temp_png_path, bounds, uploaded_file.name)

                # Supprimer le fichier temporaire
                os.remove(temp_png_path)
            except Exception as e:
                st.error(f"Erreur lors de la reprojection : {e}")

        elif uploaded_file.name.endswith(".geojson"):
            try:
                geojson_data = json.load(uploaded_file)
                folium.GeoJson(
                    geojson_data,
                    name=uploaded_file.name,
                    style_function=lambda x: {
                        "color": "blue",
                        "weight": 2,
                        "opacity": 0.7
                    },
                ).add_to(fmap)
            except Exception as e:
                st.error(f"Erreur lors du chargement du GeoJSON : {e}")

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

# Calcul de hauteur relative
def calculate_heights(mns, mnt):
    return np.maximum(0, mns - mnt)

# Détection des arbres avec DBSCAN
def detect_trees(heights, threshold, eps, min_samples):
    tree_mask = heights > threshold
    coords = np.column_stack(np.where(tree_mask))

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    tree_clusters = clustering.labels_

    return coords, tree_clusters

# Calcul des centroïdes
def calculate_cluster_centroids(coords, clusters):
    unique_clusters = set(clusters) - {-1}
    centroids = []

    for cluster_id in unique_clusters:
        cluster_coords = coords[clusters == cluster_id]
        centroid = cluster_coords.mean(axis=0)
        centroids.append((cluster_id, centroid))

    return centroids

def reproject_tiff(input_tiff, target_crs):
    """Reproject a TIFF file to a target CRS."""
    with rasterio.open(input_tiff) as src:
        transform, width, height = rasterio.warp.calculate_default_transform(
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
                rasterio.warp.reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=rasterio.warp.Resampling.nearest
                )

    return reprojected_tiff

def add_image_overlay(map_object, tiff_path, bounds, name):
    """Add a TIFF image overlay to a Folium map."""
    with rasterio.open(tiff_path) as src:
        image = reshape_as_image(src.read())
        folium.raster_layers.ImageOverlay(
            image=image,
            bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
            name=name
        ).add_to(map_object)





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







# Boutons sous la carte
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Faire une carte"):
        st.info("Fonctionnalité 'Faire une carte' en cours de développement.")
with col2:
    if st.button("Calculer des volumes"):
        st.info("Fonctionnalité 'Calculer des volumes' en cours de développement.")
with col3:
    if st.button("Détecter les arbres"):
        st.session_state.show_sidebar = True

# Affichage des paramètres uniquement si le bouton est cliqué
if st.session_state.get("show_sidebar", False):
    st.sidebar.title("Paramètres de détection")

    # Téléversement des fichiers
    mnt_file = st.sidebar.file_uploader("Téléchargez le fichier MNT (TIFF)", type=["tif", "tiff"])
    mns_file = st.sidebar.file_uploader("Téléchargez le fichier MNS (TIFF)", type=["tif", "tiff"])
    road_file = st.sidebar.file_uploader("Téléchargez un fichier de route (optionnel)", type=["geojson", "shp"])
    polygon_file = st.sidebar.file_uploader("Téléchargez un fichier de polygone (optionnel)", type=["geojson", "shp"])

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
            height_threshold = st.sidebar.slider("Seuil de hauteur", 0.1, 20.0, 2.0, 0.1)
            eps = st.sidebar.slider("Rayon de voisinage", 0.1, 10.0, 2.0, 0.1)
            min_samples = st.sidebar.slider("Min. points pour un cluster", 1, 10, 5, 1)

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

                # Ajout des routes et polygones
                if road_file:
                    roads_gdf = load_and_reproject_shapefile(road_file)
                    folium.GeoJson(roads_gdf, name="Routes", style_function=lambda x: {'color': 'orange', 'weight': 2}).add_to(fmap)

                if polygon_file:
                    polygons_gdf = load_and_reproject_shapefile(polygon_file)
                    folium.GeoJson(polygons_gdf, name="Polygones", style_function=lambda x: {'fillOpacity': 0, 'color': 'red', 'weight': 2}).add_to(fmap)

                fmap.add_child(MeasureControl(position='topleft'))
                fmap.add_child(Draw(position='topleft', export=True))
                fmap.add_child(folium.LayerControl(position='topright'))

                folium_static(fmap)

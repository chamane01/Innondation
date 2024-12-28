import streamlit as st
import rasterio
import rasterio.warp
import folium
from folium import plugins
from folium.plugins import MeasureControl, Draw
from rasterio.plot import reshape_as_image
from PIL import Image
from streamlit_folium import folium_static
import geopandas as gpd

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
        # Read the image and reshape it into a format compatible with Folium
        image = reshape_as_image(src.read())
        folium.raster_layers.ImageOverlay(
            image=image,
            bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
            name=name,
            opacity=0.6
        ).add_to(map_object)

def add_geojson_layer(map_object, geojson_file, name):
    """Add a GeoJSON file as a layer to the map."""
    if geojson_file is not None:
        gdf = gpd.read_file(geojson_file)
        folium.GeoJson(gdf).add_to(map_object)
        st.write(f"Layer '{name}' added successfully.")

# Streamlit app
def main():
    st.title("TIFF Viewer and Interactive Map")

    # Button to toggle sidebar visibility
    if st.button("Dessiner"):
        with st.sidebar:
            uploaded_file = st.file_uploader("Upload a TIFF file", type=["tif", "tiff"])

            # Optional GeoJSON uploads
            st.subheader("Ajouter des couches (facultatif)")
            uploaded_routes = st.file_uploader("Télécharger les routes (GeoJSON)", type="geojson")
            uploaded_buildings = st.file_uploader("Télécharger les bâtiments (GeoJSON)", type="geojson")
            uploaded_waterways = st.file_uploader("Télécharger les cours d'eau (GeoJSON)", type="geojson")

        if uploaded_file is not None:
            tiff_path = uploaded_file.name
            with open(tiff_path, "wb") as f:
                f.write(uploaded_file.read())

            st.write("Reprojecting TIFF file...")

            # Reproject TIFF to target CRS (e.g., EPSG:4326)
            reprojected_tiff = reproject_tiff(tiff_path, "EPSG:4326")

            # Read bounds from reprojected TIFF file
            with rasterio.open(reprojected_tiff) as src:
                bounds = src.bounds

            # Create Folium map centered on the bounds
            center_lat = (bounds.top + bounds.bottom) / 2
            center_lon = (bounds.left + bounds.right) / 2
            fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12)

            # Add reprojected TIFF as overlay
            add_image_overlay(fmap, reprojected_tiff, bounds, "TIFF Layer")

            # Add GeoJSON layers if uploaded
            if uploaded_routes is not None:
                add_geojson_layer(fmap, uploaded_routes, "Routes")
            if uploaded_buildings is not None:
                add_geojson_layer(fmap, uploaded_buildings, "Bâtiments")
            if uploaded_waterways is not None:
                add_geojson_layer(fmap, uploaded_waterways, "Cours d'eau")

            # Add measure control
            fmap.add_child(MeasureControl(position='topleft'))

            # Add draw control
            draw = Draw(position='topleft', export=True,
                        draw_options={'polyline': {'shapeOptions': {'color': 'blue', 'weight': 4, 'opacity': 0.7}},
                                      'polygon': {'shapeOptions': {'color': 'green', 'weight': 4, 'opacity': 0.7}},
                                      'rectangle': {'shapeOptions': {'color': 'red', 'weight': 4, 'opacity': 0.7}},
                                      'circle': {'shapeOptions': {'color': 'purple', 'weight': 4, 'opacity': 0.7}}},
                        edit_options={'edit': True,}
            )
            fmap.add_child(draw)

            # Layer control
            folium.LayerControl().add_to(fmap)

            # Display map
            folium_static(fmap)

if __name__ == "__main__":
    main()











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

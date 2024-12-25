


# Importer les bibliothèques nécessaires
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import Polygon, box
from shapely.geometry import MultiPolygon
import contextily as ctx
import ezdxf  # Bibliothèque pour créer des fichiers DXF
from datetime import datetime
import rasterio



import streamlit as st
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from scipy.ndimage import binary_erosion
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# Fonction pour charger un fichier TIFF
def load_tiff(file_path):
    with rasterio.open(file_path) as src:
        data = src.read(1)  # On lit uniquement la première bande
        profile = src.profile
    return data, profile

# Fonction pour générer un MNT
def generate_mnt(elevation_data, structure_size=3):
    """
    Génère un MNT en filtrant les objets élevés et en interpolant les données manquantes.
    """
    # Étape 1 : Filtrage morphologique pour détecter le sol
    ground_mask = binary_erosion(elevation_data > 0, structure=np.ones((structure_size, structure_size)))

    # Étape 2 : Interpolation pour combler les trous
    x, y = np.meshgrid(np.arange(elevation_data.shape[1]), np.arange(elevation_data.shape[0]))
    points = np.column_stack((x[ground_mask], y[ground_mask]))
    values = elevation_data[ground_mask]
    grid_z = griddata(points, values, (x, y), method='linear', fill_value=np.nan)

    # Étape 3 : Remplir les valeurs NaN par une moyenne locale
    grid_z_filled = np.nan_to_num(grid_z, nan=np.nanmean(grid_z))
    return grid_z_filled

# Fonction pour exporter le MNT en TIFF dans un fichier mémoire
def export_mnt_to_tiff(mnt, profile):
    """
    Exporte le MNT dans un fichier TIFF et retourne un objet binaire pour le téléchargement.
    """
    profile.update({
        "dtype": "float32",
        "count": 1,
        "compress": "lzw",
    })

    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(mnt, 1)
        return memfile.read()

# Interface Streamlit
st.title("Génération d'un MNT à partir d'un fichier TIFF")

# Chargement du fichier TIFF
tiff_file = st.file_uploader("Téléchargez un fichier TIFF contenant des données d'altitude", type=["tif", "tiff"])

if tiff_file:
    # Charger les données
    elevation_data, profile = load_tiff(tiff_file)
    st.write("Données d'altitude chargées :")
    st.image(elevation_data, caption="Image d'altitude", use_column_width=True, clamp=True)

    # Générer le MNT
    structure_size = st.sidebar.slider("Taille du filtre morphologique (px)", 1, 10, 3)
    mnt = generate_mnt(elevation_data, structure_size=structure_size)

    # Afficher les résultats
    st.write("Modèle Numérique de Terrain (MNT) généré :")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(elevation_data, cmap="terrain", interpolation="none")
    ax[0].set_title("Données originales")
    ax[1].imshow(mnt, cmap="terrain", interpolation="none")
    ax[1].set_title("MNT généré")
    st.pyplot(fig)

    # Exportation et téléchargement
    tiff_data = export_mnt_to_tiff(mnt, profile)
    st.download_button(
        label="Télécharger le MNT en TIFF",
        data=tiff_data,
        file_name="mnt_generated.tif",
        mime="image/tiff"
    )








import streamlit as st
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from sklearn.cluster import DBSCAN
import folium
from folium.plugins import MeasureControl, Draw
from streamlit_folium import folium_static
import json
import geopandas as gpd
from shapely.geometry import Polygon

# Fonction pour charger un fichier TIFF et reprojeter les bornes
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

# Fonction pour charger un fichier GeoJSON et le projeter
def load_geojson(file_path, target_crs="EPSG:4326"):
    try:
        gdf = gpd.read_file(file_path)
        gdf = gdf.to_crs(target_crs)  # Reprojection vers le CRS cible
        return gdf
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier GeoJSON : {e}")
        return None

# Fonction pour calculer la hauteur relative (MNS - MNT)
def calculate_heights(mns, mnt):
    return np.maximum(0, mns - mnt)  # Évite les valeurs négatives

# Fonction pour détecter les arbres avec DBSCAN
def detect_trees(heights, threshold, eps, min_samples):
    tree_mask = heights > threshold
    coords = np.column_stack(np.where(tree_mask))

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    tree_clusters = clustering.labels_

    return coords, tree_clusters

# Fonction pour calculer les centroïdes des clusters
def calculate_cluster_centroids(coords, clusters):
    unique_clusters = set(clusters) - {-1}  # Ignorer le bruit (-1)
    centroids = []

    for cluster_id in unique_clusters:
        cluster_coords = coords[clusters == cluster_id]
        centroid = cluster_coords.mean(axis=0)
        centroids.append((cluster_id, centroid))

    return centroids

# Fonction pour ajouter les centroïdes des arbres sous forme de cercles
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
            radius=3,  # Rayon du cercle en pixels
            color="green",
            fill=True,
            fill_color="green",
            fill_opacity=0.8,
        ).add_to(feature_group)

    feature_group.add_to(map_object)

# Fonction pour exporter une couche en GeoJSON
def export_layer(data, bounds, layer_name):
    """Créer un GeoJSON pour une couche donnée."""
    features = []
    if layer_name == "Arbres":
        for centroid in centroids:
            _, (row, col) = centroid
            lat1 = bounds[3] - (bounds[3] - bounds[1]) * (row / mnt.shape[0])
            lon1 = bounds[0] + (bounds[2] - bounds[0]) * (col / mnt.shape[1])
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon1, lat1]
                },
                "properties": {"type": "Arbre"}
            })
    else:
        # Ajouter les bornes et valeurs en tant que polygones
        for i, row in enumerate(data):
            for j, value in enumerate(row):
                lat1 = bounds[3] - (bounds[3] - bounds[1]) * (i / data.shape[0])
                lat2 = bounds[3] - (bounds[3] - bounds[1]) * ((i + 1) / data.shape[0])
                lon1 = bounds[0] + (bounds[2] - bounds[0]) * (j / data.shape[1])
                lon2 = bounds[0] + (bounds[2] - bounds[0]) * ((j + 1) / data.shape[1])
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [lon1, lat1], [lon2, lat1], [lon2, lat2], [lon1, lat2], [lon1, lat1]
                        ]]
                    },
                    "properties": {"value": value}
                })

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    return json.dumps(geojson)

# Interface Streamlit
st.title("Détection d'arbres automatique ")

mnt_file = st.file_uploader("Téléchargez le fichier MNT (TIFF)", type=["tif", "tiff"])
mns_file = st.file_uploader("Téléchargez le fichier MNS (TIFF)", type=["tif", "tiff"])
geojson_file = st.file_uploader("Téléchargez la polygonale (GeoJSON)", type=["geojson"])
route_file = st.file_uploader("Téléchargez le fichier de route (GeoJSON)", type=["geojson"])  # Nouveau

if mnt_file and mns_file:
    mnt, mnt_bounds = load_tiff(mnt_file)
    mns, mns_bounds = load_tiff(mns_file)

    if mnt is None or mns is None:
        st.error("Erreur lors du chargement des fichiers.")
    elif mnt_bounds != mns_bounds:
        st.error("Les fichiers doivent avoir les mêmes bornes géographiques.")
    else:
        heights = calculate_heights(mns, mnt)
        st.write("Hauteurs calculées (MNS - MNT)")

        st.sidebar.title("Paramètres de détection")
        height_threshold = st.sidebar.slider("Seuil de hauteur", 0.1, 20.0, 2.0, 0.1)
        eps = st.sidebar.slider("Rayon de voisinage", 0.1, 10.0, 2.0, 0.1)
        min_samples = st.sidebar.slider("Min. points pour un cluster", 1, 10, 5, 1)

        coords, tree_clusters = detect_trees(heights, height_threshold, eps, min_samples)
        num_trees = len(set(tree_clusters)) - (1 if -1 in tree_clusters else 0)
        st.write(f"Nombre d'arbres détectés : {num_trees}")

        # Calcul des centroïdes
        centroids = calculate_cluster_centroids(coords, tree_clusters)

        # Ajouter un bouton pour afficher la carte
        if st.button("Afficher la carte"):
            # Création de la carte
            center_lat = (mnt_bounds[1] + mnt_bounds[3]) / 2
            center_lon = (mnt_bounds[0] + mnt_bounds[2]) / 2
            fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12)

            # Ajout du fichier TIFF MNT à la carte
            folium.raster_layers.ImageOverlay(
                image=mnt,
                bounds=[[mnt_bounds[1], mnt_bounds[0]], [mnt_bounds[3], mnt_bounds[2]]],
                opacity=0.5,
                name="MNT"
            ).add_to(fmap)

            # Ajout du fichier TIFF MNS à la carte
            folium.raster_layers.ImageOverlay(
                image=mns,
                bounds=[[mns_bounds[1], mns_bounds[0]], [mns_bounds[3], mns_bounds[2]]],
                opacity=0.5,
                name="MNS"
            ).add_to(fmap)

            # Ajouter la couche des arbres à la carte
            add_tree_centroids_layer(fmap, centroids, mnt_bounds, mnt.shape, "Arbres")

            # Si un fichier GeoJSON est téléchargé, l'ajouter à la carte
            if geojson_file:
                geojson_data = load_geojson(geojson_file)
                if geojson_data is not None:
                    folium.GeoJson(
                        geojson_data,
                        style_function=lambda x: {
                            'fillColor': 'transparent',
                            'color': 'white',
                            'weight': 2
                        }
                    ).add_to(fmap)

            if geojson_file:
                polygon_data = load_geojson(geojson_file)
                if polygon_data is not None:
                    polygon = Polygon(polygon_data.geometry[0].coordinates[0])  # Assumer qu'il n'y a qu'un seul polygone
                    # Compter les arbres dans la polygonale
                    trees_in_polygon = 0
                    for _, centroid in centroids:
                        point = Point(centroid[1], centroid[0])  # (latitude, longitude)
                        if point.within(polygon):
                            trees_in_polygon += 1
                    st.write(f"Nombre d'arbres à l'intérieur de la polygonale : {trees_in_polygon}")

        
                
            
            
        
                    
        
        
        
        
        
    
    

            # Si un fichier de route est téléchargé, l'ajouter à la carte
            if route_file:
                route_data = load_geojson(route_file)
                if route_data is not None:
                    folium.GeoJson(
                        route_data,
                        style_function=lambda x: {
                            'fillColor': 'transparent',
                            'color': 'blue',
                            'weight': 3
                        },
                        name="Route"
                    ).add_to(fmap)

            fmap.add_child(MeasureControl(position='topleft'))
            fmap.add_child(Draw(position='topleft', export=True))

            # Ajouter le contrôle des couches à la carte (en haut à droite)
            fmap.add_child(folium.LayerControl(position='topright'))

            # Afficher la carte
            folium_static(fmap)

        # Ajouter un bouton pour exporter toutes les couches en GeoJSON
        if st.button("Exporter les couches en GeoJSON"):
            # Export du MNT
            mnt_geojson = export_layer(mnt, mnt_bounds, "MNT")
            st.download_button("Télécharger MNT", data=mnt_geojson, file_name="mnt.geojson", mime="application/json")

            # Export du MNS
            mns_geojson = export_layer(mns, mns_bounds, "MNS")
            st.download_button("Télécharger MNS", data=mns_geojson, file_name="mns.geojson", mime="application/json")

            # Export des arbres
            tree_geojson = export_layer(centroids, mnt_bounds, "Arbres")
            st.download_button("Télécharger Arbres", data=tree_geojson, file_name="arbres.geojson", mime="application/json")
















import streamlit as st
import rasterio
import rasterio.warp
import folium
from folium import plugins
from folium.plugins import MeasureControl, Draw
from rasterio.plot import reshape_as_image
from PIL import Image
from streamlit_folium import folium_static

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

# Streamlit app
def main():
    st.title("TIFF Viewer and Interactive Map")

    # Upload TIFF file
    uploaded_file = st.file_uploader("Upload a TIFF file", type=["tif", "tiff"])

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

        # Create Folium map
        center_lat = (bounds.top + bounds.bottom) / 2
        center_lon = (bounds.left + bounds.right) / 2
        fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12)

        # Add reprojected TIFF as overlay
        add_image_overlay(fmap, reprojected_tiff, bounds, "TIFF Layer")

        # Add measure control
        fmap.add_child(MeasureControl())

        # Add draw control
        draw = Draw(export=True)
        fmap.add_child(draw)

        # Layer control
        folium.LayerControl().add_to(fmap)

        # Display map
        folium_static(fmap)

if __name__ == "__main__":
    main()







import streamlit as st
from PIL import Image
import numpy as np
import io

# Fonction pour appliquer la compression LZW (simulée dans ce cas)
def apply_lzw_compression(image):
    # Pillow ne prend pas en charge directement LZW, mais on peut simuler en optimisant le fichier.
    output = io.BytesIO()
    image.save(output, format='TIFF', compression='tiff_lzw')
    output.seek(0)
    return Image.open(output)

# Fonction pour réduire la profondeur des couleurs
def reduce_color_depth(image, bit_depth=4):
    factor = 2 ** (8 - bit_depth)
    array = np.array(image)
    reduced_array = (array // factor) * factor
    return Image.fromarray(reduced_array.astype('uint8'))

# Fonction pour convertir en PNG
def convert_to_png(image):
    output = io.BytesIO()
    image.save(output, format='PNG', optimize=True)
    output.seek(0)
    return output

# Interface utilisateur Streamlit
st.title("Transformation de fichier TIFF")
st.write("Téléversez un fichier TIFF pour appliquer des transformations : compression LZW, réduction de profondeur de couleurs, et conversion en PNG.")

# Téléversement de fichier
tiff_file = st.file_uploader("Choisissez un fichier TIFF", type=["tiff", "tif"])

if tiff_file is not None:
    # Chargement du fichier TIFF
    image = Image.open(tiff_file)

    st.write("### Aperçu de l'image originale")
    st.image(image, caption="Image originale", use_column_width=True)

    # Étape 1 : Compression LZW
    st.write("### Compression LZW")
    compressed_image = apply_lzw_compression(image)
    st.image(compressed_image, caption="Image après compression LZW", use_column_width=True)

    # Étape 2 : Réduction de profondeur des couleurs
    st.write("### Réduction de profondeur des couleurs")
    reduced_image = reduce_color_depth(compressed_image)
    st.image(reduced_image, caption="Image avec profondeur réduite", use_column_width=True)

    # Étape 3 : Conversion en PNG
    st.write("### Conversion en PNG")
    png_file = convert_to_png(reduced_image)

    # Téléchargement du fichier final
    st.download_button(
        label="Télécharger l'image PNG compressée",
        data=png_file,
        file_name="image_compressée.png",
        mime="image/png"
    )






import streamlit as st
import numpy as np
import rasterio
import folium
from streamlit_folium import st_folium
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from folium.plugins import MeasureControl
import geopandas as gpd
from folium.plugins import Draw

# Fonction pour charger un fichier TIFF
def charger_tiff(fichier_tiff):
    try:
        with rasterio.open(fichier_tiff) as src:
            data = src.read(1)  # Lire la première bande
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
            return data, transform, crs, bounds
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier GeoTIFF : {e}")
        return None, None, None, None

# Fonction pour charger un fichier GeoJSON
def charger_geojson(fichier_geojson):
    try:
        gdf = gpd.read_file(fichier_geojson)
        return gdf
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier GeoJSON : {e}")
        return None

# Calcul de la taille d'un pixel
def calculer_taille_pixel(transform):
    return transform[0], -transform[4]

# Taille réelle d'une unité (pixel)
def calculer_taille_unite(bounds_tiff, largeur_pixels, hauteur_pixels):
    point1 = (bounds_tiff[1], bounds_tiff[0])
    point2 = (bounds_tiff[1], bounds_tiff[2])
    distance_x = geodesic(point1, point2).meters

    point3 = (bounds_tiff[3], bounds_tiff[0])
    distance_y = geodesic(point1, point3).meters

    taille_x = distance_x / largeur_pixels
    taille_y = distance_y / hauteur_pixels
    return (taille_x + taille_y) / 2

# Pixels inondés
def calculer_pixels_inondes(data, niveau_inondation):
    return np.sum(data <= niveau_inondation)

# Surface inondée
def calculer_surface_inondee(nombre_pixels_inondes, taille_unite):
    surface_pixel = taille_unite ** 2
    surface_totale_m2 = nombre_pixels_inondes * surface_pixel
    surface_totale_hectares = surface_totale_m2 / 10000
    return surface_totale_m2, surface_totale_hectares

# Génération d'une image de profondeur
def generer_image_profondeur(data_tiff, bounds_tiff, output_path):
    extent = [bounds_tiff[0], bounds_tiff[2], bounds_tiff[1], bounds_tiff[3]]
    plt.figure(figsize=(8, 6))
    plt.imshow(data_tiff, cmap='terrain', extent=extent)
    plt.colorbar(label="Altitude (m)")
    plt.title("Carte de profondeur")
    plt.savefig(output_path, format='png', bbox_inches='tight')
    plt.close()

# Calcul de la surface d'un polygone (en hectares)
def calculer_surface_polygone(geojson_polygon):
    try:
        surface_totale_m2 = geojson_polygon.geometry.area.sum()
        surface_totale_ha = surface_totale_m2 / 10000
        return surface_totale_m2, surface_totale_ha
    except Exception as e:
        st.error(f"Erreur lors du calcul de la surface du polygone : {e}")
        return None, None

# Carte Folium avec superposition
def creer_carte_osm(data_tiff, bounds_tiff, niveau_inondation=None, **geojson_layers):
    lat_min, lon_min = bounds_tiff[1], bounds_tiff[0]
    lat_max, lon_max = bounds_tiff[3], bounds_tiff[2]
    center = [(lat_min + lat_max) / 2, (lon_min + lon_max) / 2]

    m = folium.Map(location=center, zoom_start=13, control_scale=True)
    depth_map_path = "temp_depth_map.png"
    generer_image_profondeur(data_tiff, bounds_tiff, depth_map_path)

    img_overlay = folium.raster_layers.ImageOverlay(
        image=depth_map_path,
        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
        opacity=0.7
    )
    img_overlay.add_to(m)

    if niveau_inondation is not None:
        inondation_mask = data_tiff <= niveau_inondation
        zone_inondee = np.zeros_like(data_tiff, dtype=np.uint8)
        zone_inondee[inondation_mask] = 255

        flood_map_path = "temp_flood_map.png"
        extent = [lon_min, lon_max, lat_min, lat_max]
        plt.figure(figsize=(8, 6))
        plt.imshow(zone_inondee, cmap=ListedColormap(['none', 'magenta']), extent=extent, alpha=0.5)
        plt.axis('off')
        plt.savefig(flood_map_path, format='png', transparent=True, bbox_inches='tight')
        plt.close()

        flood_overlay = folium.raster_layers.ImageOverlay(
            image=flood_map_path,
            bounds=[[lat_min, lon_min], [lat_max, lon_max]],
            opacity=0.6
        )
        flood_overlay.add_to(m)
        # Ajouter les contours magenta foncés
        plt.figure(figsize=(8, 6))
        flipped_zone_inondee = np.flipud(zone_inondee)  # Retourne les données verticalement si nécessaire
        plt.contour(
            flipped_zone_inondee,  # Utiliser les données corrigées
            levels=[127],  # Niveau de contour
            colors='darkmagenta',  # Couleur des contours
            linewidths=1.5,  # Épaisseur des contours
            extent=extent  # Étendue géographique (doit correspondre à votre image)
        )
        plt.axis('off')
        plt.savefig(flood_map_path, format='png', transparent=True, bbox_inches='tight')
        plt.close()
        flood_overlay = folium.raster_layers.ImageOverlay(
            image=flood_map_path,
            bounds=[[lat_min, lon_min], [lat_max, lon_max]],
            opacity=0.6

        )
        flood_overlay.add_to(m)


    
    measure_control = MeasureControl(primary_length_unit='meters', primary_area_unit='sqmeters')
    measure_control = MeasureControl(
        primary_length_unit='meters',
        secondary_length_unit='kilometers',
        primary_area_unit='sqmeters',
        secondary_area_unit='hectares'
    )
    measure_control.add_to(m)


    # Ajouter l'outil de dessin
    draw = Draw(
        export=True,  # Permet l'export des couches dessinées en GeoJSON
        position='topleft',  # Position de l'icône sur la carte
        draw_options={
            'polyline': {'allowIntersection': False},  # Empêche l'intersection des lignes
            'polygon': {'showArea': True},  # Affiche la surface des polygones
            'circle': False,  # Désactive l'outil cercle
            'circlemarker': False,  # Désactive l'outil cercle-marqueur
        },
        edit_options={
            'remove': True  # Permet de supprimer des couches
        }
    )
    draw.add_to(m)
    
    


    # Ajouter les GeoJSON avec des styles spécifiques
    styles = {
        "routes": {"color": "orange", "weight": 2},
        "polygon": {"fillColor": "semi-transparent", "color": "black", "weight": 2},
        "pistes": {"color": "blue", "weight": 2},
        "cours_eau": {"color": "cyan", "weight": 2},
        "batiments": {"fillColor": "red", "color": "red", "weight": 1, "fillOpacity": 0.5},
        "ville": {"fillColor": "green", "color": "green", "weight": 1, "fillOpacity": 0.3},
        "plantations": {"fillColor": "yellow", "color": "yellow", "weight": 1, "fillOpacity": 0.3},
    }

    for layer, geojson_data in geojson_layers.items():
        if geojson_data is not None:
            folium.GeoJson(
                geojson_data,
                style_function=lambda feature, style=styles[layer]: style
            ).add_to(m)

    folium.LayerControl().add_to(m)
    return m

# Interface principale Streamlit
def main():
    st.title("Analyse des zones inondées")
    st.markdown("### Téléchargez les fichiers nécessaires pour visualiser les données.")

    fichier_tiff = st.file_uploader("Fichier GeoTIFF", type=["tif"])
    fichier_geojson_routes = st.file_uploader("GeoJSON (routes)", type=["geojson"])
    fichier_geojson_polygon = st.file_uploader("GeoJSON (polygone)", type=["geojson"])
    fichier_geojson_pistes = st.file_uploader("GeoJSON (pistes)", type=["geojson"])
    fichier_geojson_cours_eau = st.file_uploader("GeoJSON (cours d'eau)", type=["geojson"])
    fichier_geojson_batiments = st.file_uploader("GeoJSON (bâtiments)", type=["geojson"])
    fichier_geojson_ville = st.file_uploader("GeoJSON (ville)", type=["geojson"])
    fichier_geojson_plantations = st.file_uploader("GeoJSON (plantations)", type=["geojson"])

    geojson_data = {
        "routes": charger_geojson(fichier_geojson_routes) if fichier_geojson_routes else None,
        "polygon": charger_geojson(fichier_geojson_polygon) if fichier_geojson_polygon else None,
        "pistes": charger_geojson(fichier_geojson_pistes) if fichier_geojson_pistes else None,
        "cours_eau": charger_geojson(fichier_geojson_cours_eau) if fichier_geojson_cours_eau else None,
        "batiments": charger_geojson(fichier_geojson_batiments) if fichier_geojson_batiments else None,
        "ville": charger_geojson(fichier_geojson_ville) if fichier_geojson_ville else None,
        "plantations": charger_geojson(fichier_geojson_plantations) if fichier_geojson_plantations else None,
    }
    # Calcul du nombre de bâtiments dans l'emprise du polygone
    if fichier_geojson_batiments and fichier_geojson_polygon:
        geojson_batiments = charger_geojson(fichier_geojson_batiments)
        geojson_polygon = charger_geojson(fichier_geojson_polygon)
        if geojson_batiments is not None and geojson_polygon is not None:
            # Vérifier si les CRS sont compatibles
            if geojson_batiments.crs != geojson_polygon.crs:
                geojson_batiments = geojson_batiments.to_crs(geojson_polygon.crs)
                
            # Effectuer l'intersection
            intersection = gpd.overlay(geojson_batiments, geojson_polygon, how='intersection')
            # Compter le nombre de bâtiments
            nombre_batiments = len(intersection)
            st.write(f"Nombre de bâtiments dans l'emprise du polygone : {nombre_batiments}")



    if fichier_tiff:
        data_tiff, transform_tiff, crs_tiff, bounds_tiff = charger_tiff(fichier_tiff)
        #afficher surface polygone
        if fichier_geojson_polygon:
             geojson_polygon = charger_geojson(fichier_geojson_polygon)
             if geojson_polygon is not None:
                 surface_m2, surface_ha = calculer_surface_polygone(geojson_polygon)
                 st.write(f"Surface du polygone : {surface_m2:.2f} m² ({surface_ha:.2f} ha)")
        

        
        if data_tiff is not None:
            st.write(f"Dimensions : {data_tiff.shape}")
            st.write(f"Altitude : min {data_tiff.min()} m, max {data_tiff.max()} m")

            taille_unite = calculer_taille_unite(bounds_tiff, data_tiff.shape[1], data_tiff.shape[0])
            st.write(f"Taille moyenne d'une unité : {taille_unite:.2f} m")

            niveau_inondation = st.slider("Niveau d'inondation", float(data_tiff.min()), float(data_tiff.max()), step=0.1)
            if niveau_inondation:
                pixels_inondes = calculer_pixels_inondes(data_tiff, niveau_inondation)
                surface_m2, surface_ha = calculer_surface_inondee(pixels_inondes, taille_unite)
                st.write(f"Surface inondée : {surface_m2:.2f} m² ({surface_ha:.2f} ha)")

            m = creer_carte_osm(data_tiff, bounds_tiff, niveau_inondation, **geojson_data)
            st_folium(m, width=700, height=500)

if __name__ == "__main__":
    main()















import streamlit as st
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Fonction pour charger le fichier TIFF
def charger_tiff(fichier_tiff):
    try:
        with rasterio.open(fichier_tiff) as src:
            data = src.read(1)  # Lire la première bande
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
            return data, transform, crs, bounds
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier GeoTIFF : {e}")
        return None, None, None, None

# Fonction pour afficher la carte de profondeur
def afficher_carte_profondeur(data_tiff, bounds_tiff):
    # Étendue géographique (extent)
    extent = [bounds_tiff[0], bounds_tiff[2], bounds_tiff[1], bounds_tiff[3]]

    # Créer la figure
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data_tiff, cmap='terrain', extent=extent)
    cbar = fig.colorbar(im, ax=ax, label="Altitude (m)")

    # Titre et axes
    ax.set_title("Carte de profondeur (terrain)", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    st.pyplot(fig)

# Fonction pour afficher la zone inondée en magenta
def afficher_zone_inondee(data_tiff, niveau_inondation, bounds_tiff):
    # Étendue géographique (extent)
    extent = [bounds_tiff[0], bounds_tiff[2], bounds_tiff[1], bounds_tiff[3]]

    # Créer un masque des pixels inondés
    inondation_mask = data_tiff <= niveau_inondation
    nb_pixels_inondes = np.sum(inondation_mask)

    # Créer une nouvelle couche pour la zone inondée (valeurs 1 pour inondées, 0 sinon)
    zone_inondee = np.zeros_like(data_tiff, dtype=np.uint8)
    zone_inondee[inondation_mask] = 1

    # Créer la figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Afficher l'image de fond (altitudes)
    im = ax.imshow(data_tiff, cmap='terrain', extent=extent)
    cbar = fig.colorbar(im, ax=ax, label="Altitude (m)")

    # Superposer la couche des zones inondées en magenta
    cmap = ListedColormap(["none", "red"])
    ax.imshow(
        zone_inondee,
        cmap=cmap,
        alpha=0.5,
        extent=extent
    )

    # Ajouter une légende manuelle pour les zones inondées
    legend_elements = [
        Patch(facecolor='red', edgecolor='none', label='Zone inondée (red)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Titre et axes
    ax.set_title(f"Zone inondée pour une cote de {niveau_inondation:.2f} m", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Afficher le nombre de pixels inondés
    st.write(f"**Nombre de pixels inondés :** {nb_pixels_inondes}")
    st.pyplot(fig)

# Interface Streamlit
def main():
    st.title("Analyse des zones inondées")
    st.markdown("### Téléchargez un fichier GeoTIFF pour analyser les zones inondées.")

    # Téléversement du fichier GeoTIFF
    fichier_tiff = st.file_uploader("Téléchargez un fichier GeoTIFF", type=["tif"])

    if fichier_tiff is not None:
        # Charger le fichier TIFF
        data_tiff, transform_tiff, crs_tiff, bounds_tiff = charger_tiff(fichier_tiff)

        if data_tiff is not None:
            # Afficher les informations de base
            st.write(f"Dimensions : {data_tiff.shape}")
            st.write(f"Altitude min : {data_tiff.min()}, max : {data_tiff.max()}")

            # Afficher la carte de profondeur
            if st.checkbox("Afficher la carte de profondeur"):
                afficher_carte_profondeur(data_tiff, bounds_tiff)

            # Sélectionner le niveau d'inondation
            niveau_inondation = st.slider(
                "Choisissez le niveau d'inondation",
                float(data_tiff.min()),
                float(data_tiff.max()),
                float(np.percentile(data_tiff, 50)),  # Par défaut, la médiane
                step=0.1
            )

            # Bouton pour afficher la zone inondée
            if st.button("Afficher la zone inondée"):
                afficher_zone_inondee(data_tiff, niveau_inondation, bounds_tiff)

if __name__ == "__main__":
    main()

















# Streamlit - Titre de l'application avec deux logos centrés
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.image("POPOPO.jpg", width=150)
with col2:
    st.image("logo.png", width=150)
with col3:
    st.write("")  # Cette colonne est laissée vide pour centrer les logos

st.title("Carte des zones inondées avec niveaux d'eau et surface")

# Initialiser session_state pour stocker les données d'inondation
if 'flood_data' not in st.session_state:
    st.session_state.flood_data = {
        'surface_bleu': None,  
        'volume_eau': None,
        'niveau_inondation': 0.0
    }

# Étape 1 : Sélectionner un site ou téléverser un fichier
st.markdown("## Sélectionner un site ou téléverser un fichier GeoTIFF")
uploaded_tiff_file = st.file_uploader("Téléversez un fichier GeoTIFF (.tif)", type=["tif"])

# Charger les données depuis un fichier GeoTIFF
def charger_tiff(fichier_tiff):
    try:
        with rasterio.open(fichier_tiff) as src:
            # Lire les métadonnées et les données raster
            data = src.read(1)  # Lire la première bande
            transform = src.transform  # Transformation spatiale
            crs = src.crs  # Système de coordonnées
            return data, transform, crs
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier GeoTIFF : {e}")
        return None, None, None

# Si un fichier GeoTIFF est téléversé
if uploaded_tiff_file is not None:
    data_tiff, transform_tiff, crs_tiff = charger_tiff(uploaded_tiff_file)

    if data_tiff is not None:
        st.write("**Informations sur le fichier GeoTIFF :**")
        st.write(f"Dimensions : {data_tiff.shape}")
        st.write(f"Valeurs min : {data_tiff.min()}, max : {data_tiff.max()}")
        st.write(f"Système de coordonnées : {crs_tiff}")

        # Afficher les données raster sous forme d'image
        fig, ax = plt.subplots(figsize=(8, 6))
        extent = (
            transform_tiff[2],  # Min X
            transform_tiff[2] + transform_tiff[0] * data_tiff.shape[1],  # Max X
            transform_tiff[5] + transform_tiff[4] * data_tiff.shape[0],  # Min Y
            transform_tiff[5]  # Max Y
        )
        cax = ax.imshow(data_tiff, cmap='terrain', extent=extent)
        fig.colorbar(cax, ax=ax, label="Altitude (m)")
        ax.set_title("Carte d'altitude (GeoTIFF)")
        st.pyplot(fig)

        # Niveau d'eau et analyse
        st.session_state.flood_data['niveau_inondation'] = st.number_input(
            "Entrez le niveau d'eau (mètres)", min_value=float(data_tiff.min()), max_value=float(data_tiff.max()), step=0.1
        )

        if st.button("Calculer et afficher la zone inondée"):
            # Calculer la zone inondée
            inondation_mask = data_tiff <= st.session_state.flood_data['niveau_inondation']
            surface_inondee = np.sum(inondation_mask) * (transform_tiff[0] * transform_tiff[4]) / 10_000  # En hectares
            st.session_state.flood_data['surface_bleu'] = surface_inondee
            st.write(f"**Surface inondée :** {surface_inondee:.2f} hectares")

            # Afficher la carte de l'inondation
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(data_tiff, cmap='terrain', extent=extent)
            ax.imshow(inondation_mask, cmap='Blues', alpha=0.5, extent=extent)
            ax.set_title("Zone inondée (en bleu)")
            fig.colorbar(cax, ax=ax, label="Altitude (m)")
            st.pyplot(fig)


st.title("Carte des zones inondées avec niveaux d'eau et surface")

# Initialiser session_state pour stocker les données d'inondation
if 'flood_data' not in st.session_state:
    st.session_state.flood_data = {
        'surface_bleu': None,  
        'volume_eau': None,
        'niveau_inondation': 0.0
    }

# Étape 1 : Sélectionner un site ou téléverser un fichier
st.markdown("## Sélectionner un site ou téléverser un fichier")
option_site = st.selectbox("Sélectionnez un site", ("Aucun", "AYAME 1", "AYAME 2"))
uploaded_file = st.file_uploader("Téléversez un fichier Excel ou TXT", type=["xlsx", "txt"])

# Charger les données en fonction de l'option sélectionnée
def charger_fichier(fichier, is_uploaded=False):
    try:
        if is_uploaded:
            if fichier.name.endswith('.xlsx'):
                df = pd.read_excel(fichier)
            elif fichier.name.endswith('.txt'):
                df = pd.read_csv(fichier, sep=",", header=None, names=["X", "Y", "Z"])
        else:
            df = pd.read_csv(fichier, sep=",", header=None, names=["X", "Y", "Z"])
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {e}")
        return None

if option_site == "AYAME 1":
    df = charger_fichier('AYAME1.txt')
elif option_site == "AYAME 2":
    df = charger_fichier('AYAME2.txt')
elif uploaded_file is not None:
    df = charger_fichier(uploaded_file, is_uploaded=True)
else:
    st.warning("Veuillez sélectionner un site ou téléverser un fichier pour démarrer.")
    df = None




uploaded_geojson_file = st.file_uploader("Téléversez un fichier GeoJSON pour les routes", type=["geojson"])
def charger_geojson(fichier):
    try:
        gdf = gpd.read_file(fichier)
        return gdf
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier GeoJSON : {e}")
        return None

# Charger les données du fichier GeoJSON des routes
routes_gdf = None
if uploaded_geojson_file is not None:
    routes_gdf = charger_geojson(uploaded_geojson_file)

# Charger et filtrer les bâtiments dans l'emprise de la carte
try:
    batiments_gdf = gpd.read_file("batiments2.geojson")
    if df is not None:
        emprise = box(df['X'].min(), df['Y'].min(), df['X'].max(), df['Y'].max())
        batiments_gdf = batiments_gdf.to_crs(epsg=32630)
        batiments_dans_emprise = batiments_gdf[batiments_gdf.intersects(emprise)]
    else:
        batiments_dans_emprise = None
except Exception as e:
    st.error(f"Erreur lors du chargement des bâtiments : {e}")
    batiments_dans_emprise = None

# Traitement des données si le fichier est chargé
if df is not None:
    st.markdown("---")

    # Vérification du fichier : colonnes X, Y, Z
    if 'X' not in df.columns or 'Y' not in df.columns or 'Z' not in df.columns:
        st.error("Erreur : colonnes 'X', 'Y' et 'Z' manquantes.")
    else:
        st.session_state.flood_data['niveau_inondation'] = st.number_input("Entrez le niveau d'eau (mètres)", min_value=0.0, step=0.1)
        interpolation_method = st.selectbox("Méthode d'interpolation", ['linear', 'nearest'])

        X_min, X_max = df['X'].min(), df['X'].max()
        Y_min, Y_max = df['Y'].min(), df['Y'].max()
        resolution = st.number_input("Résolution de la grille", value=300, min_value=100, max_value=1000)
        grid_X, grid_Y = np.mgrid[X_min:X_max:resolution*1j, Y_min:Y_max:resolution*1j]
        grid_Z = griddata((df['X'], df['Y']), df['Z'], (grid_X, grid_Y), method=interpolation_method)

        def calculer_surface_bleue(niveau_inondation):
            return np.sum((grid_Z <= niveau_inondation)) * (grid_X[1, 0] - grid_X[0, 0]) * (grid_Y[0, 1] - grid_Y[0, 0]) / 10000

        def calculer_volume(surface_bleue):
            return surface_bleue * st.session_state.flood_data['niveau_inondation'] * 10000

        if st.button("Afficher la carte d'inondation"):
            surface_bleue = calculer_surface_bleue(st.session_state.flood_data['niveau_inondation'])
            volume_eau = calculer_volume(surface_bleue)
            st.session_state.flood_data['surface_bleu'] = surface_bleue
            st.session_state.flood_data['volume_eau'] = volume_eau

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlim(X_min, X_max)
            ax.set_ylim(Y_min, Y_max)
            ctx.add_basemap(ax, crs="EPSG:32630", source=ctx.providers.OpenStreetMap.Mapnik)
            # Ajouter des coordonnées sur les quatre côtés
            ax.tick_params(axis='both', which='both', direction='in', length=6, width=1, color='black', labelsize=10)
            ax.set_xticks(np.linspace(X_min, X_max, num=5))# Coordonnées sur l'axe X
            ax.set_yticks(np.linspace(Y_min, Y_max, num=5))# Coordonnées sur l'axe Y
            ax.xaxis.set_tick_params(labeltop=True)# Affiche les labels sur le haut
            ax.yaxis.set_tick_params(labelright=True)# Affiche les labels à droite
            
            # Ajouter les lignes pour relier les tirets (lignes horizontales et verticales)
            # Lignes verticales (de haut en bas)
            for x in np.linspace(X_min, X_max, num=5):
                ax.axvline(x, color='black', linewidth=0.5, linestyle='--',alpha=0.2)
            # Lignes horizontales (de gauche à droite)
            for y in np.linspace(Y_min, Y_max, num=5):
                ax.axhline(y, color='black', linewidth=0.5, linestyle='--',alpha=0.2)

            # Ajouter les croisillons aux intersections avec opacité à 100%
            # Déterminer les positions d'intersection
            intersections_x = np.linspace(X_min, X_max, num=5)
            intersections_y = np.linspace(Y_min, Y_max, num=5)
            # Tracer les croisillons aux intersections avec opacité à 100%
            for x in intersections_x:
                for y in intersections_y:
                    ax.plot(x, y, 'k+', markersize=7, alpha=1.0) # 'k+' : plus noire, alpha=1 pour opacité 100%
                    


            # Tracer la zone inondée avec les contours
            contours_inondation = ax.contour(grid_X, grid_Y, grid_Z, levels=[st.session_state.flood_data['niveau_inondation']], colors='red', linewidths=1)
            ax.clabel(contours_inondation, inline=True, fontsize=10, fmt='%1.1f m')
            ax.contourf(grid_X, grid_Y, grid_Z, levels=[-np.inf, st.session_state.flood_data['niveau_inondation']], colors='#007FFF', alpha=0.5)

            # Transformer les contours en polygones pour analyser les bâtiments
            contour_paths = [Polygon(path.vertices) for collection in contours_inondation.collections for path in collection.get_paths()]
            zone_inondee = gpd.GeoDataFrame(geometry=[MultiPolygon(contour_paths)], crs="EPSG:32630")

            # Filtrer et afficher tous les bâtiments
            if batiments_dans_emprise is not None:
                batiments_dans_emprise.plot(ax=ax, facecolor='grey', edgecolor='black', linewidth=0.5, alpha=0.6, label="Bâtiments non inondés")
                
                # Séparer les bâtiments inondés
                batiments_inondes = batiments_dans_emprise[batiments_dans_emprise.intersects(zone_inondee.unary_union)]
                nombre_batiments_inondes = len(batiments_inondes)

                # Afficher les bâtiments inondés en rouge
                batiments_inondes.plot(ax=ax, facecolor='red', edgecolor='red', linewidth=1, alpha=0.8, label="Bâtiments inondés")

                st.write(f"Nombre de bâtiments dans la zone inondée : {nombre_batiments_inondes}")
                ax.legend()
            else:
                st.write("Aucun bâtiment à analyser dans cette zone.")

            if routes_gdf is not None:
                routes_gdf = routes_gdf.to_crs(epsg=32630)  # Reprojeter les données si nécessaire
                routes_gdf.plot(ax=ax, color='orange', linewidth=2, label="Routes")
                st.write(f"**Nombre de routes affichées :** {len(routes_gdf)}")

            

            

            st.pyplot(fig)

            # Enregistrer les contours en fichier DXF
            doc = ezdxf.new(dxfversion='R2010')
            msp = doc.modelspace()
            for collection in contours_inondation.collections:
                for path in collection.get_paths():
                    points = path.vertices
                    for i in range(len(points)-1):
                        msp.add_line(points[i], points[i+1])

            dxf_file = "contours_inondation.dxf"
            doc.saveas(dxf_file)
            carte_file = "carte_inondation.png"
            fig.savefig(carte_file)

            with open(carte_file, "rb") as carte:
                st.download_button(label="Télécharger la carte", data=carte, file_name=carte_file, mime="image/png")

            with open(dxf_file, "rb") as dxf:
                st.download_button(label="Télécharger le fichier DXF", data=dxf, file_name=dxf_file, mime="application/dxf")

            # Afficher les résultats
            now = datetime.now()
            st.markdown("## Résultats")
            st.write(f"**Surface inondée :** {surface_bleue:.2f} hectares")
            st.write(f"**Volume d'eau :** {volume_eau:.2f} m³")
            st.write(f"**Niveau d'eau :** {st.session_state.flood_data['niveau_inondation']} m")
            st.write(f"**Nombre de bâtiments inondés :** {nombre_batiments_inondes}")
            st.write(f"**Date :** {now.strftime('%Y-%m-%d')}")
            st.write(f"**Heure :** {now.strftime('%H:%M:%S')}")
            st.write(f"**Système de projection :** EPSG:32630")

# Fonction pour générer la carte de profondeur avec dégradé de couleurs
def generate_depth_map(label_rotation_x=0, label_rotation_y=0):

    # Détection des bas-fonds
    def detecter_bas_fonds(grid_Z, seuil_rel_bas_fond=1.5):
        """
        Détermine les bas-fonds en fonction de la profondeur Z relative.
        Bas-fond = Z < moyenne(Z) - seuil_rel_bas_fond * std(Z)
        """
        moyenne_Z = np.mean(grid_Z)
        ecart_type_Z = np.std(grid_Z)
        seuil_bas_fond = moyenne_Z - seuil_rel_bas_fond * ecart_type_Z
        bas_fonds = grid_Z < seuil_bas_fond
        return bas_fonds, seuil_bas_fond

    # Calcul des surfaces des bas-fonds
    def calculer_surface_bas_fond(bas_fonds, grid_X, grid_Y):
        """
        Calcule la surface des bas-fonds en hectares.
        """
        resolution = (grid_X[1, 0] - grid_X[0, 0]) * (grid_Y[0, 1] - grid_Y[0, 0]) / 10000  # Résolution en hectares
        surface_bas_fond = np.sum(bas_fonds) * resolution
        return surface_bas_fond

    bas_fonds, seuil_bas_fond = detecter_bas_fonds(grid_Z)
    surface_bas_fond = calculer_surface_bas_fond(bas_fonds, grid_X, grid_Y)

    
    # Appliquer un dégradé de couleurs sur la profondeur (niveau de Z)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(X_min, X_max)
    ax.set_ylim(Y_min, Y_max)
    ctx.add_basemap(ax, crs="EPSG:32630", source=ctx.providers.OpenStreetMap.Mapnik)
    ax.tick_params(axis='both', which='both', direction='in', length=6, width=1, color='black', labelsize=10)
    ax.set_xticks(np.linspace(X_min, X_max, num=5))
    ax.set_yticks(np.linspace(Y_min, Y_max, num=5))
    ax.xaxis.set_tick_params(labeltop=True)
    ax.yaxis.set_tick_params(labelright=True)

     # Masquer les coordonnées aux extrémités
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    ax.set_xticklabels(
         ["" if x == X_min or x == X_max else f"{int(x)}" for x in xticks],
        rotation=label_rotation_x,
    )
    ax.set_yticklabels(
        ["" if y == Y_min or y == Y_max else f"{int(y)}" for y in yticks],
        rotation=label_rotation_y,
        va="center"  # Alignement vertical des étiquettes Y
    )
    #modifier rotation
    for label in ax.get_xticklabels():
        label.set_rotation(label_rotation_x)

    for label in ax.get_yticklabels():
        label.set_rotation(label_rotation_y)

    

    # Ajouter les contours pour la profondeur
    depth_levels = np.linspace(grid_Z.min(), grid_Z.max(), 100)
    cmap = plt.cm.plasma  # Couleurs allant de bleu à jaune
    cont = ax.contourf(grid_X, grid_Y, grid_Z, levels=depth_levels, cmap=cmap)
    cbar = plt.colorbar(cont, ax=ax)
    cbar.set_label('Profondeur (m)', rotation=270)

    # Ajouter les bas-fonds en cyan
    ax.contourf(grid_X, grid_Y, bas_fonds, levels=[0.5, 1], colors='cyan', alpha=0.4, label='Bas-fonds')
    
    # Ajouter une ligne de contour autour des bas-fonds
    contour_lines = ax.contour(
        grid_X, grid_Y, grid_Z,
        levels=[seuil_bas_fond],  # Niveau correspondant au seuil des bas-fonds
        colors='black',  # Couleur des contours
        linewidths=1.5,
        linestyles='solid',# Épaisseur de la ligne
    )
    # Ajouter des labels pour les contours
    ax.clabel(contour_lines,
        inline=True,
        fmt={seuil_bas_fond: f"{seuil_bas_fond:.2f} m"},  # Format du label
        fontsize=12
    )


    # Ajouter des lignes pour relier les tirets
    for x in np.linspace(X_min, X_max, num=5):
        ax.axvline(x, color='black', linewidth=0.5, linestyle='--', alpha=0.2)
    for y in np.linspace(Y_min, Y_max, num=5):
        ax.axhline(y, color='black', linewidth=0.5, linestyle='--', alpha=0.2)
#croisillon 
    intersections_x = np.linspace(X_min, X_max, num=5)
    intersections_y = np.linspace(Y_min, Y_max, num=5)
    for x in intersections_x:
        for y in intersections_y:
            ax.plot(x, y, 'k+', markersize=7, alpha=1.0)

    


    # Ajouter les bâtiments
    if batiments_dans_emprise is not None:
        batiments_dans_emprise.plot(ax=ax, facecolor='grey', edgecolor='black', linewidth=0.5, alpha=0.6)

    # Affichage de la carte de profondeur
    st.pyplot(fig)
    # Afficher les surfaces calculées
    st.write(f"**Surface des bas-fonds** : {surface_bas_fond:.2f} hectares")

# Ajouter un bouton pour générer la carte de profondeur
if st.button("Générer la carte de profondeur avec bas-fonds"):
    generate_depth_map(label_rotation_x=0, label_rotation_y=-90)







# Fonction pour charger les polygones
def charger_polygones(uploaded_file):
    try:
        if uploaded_file is not None:
            # Lire le fichier GeoJSON téléchargé
            polygones_gdf = gpd.read_file(uploaded_file)
            
            # Convertir le GeoDataFrame au CRS EPSG:32630
            polygones_gdf = polygones_gdf.to_crs(epsg=32630)
            
            # Créer une emprise (bounding box) basée sur les données
            if 'X' in df.columns and 'Y' in df.columns:
                emprise = box(df['X'].min(), df['Y'].min(), df['X'].max(), df['Y'].max())
                polygones_dans_emprise = polygones_gdf[polygones_gdf.intersects(emprise)]  # Filtrer les polygones dans l'emprise
            else:
                polygones_dans_emprise = polygones_gdf  # Si pas de colonne X/Y dans df, prendre tous les polygones
        else:
            polygones_dans_emprise = None
    except Exception as e:
        st.error(f"Erreur lors du chargement des polygones : {e}")
        polygones_dans_emprise = None

    return polygones_dans_emprise

# Fonction pour afficher les polygones
def afficher_polygones(ax, gdf_polygones, edgecolor='white', linewidth=1.0):
    if gdf_polygones is not None and not gdf_polygones.empty:
        gdf_polygones.plot(ax=ax, facecolor='none', edgecolor=edgecolor, linewidth=linewidth)
    else:
        st.warning("Aucun polygone à afficher dans l'emprise.")

# Exemple d'appel dans l'interface Streamlit
st.title("Affichage des Polygones et Profondeur")

# Téléchargement du fichier GeoJSON pour les polygones
uploaded_file = st.file_uploader("Téléverser un fichier GeoJSON", type="geojson")



def calculer_surface_bas_fonds_polygones(polygones, bas_fonds, grid_X, grid_Y):
    try:
        # Conversion des bas-fonds en GeoDataFrame
        resolution = (grid_X[1, 0] - grid_X[0, 0]) * (grid_Y[0, 1] - grid_Y[0, 0])
        bas_fonds_coords = [
            Polygon([
                (grid_X[i, j], grid_Y[i, j]),
                (grid_X[i + 1, j], grid_Y[i + 1, j]),
                (grid_X[i + 1, j + 1], grid_Y[i + 1, j + 1]),
                (grid_X[i, j + 1], grid_Y[i, j + 1])
            ])
            for i in range(grid_X.shape[0] - 1)
            for j in range(grid_X.shape[1] - 1)
            if bas_fonds[i, j]
        ]
        bas_fonds_gdf = gpd.GeoDataFrame(geometry=bas_fonds_coords, crs="EPSG:32630")

        # Intersection entre bas-fonds et polygones
        intersection = gpd.overlay(polygones, bas_fonds_gdf, how="intersection")
        
        # Calcul de la surface totale
        surface_totale = intersection.area.sum() / 10_000  # Convertir en hectares
        return surface_totale
    except Exception as e:
        st.error(f"Erreur dans le calcul de la surface des bas-fonds : {e}")
        return 0


# Définir la fonction detecter_bas_fonds en dehors de generate_depth_map
def detecter_bas_fonds(grid_Z, seuil_rel_bas_fond=1.5):
    moyenne_Z = np.mean(grid_Z)
    ecart_type_Z = np.std(grid_Z)
    seuil_bas_fond = moyenne_Z - seuil_rel_bas_fond * ecart_type_Z
    bas_fonds = grid_Z < seuil_bas_fond
    return bas_fonds, seuil_bas_fond

# Définir la fonction calculer_surface_bas_fond en dehors de generate_depth_map
def calculer_surface_bas_fond(bas_fonds, grid_X, grid_Y):
    resolution = (grid_X[1, 0] - grid_X[0, 0]) * (grid_Y[0, 1] - grid_Y[0, 0]) / 10000  # Résolution en hectares
    surface_bas_fond = np.sum(bas_fonds) * resolution
    return surface_bas_fond

# Fonction pour générer la carte de profondeur
def generate_depth_map(ax, grid_Z, grid_X, grid_Y, X_min, X_max, Y_min, Y_max, label_rotation_x=0, label_rotation_y=0):
    # Appliquer un dégradé de couleurs sur la profondeur (niveau de Z)
    ax.set_xlim(X_min, X_max)
    ax.set_ylim(Y_min, Y_max)

    # Afficher la carte de fond OpenStreetMap en EPSG:32630
    ctx.add_basemap(ax, crs="EPSG:32630", source=ctx.providers.OpenStreetMap.Mapnik)

    ax.tick_params(axis='both', which='both', direction='in', length=6, width=1, color='black', labelsize=10)
    ax.set_xticks(np.linspace(X_min, X_max, num=5))
    ax.set_yticks(np.linspace(Y_min, Y_max, num=5))
    ax.xaxis.set_tick_params(labeltop=True)
    ax.yaxis.set_tick_params(labelright=True)

    # Masquer les coordonnées aux extrémités
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    ax.set_xticklabels(
        ["" if x == X_min or x == X_max else f"{int(x)}" for x in xticks],
        rotation=label_rotation_x,
    )
    ax.set_yticklabels(
        ["" if y == Y_min or y == Y_max else f"{int(y)}" for y in yticks],
        rotation=label_rotation_y,
        va="center"  # Alignement vertical des étiquettes Y
    )

    # Modifier rotation
    for label in ax.get_xticklabels():
        label.set_rotation(label_rotation_x)

    for label in ax.get_yticklabels():
        label.set_rotation(label_rotation_y)

    # Ajouter les contours pour la profondeur et Barre verticale
    depth_levels = np.linspace(grid_Z.min(), grid_Z.max(), 100)
    cmap = plt.cm.plasma  # Couleurs allant de bleu à jaune
    cont = ax.contourf(grid_X, grid_Y, grid_Z, levels=depth_levels, cmap=cmap)
    cbar = plt.colorbar(cont, ax=ax)
    cbar.set_label('Profondeur (m)', rotation=270, labelpad=20)

    # Ajouter les bas-fonds en cyan
    bas_fonds, seuil_bas_fond = detecter_bas_fonds(grid_Z)  # Appel à la fonction externe
    ax.contourf(grid_X, grid_Y, bas_fonds, levels=[0.5, 1], colors='cyan', alpha=0.4, label='Bas-fonds')

    # Ajouter une ligne de contour autour des bas-fonds
    contour_lines = ax.contour(
        grid_X, grid_Y, grid_Z,
        levels=[seuil_bas_fond],  # Niveau correspondant au seuil des bas-fonds
        colors='black',  # Couleur des contours
        linewidths=1.5,
        linestyles='solid',
    )
    intersections_x = np.linspace(X_min, X_max, num=5)
    intersections_y = np.linspace(Y_min, Y_max, num=5)
    for x in intersections_x:
        for y in intersections_y:
            ax.plot(x, y, 'k+', markersize=7, alpha=1.0)

    # Ajouter des labels pour les contours
    ax.clabel(contour_lines, inline=True, fmt={seuil_bas_fond: f"{seuil_bas_fond:.2f} m"}, fontsize=12, colors='white')

    # Ajouter des lignes pour relier les tirets
    for x in np.linspace(X_min, X_max, num=5):
        ax.axvline(x, color='black', linewidth=0.5, linestyle='--', alpha=0.2)
    for y in np.linspace(Y_min, Y_max, num=5):
        ax.axhline(y, color='black', linewidth=0.5, linestyle='--', alpha=0.2)

    # Affichage de la carte de profondeur
    surface_bas_fond = calculer_surface_bas_fond(bas_fonds, grid_X, grid_Y)
    st.write(f"**Surface des bas-fonds** : {surface_bas_fond:.2f} hectares")
    # Afficher la surface des bas-fonds dans les polygones
    st.write(f"**Surface des bas-fonds dans les polygones** : {surface_bas_fond_polygones:.2f} hectares")

    
    # Ajouter des labels sous l'emprise de la carte de profondeur
    label_y_position = Y_min - (Y_max - Y_min) * 0.10
    ax.text(
        X_min + (X_max - X_min) * 0,  # Position horizontale (10% de la largeur)
        label_y_position,
        f"Surface des bas-fonds :",
        fontsize=12,
        color="black",
        ha="left",  # Aligné à gauche
        va="top",# Aligné en haut
        fontweight='bold',
    )
    ax.text(
        X_min + (X_max - X_min) * 0.37,  # Position horizontale (10% de la largeur)
        label_y_position - (Y_max - Y_min) * 0,  # Légèrement plus bas
        f"{surface_bas_fond:.2f} hectares",
        fontsize=12,
        color="black",
        ha="left",  # Aligné à gauche
        va="top",   # Aligné en haut
    )
    
    ax.text(
        X_min + (X_max - X_min) * 0,  # Position horizontale (10% de la largeur)
        label_y_position - (Y_max - Y_min) * 0.10,  # Légèrement plus bas
        f"Surface des bas-fonds dans les polygones :",
        fontsize=12,
        color="black",
        ha="left",  # Aligné à gauche
        va="top",
        fontweight='bold',# Aligné en haut
    )
    ax.text(
        X_min + (X_max - X_min) * 0.67,  # Position horizontale (10% de la largeur)
        label_y_position - (Y_max - Y_min) * 0.10,  # Légèrement plus bas
       f"{surface_bas_fond_polygones:.2f} hectares",
        fontsize=12,
        color="black",
        ha="left",  # Aligné à gauche
        va="top",# Aligné en haut
    )
    ax.text(
        X_min + (X_max - X_min) * 0,  # Position horizontale (10% de la largeur)
        label_y_position - (Y_max - Y_min) * 0.20,  # Légèrement plus bas
        f"Cote du bafond :",
        fontsize=12,
        color="black",
        ha="left",  # Aligné à gauche
        va="top",
        fontweight='bold',# Aligné en haut
    )
    ax.text(
        X_min + (X_max - X_min) * 0.26,  # Position horizontale (10% de la largeur)
        label_y_position - (Y_max - Y_min) * 0.20,  # Légèrement plus bas
        f"{seuil_bas_fond:.2f} m",
        fontsize=12,
        color="black",
        ha="left",  # Aligné à gauche
        va="top",# Aligné en haut
    )
    


# Ajouter les polygones sur la carte
if st.button("Afficher les polygones"):
    # Charger les polygones
    polygones_dans_emprise = charger_polygones(uploaded_file)

    # Si des polygones sont chargés, utiliser leur emprise pour ajuster les limites
    if polygones_dans_emprise is not None:
        # Calculer les limites du polygone
        X_min_polygone, Y_min_polygone, X_max_polygone, Y_max_polygone = polygones_dans_emprise.total_bounds
        
        # Calculer les limites de la carte de profondeur
        X_min_depth, Y_min_depth, X_max_depth, Y_max_depth = grid_X.min(), grid_Y.min(), grid_X.max(), grid_Y.max()

        # Vérifier si l'emprise de la carte de profondeur couvre celle des polygones
        if (X_min_depth <= X_min_polygone and X_max_depth >= X_max_polygone and
            Y_min_depth <= Y_min_polygone and Y_max_depth >= Y_max_polygone):
            X_min, Y_min, X_max, Y_max = X_min_depth, Y_min_depth, X_max_depth, Y_max_depth
        else:
            marge = 0.1
            X_range = X_max_polygone - X_min_polygone
            Y_range = Y_max_polygone - Y_min_polygone
            
            X_min = min(X_min_depth, X_min_polygone - X_range * marge)
            Y_min = min(Y_min_depth, Y_min_polygone - Y_range * marge)
            X_max = max(X_max_depth, X_max_polygone + X_range * marge)
            Y_max = max(Y_max_depth, Y_max_polygone + Y_range * marge)

        # Calculer les bas-fonds
        bas_fonds, _ = detecter_bas_fonds(grid_Z)

        # Calculer la surface des bas-fonds à l'intérieur des polygones
        surface_bas_fond_polygones = calculer_surface_bas_fonds_polygones(
            polygones_dans_emprise, bas_fonds, grid_X, grid_Y
        )

        # Affichage de la carte
        fig, ax = plt.subplots(figsize=(10, 10))
        generate_depth_map(ax, grid_Z, grid_X, grid_Y, X_min, X_max, Y_min, Y_max, label_rotation_x=0, label_rotation_y=-90)
        afficher_polygones(ax, polygones_dans_emprise)
        st.pyplot(fig)

        
        

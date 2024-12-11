
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
import folium
from streamlit_folium import st_folium
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from folium.plugins import MeasureControl
import geopandas as gpd

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

def generer_zones_inondees(data_tiff, transform_tiff, niveau_inondation):
    # Extraire les pixels inondés
    inonde = data_tiff <= niveau_inondation
    indices = np.argwhere(inonde)  # Indices des pixels inondés
    
    # Convertir les indices des pixels en coordonnées géographiques (limites des pixels)
    polygons = []
    for row, col in indices:
        x_min, y_max = rasterio.transform.xy(transform_tiff, row, col, offset="ul")
        x_max, y_min = rasterio.transform.xy(transform_tiff, row + 1, col + 1, offset="ul")
        polygons.append(box(x_min, y_min, x_max, y_max))  # Crée une polygonale pour le pixel inondé
    
    # Combiner toutes les polygonales inondées
    zones_inondees = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polygons))
    zones_inondees = zones_inondees.dissolve()  # Fusionner toutes les zones
    return zones_inondees

# Fonction principale pour calculer la surface inondée dans l'emprise de la polygonale téléversée
def calculer_surface_inondee_dans_emprise(data_tiff, transform_tiff, geojson_polygon, niveau_inondation):
    # Générer les zones inondées à partir des données du GeoTIFF
    zones_inondees = generer_zones_inondees(data_tiff, transform_tiff, niveau_inondation)
    
    # Charger la polygonale téléversée
    polygon_emprise = gpd.read_file(geojson_polygon)
    
    # Vérifier si les CRS sont compatibles
    if zones_inondees.crs != polygon_emprise.crs:
        polygon_emprise = polygon_emprise.to_crs(zones_inondees.crs)
    
    # Effectuer l'intersection entre les zones inondées et l'emprise
    intersection = gpd.overlay(zones_inondees, polygon_emprise, how="intersection")
    
    # Calculer la surface totale inondée
    surface_inondee = intersection.geometry.area.sum()  # La surface sera dans les unités du CRS (souvent m²)
    return surface_inondee


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
    measure_control.add_to(m)

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
    def main():
        st.title("Calcul de la surface inondée dans l'emprise")

        if fichier_tiff and fichier_geojson_polygon:
            data_tiff, transform_tiff, crs_tiff, bounds_tiff = charger_tiff(fichier_tiff)
            geojson_polygon = charger_geojson(fichier_geojson_polygon)

            if data_tiff is not None and geojson_polygon is not None:
                niveau_inondation = st.slider("Niveau d'inondation", float(data_tiff.min()), float(data_tiff.max()), step=0.1)
                surface_inondee = calculer_surface_inondee_dans_emprise(data_tiff, transform_tiff, fichier_geojson_polygon, niveau_inondation)
                if surface_inondee is not None:
                    st.write(f"Surface inondée dans l'emprise : {surface_inondee:.2f} m²")
            
            
    
    
    
    
    
    
        
        
        
        
            
            
            # Calcul de la surface inondée dans l'emprise de la polygonale
            
            
            
                

    


    







   
        
        

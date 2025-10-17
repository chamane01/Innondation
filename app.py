import streamlit as st 
import os
import rasterio
import rasterio.merge
import rasterio.mask
import rasterio.warp
import folium
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from streamlit_folium import st_folium
from folium.plugins import Draw, MiniMap, Fullscreen, MeasureControl, MousePosition
from io import BytesIO
from datetime import date, datetime
import base64
import contextily as ctx
import json
import pandas as pd
import zipfile
import shutil

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph
from reportlab.lib import colors

from shapely.geometry import shape, Point, LineString, Polygon

# ============================================================================
# CONFIGURATION DE LA PAGE
# ============================================================================
st.set_page_config(
    page_title="CartoTools Pro",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dimensions PDF
PAGE_WIDTH, PAGE_HEIGHT = A4
SECTION_HEIGHT = PAGE_HEIGHT / 3
COLUMN_WIDTH = PAGE_WIDTH / 2

# ============================================================================
# FONCTIONS UTILITAIRES - INITIALISATION
# ============================================================================

def initialize_session_state():
    """Initialise toutes les variables de session nécessaires"""
    defaults = {
        "analysis_mode": "none",
        "raw_drawings": [],
        "analysis_results": [],
        "elements": [],
        "saved_projects": {},
        "statistics": {},
        "layer_visibility": {},
        "measurement_data": [],
        "annotations": [],
        "color_scheme": "terrain",
        "contour_levels": 15,
        "profile_resolution": 50,
        "basemap_opacity": 0.5,
        "export_format": "PNG",
        "show_hillshade": False,
        "show_slope": True,
        "current_project": None,
        "use_uploaded_tiff": False,
        "uploaded_tiff_path": None,
        "uploaded_shapefiles": [],
        "uploaded_csv_data": None,
        "active_mosaic": "default",
        "shapefile_layers": []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ============================================================================
# FONCTIONS UTILITAIRES - CALCULS GÉOGRAPHIQUES
# ============================================================================

def haversine(lon1, lat1, lon2, lat2):
    """Calcule la distance en mètres entre deux points GPS"""
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return c * 6371000

def calculate_area(coords):
    """Calcule l'aire d'un polygone en m²"""
    try:
        poly = Polygon(coords)
        center_lat = sum(c[1] for c in coords) / len(coords)
        meters_per_degree = 111320 * math.cos(math.radians(center_lat))
        return poly.area * meters_per_degree * meters_per_degree
    except:
        return 0

def get_elevation_stats(data, nodata=None):
    """Calcule les statistiques d'élévation"""
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)
    
    return {
        "min": float(np.nanmin(data)),
        "max": float(np.nanmax(data)),
        "mean": float(np.nanmean(data)),
        "median": float(np.nanmedian(data)),
        "std": float(np.nanstd(data)),
        "range": float(np.nanmax(data) - np.nanmin(data))
    }

# ============================================================================
# FONCTIONS UTILITAIRES - GESTION DES FICHIERS UPLOADÉS
# ============================================================================

def save_uploaded_file(uploaded_file, folder="uploads"):
    """Sauvegarde un fichier uploadé et retourne le chemin"""
    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        file_path = os.path.join(folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"❌ Erreur lors de la sauvegarde du fichier: {e}")
        return None

def validate_tiff_file(file_path):
    """Valide qu'un fichier TIFF est correct"""
    try:
        with rasterio.open(file_path) as src:
            if src.count < 1:
                return False, "Le fichier TIFF doit contenir au moins une bande"
            if src.crs is None:
                return False, "Le fichier TIFF doit avoir un système de coordonnées (CRS)"
            return True, "Fichier TIFF valide"
    except Exception as e:
        return False, f"Erreur de validation: {e}"

def load_shapefile(shp_path):
    """Charge un shapefile et retourne les géométries"""
    try:
        import geopandas as gpd
        gdf = gpd.read_file(shp_path)
        
        # Convertir en EPSG:4326 si nécessaire
        if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        
        return gdf, None
    except ImportError:
        return None, "❌ GeoPandas non installé. Exécutez: pip install geopandas"
    except Exception as e:
        return None, f"❌ Erreur lors du chargement du shapefile: {e}"

def load_csv_points(csv_path, lon_col="longitude", lat_col="latitude"):
    """Charge des points depuis un CSV"""
    try:
        df = pd.read_csv(csv_path)
        
        # Vérifier les colonnes requises
        if lon_col not in df.columns or lat_col not in df.columns:
            available_cols = ", ".join(df.columns.tolist())
            return None, f"❌ Colonnes requises: '{lon_col}' et '{lat_col}'. Colonnes disponibles: {available_cols}"
        
        # Supprimer les lignes avec des coordonnées manquantes
        df = df.dropna(subset=[lon_col, lat_col])
        
        if len(df) == 0:
            return None, "❌ Aucun point valide trouvé dans le CSV"
        
        return df, None
    except Exception as e:
        return None, f"❌ Erreur lors du chargement du CSV: {e}"

def extract_shapefile_from_zip(zip_file):
    """Extrait un shapefile d'un fichier ZIP"""
    try:
        extract_folder = "uploads/shapefiles"
        if not os.path.exists(extract_folder):
            os.makedirs(extract_folder)
        
        zip_path = save_uploaded_file(zip_file, "uploads")
        if not zip_path:
            return None, "Erreur lors de la sauvegarde du ZIP"
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        
        # Trouver le fichier .shp
        for root, dirs, files in os.walk(extract_folder):
            for file in files:
                if file.endswith('.shp'):
                    return os.path.join(root, file), None
        
        return None, "❌ Aucun fichier .shp trouvé dans le ZIP"
    except Exception as e:
        return None, f"❌ Erreur lors de l'extraction: {e}"

# ============================================================================
# FONCTIONS UTILITAIRES - GESTION DES FICHIERS TIFF
# ============================================================================

def load_tiff_files(folder_path):
    """Charge les fichiers TIFF du dossier spécifié"""
    try:
        tiff_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path) 
            if f.lower().endswith(('.tif', '.tiff'))
        ]
    except Exception as e:
        st.error(f"❌ Erreur lors de la lecture du dossier {folder_path}: {e}")
        return []
    
    if not tiff_files:
        st.error("❌ Aucun fichier TIFF trouvé dans le dossier.")
        return []
    
    return [f for f in tiff_files if os.path.exists(f)]

def build_mosaic(tiff_files, mosaic_path="mosaic.tif"):
    """Construit une mosaïque à partir de plusieurs fichiers TIFF"""
    try:
        with st.spinner("🔄 Construction de la mosaïque..."):
            src_files = [rasterio.open(fp) for fp in tiff_files]
            mosaic, out_trans = rasterio.merge.merge(src_files)
            
            out_meta = src_files[0].meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "compress": "lzw"
            })
            
            with rasterio.open(mosaic_path, "w", **out_meta) as dest:
                dest.write(mosaic)
            
            for src in src_files:
                src.close()
            
            st.success(f"✅ Mosaïque créée: {len(tiff_files)} fichiers fusionnés")
            return mosaic_path
    except Exception as e:
        st.error(f"❌ Erreur lors de la création de la mosaïque: {e}")
        return None

def get_mosaic_info(mosaic_file):
    """Récupère les informations de la mosaïque"""
    try:
        with rasterio.open(mosaic_file) as src:
            return {
                "bounds": src.bounds,
                "crs": src.crs.to_string(),
                "width": src.width,
                "height": src.height,
                "resolution": src.res,
                "nodata": src.nodata,
                "dtype": src.dtypes[0],
                "count": src.count
            }
    except Exception as e:
        st.error(f"❌ Erreur lors de la lecture des infos: {e}")
        return None

# ============================================================================
# FONCTIONS UTILITAIRES - CARTOGRAPHIE
# ============================================================================

def add_shapefile_to_map(m, gdf, layer_name="Shapefile"):
    """Ajoute un shapefile à la carte Folium"""
    try:
        feature_group = folium.FeatureGroup(name=layer_name, show=True)
        
        for idx, row in gdf.iterrows():
            geom = row.geometry
            
            # Style selon le type de géométrie
            if geom.geom_type == 'Point':
                folium.CircleMarker(
                    location=[geom.y, geom.x],
                    radius=5,
                    color='red',
                    fill=True,
                    fillColor='red',
                    fillOpacity=0.6,
                    popup=f"Point {idx}"
                ).add_to(feature_group)
            
            elif geom.geom_type in ['LineString', 'MultiLineString']:
                if geom.geom_type == 'LineString':
                    coords = list(geom.coords)
                    folium.PolyLine(
                        locations=[(c[1], c[0]) for c in coords],
                        color='blue',
                        weight=2,
                        opacity=0.8,
                        popup=f"Ligne {idx}"
                    ).add_to(feature_group)
            
            elif geom.geom_type in ['Polygon', 'MultiPolygon']:
                if geom.geom_type == 'Polygon':
                    coords = list(geom.exterior.coords)
                    folium.Polygon(
                        locations=[(c[1], c[0]) for c in coords],
                        color='green',
                        fill=True,
                        fillColor='green',
                        fillOpacity=0.3,
                        weight=2,
                        popup=f"Polygone {idx}"
                    ).add_to(feature_group)
        
        feature_group.add_to(m)
        return True
    except Exception as e:
        st.error(f"❌ Erreur lors de l'ajout du shapefile: {e}")
        return False

def add_csv_points_to_map(m, df, lon_col, lat_col, layer_name="Points CSV"):
    """Ajoute des points CSV à la carte Folium"""
    try:
        feature_group = folium.FeatureGroup(name=layer_name, show=True)
        
        for idx, row in df.iterrows():
            popup_text = "<br>".join([f"{col}: {row[col]}" for col in df.columns[:5]])
            
            folium.CircleMarker(
                location=[row[lat_col], row[lon_col]],
                radius=6,
                color='orange',
                fill=True,
                fillColor='orange',
                fillOpacity=0.7,
                popup=folium.Popup(popup_text, max_width=200)
            ).add_to(feature_group)
        
        feature_group.add_to(m)
        return True
    except Exception as e:
        st.error(f"❌ Erreur lors de l'ajout des points CSV: {e}")
        return False

def create_advanced_map(mosaic_file, show_minimap=True, show_fullscreen=True):
    """Crée une carte Folium interactive avec fonctionnalités avancées"""
    
    try:
        with rasterio.open(mosaic_file) as src:
            bounds = src.bounds
            if src.crs.to_string() != "EPSG:4326":
                from rasterio.warp import transform
                left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top
                xs, ys = transform(src.crs, "EPSG:4326", [left, right], [bottom, top])
                center_lat = (min(ys) + max(ys)) / 2
                center_lon = (min(xs) + max(xs)) / 2
                bounds_latlon = [[min(ys), min(xs)], [max(ys), max(xs)]]
            else:
                center_lat = (bounds.bottom + bounds.top) / 2
                center_lon = (bounds.left + bounds.right) / 2
                bounds_latlon = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
    except Exception as e:
        st.error(f"❌ Erreur lors de l'ouverture de la mosaïque: {e}")
        center_lat, center_lon = 0, 0
        bounds_latlon = None
    
    # Création de la carte
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap',
        control_scale=True
    )
    
    # Ajout de l'emprise de la mosaïque
    if bounds_latlon:
        mosaic_group = folium.FeatureGroup(name="📍 Emprise de la mosaïque", show=True)
        folium.Rectangle(
            bounds=bounds_latlon,
            color='#1f77b4',
            fill=True,
            fillColor='#1f77b4',
            fillOpacity=0.1,
            weight=3,
            tooltip="Zone d'étude"
        ).add_to(mosaic_group)
        mosaic_group.add_to(m)
    
    # Ajout des shapefiles uploadés
    for shp_data in st.session_state.get("shapefile_layers", []):
        add_shapefile_to_map(m, shp_data['gdf'], shp_data['name'])
    
    # Ajout des points CSV
    if st.session_state.get("uploaded_csv_data") is not None:
        csv_info = st.session_state["uploaded_csv_data"]
        add_csv_points_to_map(m, csv_info['df'], csv_info['lon_col'], 
                             csv_info['lat_col'], csv_info['name'])
    
    # Outils de dessin avancés
    Draw(
        draw_options={
            'polyline': {
                'allowIntersection': True,
                'shapeOptions': {'color': '#e74c3c', 'weight': 3}
            },
            'polygon': {
                'allowIntersection': False,
                'shapeOptions': {'color': '#2ecc71', 'weight': 2}
            },
            'rectangle': {
                'shapeOptions': {'color': '#3498db', 'weight': 2}
            },
            'circle': {
                'shapeOptions': {'color': '#9b59b6', 'weight': 2}
            },
            'marker': True,
            'circlemarker': True
        },
        edit_options={'edit': True, 'remove': True}
    ).add_to(m)
    
    # Plugins supplémentaires
    if show_minimap:
        MiniMap(toggle_display=True).add_to(m)
    
    if show_fullscreen:
        Fullscreen(position='topright').add_to(m)
    
    # Outil de mesure
    MeasureControl(
        primary_length_unit='meters',
        secondary_length_unit='kilometers',
        primary_area_unit='sqmeters',
        secondary_area_unit='hectares'
    ).add_to(m)
    
    # Position de la souris
    MousePosition(
        position='bottomleft',
        separator=' | ',
        prefix='Coordonnées:',
        lat_formatter="function(num) {return L.Util.formatNum(num, 5) + ' °N';}",
        lng_formatter="function(num) {return L.Util.formatNum(num, 5) + ' °E';}"
    ).add_to(m)
    
    # Couches de fond supplémentaires
    folium.TileLayer(
        tiles='https://tiles.stadiamaps.com/tiles/stamen_terrain/{z}/{x}/{y}.png',
        attr='Map tiles by Stadia Maps, under CC BY 3.0. Data by OpenStreetMap, under ODbL',
        name='🏔️ Terrain',
        show=False
    ).add_to(m)
    folium.TileLayer('CartoDB positron', name='⚪ Clair', show=False).add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='⚫ Sombre', show=False).add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='🛰️ Satellite',
        show=False
    ).add_to(m)
    
    folium.LayerControl(position='topright').add_to(m)
    
    return m

# ============================================================================
# FONCTIONS D'ANALYSE - CONTOURS
# ============================================================================

def generate_advanced_contours(mosaic_file, drawing_geometry, show_basemap=True, 
                              levels=15, colormap='terrain', show_hillshade=False):
    """Génère des contours d'élévation avec options avancées"""
    try:
        with rasterio.open(mosaic_file) as src:
            geom = drawing_geometry
            if src.crs.to_string() != "EPSG:4326":
                from rasterio.warp import transform_geom
                geom = transform_geom("EPSG:4326", src.crs, drawing_geometry)
            
            out_image, out_transform = rasterio.mask.mask(src, [geom], crop=True)
            data = out_image[0]
            nodata = src.nodata

            nrows, ncols = data.shape
            x_coords = np.arange(ncols) * out_transform.a + out_transform.c + out_transform.a/2
            y_coords = np.arange(nrows) * out_transform.e + out_transform.f + out_transform.e/2
            X, Y = np.meshgrid(x_coords, y_coords)

            from rasterio.warp import transform
            center_x = out_transform.c + (ncols/2) * out_transform.a
            center_y = out_transform.f + (nrows/2) * out_transform.e
            if src.crs.to_string() != "EPSG:4326":
                lon, lat = transform(src.crs, "EPSG:4326", [center_x], [center_y])
                center_lon, center_lat = lon[0], lat[0]
            else:
                center_lon, center_lat = center_x, center_y
            
            utm_zone = int((center_lon + 180) / 6) + 1
            utm_crs = f"EPSG:{32600 + utm_zone if center_lat >= 0 else 32700 + utm_zone}"

            x_flat = X.flatten()
            y_flat = Y.flatten()
            X_utm_flat, Y_utm_flat = transform(src.crs, utm_crs, x_flat, y_flat)
            X_utm = np.array(X_utm_flat).reshape(X.shape)
            Y_utm = np.array(Y_utm_flat).reshape(Y.shape)

            from rasterio.warp import transform_geom
            geom_utm = transform_geom("EPSG:4326", utm_crs, drawing_geometry)
            envelope = shape(geom_utm)

            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)

            stats = get_elevation_stats(data, nodata)
            st.session_state["statistics"] = stats

            vmin = np.nanmin(data)
            vmax = np.nanmax(data)
            contour_levels = np.linspace(vmin, vmax, levels)

            fig, ax = plt.subplots(figsize=(12, 8))
            
            if show_hillshade:
                from matplotlib.colors import LightSource
                ls = LightSource(azdeg=315, altdeg=45)
                hillshade = ls.hillshade(data, vert_exag=0.1)
                ax.imshow(hillshade, extent=[X_utm.min(), X_utm.max(), Y_utm.min(), Y_utm.max()],
                         cmap='gray', alpha=0.3, zorder=1)
            
            cf = ax.contourf(X_utm, Y_utm, data, levels=contour_levels, cmap=colormap, 
                            alpha=0.7, zorder=2)
            
            cs = ax.contour(X_utm, Y_utm, data, levels=contour_levels, colors='black', 
                           linewidths=0.5, alpha=0.8, zorder=3)
            ax.clabel(cs, inline=True, fontsize=8, fmt='%1.1f m')
            
            cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
            cbar.set_label('Altitude (m)', rotation=270, labelpad=20, fontsize=10)
            
            ax.set_title(f"Carte des Contours d'Élévation (UTM Zone {utm_zone})", 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel("UTM Easting (m)", fontsize=11)
            ax.set_ylabel("UTM Northing (m)", fontsize=11)

            minx, miny, maxx, maxy = envelope.bounds
            dx = (maxx - minx) * 0.05
            dy = (maxy - miny) * 0.05
            ax.set_xlim(minx - dx, maxx + dx)
            ax.set_ylim(miny - dy, maxy + dy)

            if show_basemap:
                ctx.add_basemap(ax, crs=utm_crs, source=ctx.providers.OpenStreetMap.Mapnik, 
                               alpha=st.session_state.get("basemap_opacity", 0.5), zorder=0)

            added_profile = False
            added_polygon = False
            added_point = False
            profile_counter = 1

            if "raw_drawings" in st.session_state:
                for d in st.session_state["raw_drawings"]:
                    if isinstance(d, dict) and "geometry" in d:
                        if d.get("geometry") == drawing_geometry:
                            continue
                        try:
                            geom_other_utm = transform_geom("EPSG:4326", utm_crs, d["geometry"])
                            shapely_other = shape(geom_other_utm)
                            
                            if shapely_other.intersects(envelope):
                                clipped = shapely_other.intersection(envelope)
                                if clipped.is_empty:
                                    continue
                                
                                if clipped.geom_type in ["Polygon", "MultiPolygon"]:
                                    label = "Polygone" if not added_polygon else "_nolegend_"
                                    if not added_polygon:
                                        added_polygon = True
                                    if clipped.geom_type == "Polygon":
                                        x_other, y_other = clipped.exterior.xy
                                        ax.plot(x_other, y_other, color='green', linestyle='-', 
                                               linewidth=2.5, label=label, zorder=5)
                                    else:
                                        for part in clipped.geoms:
                                            x_other, y_other = part.exterior.xy
                                            ax.plot(x_other, y_other, color='green', linestyle='-', 
                                                   linewidth=2.5, label=label, zorder=5)
                                
                                elif clipped.geom_type in ["LineString", "MultiLineString"]:
                                    current_profile_label = f"Profil {profile_counter}"
                                    profile_counter += 1
                                    legend_label = "Profil" if not added_profile else "_nolegend_"
                                    if not added_profile:
                                        added_profile = True
                                    if clipped.geom_type == "LineString":
                                        x_other, y_other = clipped.xy
                                        ax.plot(x_other, y_other, color='red', linestyle='-', 
                                               linewidth=2.5, label=legend_label, zorder=5)
                                        if len(x_other) >= 2:
                                            dx_line = x_other[1] - x_other[0]
                                            dy_line = y_other[1] - y_other[0]
                                            angle = np.degrees(np.arctan2(dy_line, dx_line))
                                        else:
                                            angle = 0
                                        centroid = clipped.centroid
                                        ax.text(centroid.x, centroid.y, current_profile_label, 
                                               fontsize=10, color='white', ha='center', va='center',
                                               bbox=dict(boxstyle='round,pad=0.3', facecolor='red', 
                                                        alpha=0.8), rotation=angle, zorder=6)
                                    else:
                                        for part in clipped.geoms:
                                            x_other, y_other = part.xy
                                            ax.plot(x_other, y_other, color='red', linestyle='-', 
                                                   linewidth=2.5, label=legend_label, zorder=5)
                                
                                elif clipped.geom_type in ["Point", "MultiPoint"]:
                                    label = "Point" if not added_point else "_nolegend_"
                                    if not added_point:
                                        added_point = True
                                    if clipped.geom_type == "Point":
                                        ax.plot(clipped.x, clipped.y, 'o', color='orange', 
                                               markersize=10, markeredgecolor='black', 
                                               markeredgewidth=1.5, label=label, zorder=5)
                                    else:
                                        for part in clipped.geoms:
                                            ax.plot(part.x, part.y, 'o', color='orange', 
                                                   markersize=10, markeredgecolor='black', 
                                                   markeredgewidth=1.5, label=label, zorder=5)
                        except Exception as e:
                            st.warning(f"⚠️ Impossible de tracer un dessin: {e}")

            x_env, y_env = envelope.exterior.xy
            ax.plot(x_env, y_env, color='blue', linewidth=3, label="Zone d'analyse", zorder=6)

            legend = ax.legend(loc='lower right', framealpha=0.9, facecolor='white', 
                             fontsize=9, edgecolor='black')
            
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            plt.tight_layout()
            return fig
            
    except Exception as e:
        st.error(f"❌ Erreur lors de la génération des contours: {e}")
        return None

# ============================================================================
# FONCTIONS D'ANALYSE - PROFILS
# ============================================================================

def interpolate_line(coords, step=50):
    """Interpole des points le long d'une ligne"""
    if len(coords) < 2:
        return coords, [0]
    
    sampled_points = [coords[0]]
    cumulative_dist = [0]
    
    for i in range(len(coords)-1):
        start = coords[i]
        end = coords[i+1]
        seg_distance = haversine(start[0], start[1], end[0], end[1])
        num_steps = max(int(seg_distance // step), 1)
        
        for j in range(1, num_steps+1):
            fraction = j / num_steps
            lon = start[0] + fraction * (end[0]-start[0])
            lat = start[1] + fraction * (end[1]-start[1])
            dist = haversine(sampled_points[-1][0], sampled_points[-1][1], lon, lat)
            sampled_points.append([lon, lat])
            cumulative_dist.append(cumulative_dist[-1] + dist)
    
    return sampled_points, cumulative_dist

def generate_advanced_profile(mosaic_file, coords, profile_title, show_slope=True):
    """Génère un profil d'élévation avancé avec pente"""
    try:
        step = st.session_state.get("profile_resolution", 50)
        points, distances = interpolate_line(coords, step=step)
        elevations = []
        
        with rasterio.open(mosaic_file) as src:
            for p in points:
                pt = p
                if src.crs.to_string() != "EPSG:4326":
                    from rasterio.warp import transform
                    xs, ys = transform("EPSG:4326", src.crs, [p[0]], [p[1]])
                    pt = (xs[0], ys[0])
                elev = list(src.sample([pt]))[0][0]
                elevations.append(elev)
        
        slopes = []
        if show_slope and len(distances) > 1:
            for i in range(1, len(distances)):
                dh = elevations[i] - elevations[i-1]
                dd = distances[i] - distances[i-1]
                if dd > 0:
                    slope = (dh / dd) * 100
                    slopes.append(slope)
                else:
                    slopes.append(0)
            slopes.insert(0, 0)
        
        if show_slope and slopes:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), 
                                           gridspec_kw={'height_ratios': [2, 1]})
        else:
            fig, ax1 = plt.subplots(figsize=(12, 4))
        
        ax1.plot(distances, elevations, 'b-', linewidth=2, label='Élévation')
        ax1.fill_between(distances, elevations, alpha=0.3, color='lightblue')
        ax1.set_title(profile_title, fontsize=14, fontweight='bold')
        ax1.set_xlabel("Distance (m)", fontsize=11)
        ax1.set_ylabel("Altitude (m)", fontsize=11)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='upper right')
        
        stats_text = (f"Min: {min(elevations):.1f}m | Max: {max(elevations):.1f}m | "
                     f"Dénivelé: {max(elevations)-min(elevations):.1f}m | "
                     f"Distance: {distances[-1]:.0f}m")
        ax1.text(0.5, 0.98, stats_text, transform=ax1.transAxes, 
                fontsize=9, verticalalignment='top', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if show_slope and slopes:
            colors_slope = ['red' if s > 15 else 'orange' if s > 5 else 'green' 
                           for s in slopes]
            ax2.bar(distances, slopes, width=step, color=colors_slope, alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_xlabel("Distance (m)", fontsize=11)
            ax2.set_ylabel("Pente (%)", fontsize=11)
            ax2.set_title("Variation de la pente", fontsize=11)
            ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
            
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', alpha=0.7, label='Pente douce (0-5%)'),
                Patch(facecolor='orange', alpha=0.7, label='Pente moyenne (5-15%)'),
                Patch(facecolor='red', alpha=0.7, label='Pente forte (>15%)')
            ]
            ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"❌ Erreur lors de la génération du profil: {e}")
        return None

# ============================================================================
# FONCTIONS D'ANALYSE - STATISTIQUES
# ============================================================================

def analyze_zone(mosaic_file, geometry):
    """Analyse statistique d'une zone"""
    try:
        with rasterio.open(mosaic_file) as src:
            geom = geometry
            if src.crs.to_string() != "EPSG:4326":
                from rasterio.warp import transform_geom
                geom = transform_geom("EPSG:4326", src.crs, geometry)
            
            out_image, out_transform = rasterio.mask.mask(src, [geom], crop=True)
            data = out_image[0]
            nodata = src.nodata
            
            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)
            
            stats = get_elevation_stats(data, nodata)
            
            area = calculate_area(geometry["coordinates"][0])
            
            hist, bin_edges = np.histogram(data[~np.isnan(data)], bins=20)
            
            return {
                "statistics": stats,
                "area": area,
                "histogram": (hist, bin_edges),
                "pixel_count": np.sum(~np.isnan(data))
            }
    except Exception as e:
        st.error(f"❌ Erreur d'analyse: {e}")
        return None

def create_statistics_chart(analysis_data):
    """Crée un graphique des statistiques"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    stats = analysis_data["statistics"]
    labels = ['Min', 'Max', 'Moyenne', 'Médiane']
    values = [stats['min'], stats['max'], stats['mean'], stats['median']]
    colors_bar = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    ax1.bar(labels, values, color=colors_bar, alpha=0.7)
    ax1.set_title("Statistiques d'Élévation", fontweight='bold')
    ax1.set_ylabel("Altitude (m)")
    ax1.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(values):
        ax1.text(i, v + 0.1, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
    
    hist, bin_edges = analysis_data["histogram"]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax2.bar(bin_centers, hist, width=np.diff(bin_edges), alpha=0.7, color='#9b59b6')
    ax2.set_title("Distribution des Élévations", fontweight='bold')
    ax2.set_xlabel("Altitude (m)")
    ax2.set_ylabel("Nombre de pixels")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ============================================================================
# FONCTIONS D'EXPORT
# ============================================================================

def create_pdf_report(analysis_results, mosaic_info, filename="rapport_analyse.pdf"):
    """Crée un rapport PDF complet"""
    try:
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        
        # En-tête
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "Rapport d'Analyse Topographique")
        c.setFont("Helvetica", 10)
        c.drawString(50, height - 70, f"Généré le: {datetime.now().strftime('%d/%m/%Y à %H:%M')}")
        
        # Informations de la mosaïque
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, height - 100, "Informations de la Mosaïque:")
        c.setFont("Helvetica", 10)
        
        y_pos = height - 120
        info_lines = [
            f"Dimensions: {mosaic_info['width']} x {mosaic_info['height']} pixels",
            f"Résolution: {mosaic_info['resolution'][0]:.2f} x {mosaic_info['resolution'][1]:.2f} m/pixel",
            f"Système de coordonnées: {mosaic_info['crs']}",
            f"Type de données: {mosaic_info['dtype']}",
            f"Nombre de bandes: {mosaic_info['count']}"
        ]
        
        for line in info_lines:
            c.drawString(70, y_pos, line)
            y_pos -= 15
        
        # Statistiques globales
        if "statistics" in st.session_state and st.session_state["statistics"]:
            stats = st.session_state["statistics"]
            y_pos -= 20
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y_pos, "Statistiques Globales:")
            c.setFont("Helvetica", 10)
            
            y_pos -= 15
            stat_lines = [
                f"Altitude minimale: {stats['min']:.2f} m",
                f"Altitude maximale: {stats['max']:.2f} m",
                f"Altitude moyenne: {stats['mean']:.2f} m",
                f"Écart-type: {stats['std']:.2f} m",
                f"Plage: {stats['range']:.2f} m"
            ]
            
            for line in stat_lines:
                c.drawString(70, y_pos, line)
                y_pos -= 15
        
        # Résultats d'analyse
        y_pos -= 20
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_pos, "Résultats d'Analyse:")
        
        for i, result in enumerate(analysis_results):
            if y_pos < 100:
                c.showPage()
                y_pos = height - 50
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, y_pos, "Résultats d'Analyse (suite):")
                y_pos -= 20
            
            y_pos -= 15
            c.setFont("Helvetica", 10)
            c.drawString(70, y_pos, f"Analyse {i+1}: {result['type']} - {result['description']}")
            y_pos -= 10
        
        c.save()
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        st.error(f"❌ Erreur lors de la création du PDF: {e}")
        return None

def export_data(analysis_results, mosaic_info, format_type):
    """Exporte les données dans différents formats"""
    try:
        if format_type == "PDF":
            return create_pdf_report(analysis_results, mosaic_info)
        elif format_type == "JSON":
            export_data = {
                "metadata": {
                    "export_date": datetime.now().isoformat(),
                    "mosaic_info": mosaic_info,
                    "analysis_count": len(analysis_results)
                },
                "statistics": st.session_state.get("statistics", {}),
                "analyses": analysis_results
            }
            buffer = BytesIO()
            buffer.write(json.dumps(export_data, indent=2, default=str).encode())
            buffer.seek(0)
            return buffer
        elif format_type == "CSV":
            buffer = BytesIO()
            lines = ["Type,Description,Surface(m²),Altitude Min,Altitude Max,Altitude Moyenne"]
            for result in analysis_results:
                line = f"{result['type']},{result['description']},{result.get('area', 0):.2f},"
                line += f"{result.get('stats', {}).get('min', 0):.2f},"
                line += f"{result.get('stats', {}).get('max', 0):.2f},"
                line += f"{result.get('stats', {}).get('mean', 0):.2f}"
                lines.append(line)
            buffer.write("\n".join(lines).encode())
            buffer.seek(0)
            return buffer
    except Exception as e:
        st.error(f"❌ Erreur d'export: {e}")
        return None

# ============================================================================
# INTERFACE UTILISATEUR - SIDEBAR
# ============================================================================

def render_sidebar():
    """Affiche la barre latérale avec les contrôles"""
    st.sidebar.title("🗺️ CartoTools Pro")
    st.sidebar.markdown("---")
    
    # Sélection de la mosaïque
    st.sidebar.subheader("📁 Source des données")
    
    mosaic_choice = st.sidebar.radio(
        "Choisir la source:",
        ["Mosaïque par défaut", "TIFF uploadé"],
        key="mosaic_choice"
    )
    
    if mosaic_choice == "TIFF uploadé":
        st.session_state.use_uploaded_tiff = True
        if st.session_state.uploaded_tiff_path:
            st.sidebar.success(f"✅ TIFF chargé: {os.path.basename(st.session_state.uploaded_tiff_path)}")
        else:
            st.sidebar.warning("⚠️ Aucun TIFF uploadé")
    else:
        st.session_state.use_uploaded_tiff = False
    
    # Upload de fichiers
    st.sidebar.subheader("📤 Upload de données")
    
    # Upload TIFF
    uploaded_tiff = st.sidebar.file_uploader(
        "📁 Uploader un TIFF personnalisé",
        type=['tif', 'tiff'],
        help="Format requis: GeoTIFF avec CRS défini. Bande unique (élévation) recommandée."
    )
    
    if uploaded_tiff is not None:
        if st.session_state.get("uploaded_tiff_path") is None or st.sidebar.button("🔄 Recharger TIFF"):
            with st.spinner("📥 Sauvegarde du TIFF..."):
                file_path = save_uploaded_file(uploaded_tiff, "uploads/tiff")
                if file_path:
                    is_valid, message = validate_tiff_file(file_path)
                    if is_valid:
                        st.session_state.uploaded_tiff_path = file_path
                        st.sidebar.success("✅ TIFF validé et chargé")
                    else:
                        st.sidebar.error(f"❌ {message}")
    
    # Upload Shapefile
    uploaded_shp = st.sidebar.file_uploader(
        "🗺️ Uploader un Shapefile (ZIP)",
        type=['zip'],
        help="Archive ZIP contenant .shp, .shx, .dbf, .prj"
    )
    
    if uploaded_shp is not None and st.sidebar.button("📥 Charger Shapefile"):
        with st.spinner("📥 Extraction du shapefile..."):
            shp_path, error = extract_shapefile_from_zip(uploaded_shp)
            if shp_path:
                gdf, error = load_shapefile(shp_path)
                if gdf is not None:
                    layer_name = f"Shapefile_{len(st.session_state.shapefile_layers) + 1}"
                    st.session_state.shapefile_layers.append({
                        'name': layer_name,
                        'gdf': gdf,
                        'path': shp_path
                    })
                    st.sidebar.success(f"✅ Shapefile chargé: {len(gdf)} entités")
                else:
                    st.sidebar.error(error)
            else:
                st.sidebar.error(error)
    
    # Upload CSV
    uploaded_csv = st.sidebar.file_uploader(
        "📊 Uploader des points CSV",
        type=['csv', 'txt'],
        help="Colonnes requises: longitude, latitude"
    )
    
    if uploaded_csv is not None:
        if st.sidebar.button("📥 Charger CSV"):
            with st.spinner("📥 Traitement du CSV..."):
                csv_path = save_uploaded_file(uploaded_csv, "uploads/csv")
                if csv_path:
                    df, error = load_csv_points(csv_path)
                    if df is not None:
                        st.session_state.uploaded_csv_data = {
                            'df': df,
                            'path': csv_path,
                            'name': f"Points_{len(df)}",
                            'lon_col': 'longitude',
                            'lat_col': 'latitude'
                        }
                        st.sidebar.success(f"✅ CSV chargé: {len(df)} points")
                    else:
                        st.sidebar.error(error)
    
    # Gestion des couches
    if st.session_state.shapefile_layers or st.session_state.uploaded_csv_data:
        st.sidebar.subheader("🗂️ Couches chargées")
        
        for layer in st.session_state.shapefile_layers:
            if st.sidebar.button(f"❌ Supprimer {layer['name']}", key=f"del_{layer['name']}"):
                st.session_state.shapefile_layers.remove(layer)
                st.rerun()
        
        if st.session_state.uploaded_csv_data:
            if st.sidebar.button("❌ Supprimer points CSV"):
                st.session_state.uploaded_csv_data = None
                st.rerun()
    
    # Paramètres d'analyse
    st.sidebar.subheader("⚙️ Paramètres d'analyse")
    
    st.session_state.color_scheme = st.sidebar.selectbox(
        "Palette de couleurs:",
        ["terrain", "viridis", "plasma", "inferno", "magma", "cividis"],
        index=0
    )
    
    st.session_state.contour_levels = st.sidebar.slider(
        "Nombre de niveaux de contour:",
        min_value=5, max_value=50, value=15, step=1
    )
    
    st.session_state.profile_resolution = st.sidebar.slider(
        "Résolution du profil (m):",
        min_value=10, max_value=200, value=50, step=10
    )
    
    st.session_state.basemap_opacity = st.sidebar.slider(
        "Opacité de la carte de fond:",
        min_value=0.0, max_value=1.0, value=0.5, step=0.1
    )
    
    st.session_state.show_hillshade = st.sidebar.checkbox("Afficher l'ombrage", value=False)
    st.session_state.show_slope = st.sidebar.checkbox("Afficher la pente", value=True)
    
    # Export
    st.sidebar.subheader("📤 Export")
    st.session_state.export_format = st.sidebar.selectbox(
        "Format d'export:",
        ["PNG", "PDF", "JSON", "CSV"],
        index=0
    )
    
    # Informations
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **📝 Instructions:**
    1. Chargez vos données (TIFF, Shapefile, CSV)
    2. Dessinez sur la carte pour analyser
    3. Exportez vos résultats
    """)

# ============================================================================
# INTERFACE UTILISATEUR - CONTENU PRINCIPAL
# ============================================================================

def render_main_content():
    """Affiche le contenu principal de l'application"""
    st.title("🗺️ CartoTools Pro - Analyse Topographique Avancée")
    
    # Vérification de la disponibilité des données
    mosaic_file = None
    if st.session_state.use_uploaded_tiff and st.session_state.uploaded_tiff_path:
        mosaic_file = st.session_state.uploaded_tiff_path
        mosaic_source = "TIFF uploadé"
    else:
        mosaic_file = "mosaic.tif"
        mosaic_source = "Mosaïque par défaut"
        if not os.path.exists(mosaic_file):
            st.error("❌ La mosaïque par défaut n'est pas disponible. Veuillez uploader un TIFF.")
            return
    
    # Affichage des informations de la mosaïque
    mosaic_info = get_mosaic_info(mosaic_file)
    if mosaic_info:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Source", mosaic_source)
        with col2:
            st.metric("Dimensions", f"{mosaic_info['width']}×{mosaic_info['height']}")
        with col3:
            st.metric("Résolution", f"{mosaic_info['resolution'][0]:.1f}m")
    
    # Carte interactive
    st.subheader("🗺️ Carte Interactive")
    
    with st.spinner("🔄 Chargement de la carte..."):
        m = create_advanced_map(mosaic_file)
        map_data = st_folium(m, width=1200, height=500)
    
    # Gestion des dessins
    if map_data and "last_active_drawing" in map_data and map_data["last_active_drawing"]:
        drawing = map_data["last_active_drawing"]
        geometry = drawing.get("geometry")
        
        if geometry and geometry not in [d.get("geometry") for d in st.session_state.raw_drawings]:
            st.session_state.raw_drawings.append({
                "geometry": geometry,
                "type": drawing.get("geometry", {}).get("type"),
                "properties": drawing.get("properties", {})
            })
    
    # Sélection du mode d'analyse
    st.subheader("🔍 Analyse des Données")
    
    if st.session_state.raw_drawings:
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            selected_drawing_idx = st.selectbox(
                "Sélectionner un dessin à analyser:",
                range(len(st.session_state.raw_drawings)),
                format_func=lambda i: f"Dessin {i+1} - {st.session_state.raw_drawings[i]['type']}"
            )
        
        with col2:
            analysis_type = st.selectbox(
                "Type d'analyse:",
                ["Contours", "Profil", "Statistiques"]
            )
        
        with col3:
            if st.button("🚀 Lancer l'analyse", type="primary"):
                selected_drawing = st.session_state.raw_drawings[selected_drawing_idx]
                
                with st.spinner("🔄 Analyse en cours..."):
                    if analysis_type == "Contours":
                        fig = generate_advanced_contours(
                            mosaic_file,
                            selected_drawing["geometry"],
                            levels=st.session_state.contour_levels,
                            colormap=st.session_state.color_scheme,
                            show_hillshade=st.session_state.show_hillshade
                        )
                        if fig:
                            st.pyplot(fig)
                            result = {
                                "type": "Contours",
                                "description": f"Cartographie des contours - {st.session_state.contour_levels} niveaux",
                                "timestamp": datetime.now()
                            }
                            st.session_state.analysis_results.append(result)
                    
                    elif analysis_type == "Profil":
                        if selected_drawing["type"] in ["LineString", "MultiLineString"]:
                            coords = selected_drawing["geometry"]["coordinates"]
                            if selected_drawing["type"] == "MultiLineString":
                                coords = coords[0]
                            fig = generate_advanced_profile(
                                mosaic_file,
                                coords,
                                f"Profil topographique - Dessin {selected_drawing_idx + 1}",
                                show_slope=st.session_state.show_slope
                            )
                            if fig:
                                st.pyplot(fig)
                                result = {
                                    "type": "Profil",
                                    "description": f"Profil topographique - Longueur: {len(coords)} points",
                                    "timestamp": datetime.now()
                                }
                                st.session_state.analysis_results.append(result)
                        else:
                            st.error("❌ Le profil nécessite une ligne")
                    
                    elif analysis_type == "Statistiques":
                        analysis_data = analyze_zone(mosaic_file, selected_drawing["geometry"])
                        if analysis_data:
                            col_stat1, col_stat2 = st.columns(2)
                            
                            with col_stat1:
                                st.subheader("📊 Statistiques détaillées")
                                stats = analysis_data["statistics"]
                                st.metric("Altitude min", f"{stats['min']:.2f} m")
                                st.metric("Altitude max", f"{stats['max']:.2f} m")
                                st.metric("Altitude moyenne", f"{stats['mean']:.2f} m")
                                st.metric("Écart-type", f"{stats['std']:.2f} m")
                                st.metric("Surface", f"{analysis_data['area']:.2f} m²")
                            
                            with col_stat2:
                                fig_stats = create_statistics_chart(analysis_data)
                                st.pyplot(fig_stats)
                            
                            result = {
                                "type": "Statistiques",
                                "description": f"Analyse statistique - Surface: {analysis_data['area']:.0f} m²",
                                "stats": stats,
                                "area": analysis_data["area"],
                                "timestamp": datetime.now()
                            }
                            st.session_state.analysis_results.append(result)
        
        # Gestion des dessins existants
        st.subheader("📝 Dessins existants")
        for i, drawing in enumerate(st.session_state.raw_drawings):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**Dessin {i+1}** - {drawing['type']}")
            with col2:
                if st.button(f"🔍 Analyser", key=f"analyze_{i}"):
                    st.session_state.raw_drawings[selected_drawing_idx] = drawing
            with col3:
                if st.button(f"🗑️ Supprimer", key=f"delete_{i}"):
                    st.session_state.raw_drawings.pop(i)
                    st.rerun()
    
    else:
        st.info("🎨 Dessinez sur la carte pour commencer l'analyse")
    
    # Export des résultats
    if st.session_state.analysis_results:
        st.subheader("📤 Export des résultats")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("💾 Exporter les résultats", type="primary"):
                with st.spinner("🔄 Préparation de l'export..."):
                    export_buffer = export_data(
                        st.session_state.analysis_results,
                        mosaic_info,
                        st.session_state.export_format
                    )
                    
                    if export_buffer:
                        filename = f"analyse_topographique_{datetime.now().strftime('%Y%m%d_%H%M')}.{st.session_state.export_format.lower()}"
                        
                        st.download_button(
                            label=f"📥 Télécharger {st.session_state.export_format}",
                            data=export_buffer,
                            file_name=filename,
                            mime={
                                "PNG": "image/png",
                                "PDF": "application/pdf",
                                "JSON": "application/json",
                                "CSV": "text/csv"
                            }[st.session_state.export_format]
                        )
        
        with col2:
            st.write(f"**{len(st.session_state.analysis_results)}** analyses effectuées")

# ============================================================================
# APPLICATION PRINCIPALE
# ============================================================================

def main():
    """Fonction principale de l'application"""
    
    # Initialisation
    initialize_session_state()
    
    # Interface utilisateur
    render_sidebar()
    render_main_content()
    
    # Nettoyage temporaire
    if st.sidebar.button("🧹 Nettoyer les fichiers temporaires"):
        try:
            for folder in ["uploads", "temp"]:
                if os.path.exists(folder):
                    shutil.rmtree(folder)
                    os.makedirs(folder)
            st.sidebar.success("✅ Fichiers temporaires nettoyés")
        except Exception as e:
            st.sidebar.error(f"❌ Erreur lors du nettoyage: {e}")

if __name__ == "__main__":
    main()

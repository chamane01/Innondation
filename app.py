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
:
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
    
    ax1.bar(labels, values, color=colors_bar, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Altitude (m)', fontsize=11)
    ax1.set_title('Statistiques d\'Élévation', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(values):
        ax1.text(i, v + stats['range']*0.02, f'{v:.1f}m', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    hist, bin_edges = analysis_data["histogram"]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    ax2.bar(bin_centers, hist, width=np.diff(bin_edges), 
           color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Altitude (m)', fontsize=11)
    ax2.set_ylabel('Fréquence', fontsize=11)
    ax2.set_title('Distribution des Altitudes', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

# ============================================================================
# FONCTIONS DE STOCKAGE
# ============================================================================

def store_figure(fig, result_type, title, metadata=None):
    """Sauvegarde une figure dans la session"""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    
    if "analysis_results" not in st.session_state:
        st.session_state["analysis_results"] = []
    
    result = {
        "type": result_type,
        "title": title,
        "image": buf.getvalue(),
        "timestamp": datetime.now(),
        "metadata": metadata or {}
    }
    
    st.session_state["analysis_results"].append(result)
    return len(st.session_state["analysis_results"]) - 1

def export_results(format_type="PNG"):
    """Exporte tous les résultats"""
    if not st.session_state.get("analysis_results"):
        st.warning("⚠️ Aucun résultat à exporter")
        return None
    
    if format_type == "PNG":
        from zipfile import ZipFile
        zip_buffer = BytesIO()
        
        with ZipFile(zip_buffer, 'w') as zip_file:
            for i, result in enumerate(st.session_state["analysis_results"]):
                filename = f"{i+1:02d}_{result['title']}.png"
                zip_file.writestr(filename, result['image'])
        
        zip_buffer.seek(0)
        return zip_buffer
    
    elif format_type == "PDF":
        pdf_buffer = generate_analysis_report()
        return pdf_buffer

# ============================================================================
# INTERFACE - GESTION DES FICHIERS UPLOADÉS
# ============================================================================

def run_file_upload_interface():
    """Interface pour uploader et gérer les fichiers"""
    st.markdown("### 📂 Gestion des Fichiers")
    
    tab1, tab2, tab3 = st.tabs(["🗺️ GeoTIFF", "📍 Shapefile", "📊 CSV/TXT"])
    
    # TAB 1: GeoTIFF
    with tab1:
        st.markdown("""
        **📋 Format requis:**
        - **Extension:** `.tif` ou `.tiff`
        - **Système de coordonnées (CRS)** défini
        - **Au moins une bande** de données d'élévation
        - **Résolution recommandée:** < 10m pour de meilleurs résultats
        
        💡 Le fichier sera utilisé à la place de l'orthomosaïque par défaut.
        """)
        
        uploaded_tiff = st.file_uploader(
            "📤 Téléverser un GeoTIFF personnalisé",
            type=["tif", "tiff"],
            key="upload_tiff",
            help="Uploadez votre propre fichier d'élévation"
        )
        
        if uploaded_tiff:
            with st.spinner("Validation du fichier..."):
                tiff_path = save_uploaded_file(uploaded_tiff, "uploads/tiff")
                if tiff_path:
                    is_valid, message = validate_tiff_file(tiff_path)
                    
                    if is_valid:
                        st.success(f"✅ {message}")
                        st.session_state["uploaded_tiff_path"] = tiff_path
                        
                        # Afficher les infos du fichier
                        info = get_mosaic_info(tiff_path)
                        if info:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Largeur", f"{info['width']} px")
                            with col2:
                                st.metric("Hauteur", f"{info['height']} px")
                            with col3:
                                st.metric("CRS", info['crs'])
                            
                            st.info("✅ Fichier prêt à être utilisé. Activez-le dans les paramètres ci-dessous.")
                    else:
                        st.error(f"❌ {message}")
    
    # TAB 2: Shapefile
    with tab2:
        st.markdown("""
        **📋 Format requis:**
        - **Fichier ZIP** contenant tous les composants du shapefile:
          - `.shp` (géométries) - **obligatoire**
          - `.shx` (index) - **obligatoire**
          - `.dbf` (attributs) - **obligatoire**
          - `.prj` (projection) - recommandé
        - **Système de coordonnées** défini (sinon, WGS84 sera assumé)
        
        💡 Les shapefiles seront affichés comme couches sur la carte.
        """)
        
        uploaded_shp = st.file_uploader(
            "📤 Téléverser un Shapefile (ZIP)",
            type=["zip"],
            key="upload_shapefile",
            help="Uploadez un fichier ZIP contenant le shapefile complet"
        )
        
        if uploaded_shp:
            with st.spinner("Extraction et validation..."):
                shp_path, error = extract_shapefile_from_zip(uploaded_shp)
                
                if error:
                    st.error(error)
                else:
                    gdf, error = load_shapefile(shp_path)
                    
                    if error:
                        st.error(error)
                    else:
                        st.success(f"✅ Shapefile chargé: {len(gdf)} entité(s)")
                        
                        # Informations sur le shapefile
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Entités", len(gdf))
                        with col2:
                            geom_types = gdf.geometry.geom_type.unique()
                            st.metric("Type", ", ".join(geom_types))
                        with col3:
                            st.metric("CRS", str(gdf.crs) if gdf.crs else "Non défini")
                        
                        # Nom de la couche
                        layer_name = st.text_input(
                            "Nom de la couche",
                            value=uploaded_shp.name.replace('.zip', ''),
                            key="shapefile_layer_name"
                        )
                        
                        if st.button("✅ Ajouter à la carte", key="add_shapefile_layer"):
                            if "shapefile_layers" not in st.session_state:
                                st.session_state["shapefile_layers"] = []
                            
                            st.session_state["shapefile_layers"].append({
                                'gdf': gdf,
                                'name': layer_name,
                                'path': shp_path
                            })
                            st.success(f"✅ Couche '{layer_name}' ajoutée!")
                            st.rerun()
        
        # Afficher les couches existantes
        if st.session_state.get("shapefile_layers"):
            st.markdown("---")
            st.markdown("#### 📚 Couches chargées")
            for idx, layer in enumerate(st.session_state["shapefile_layers"]):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{layer['name']}** - {len(layer['gdf'])} entité(s)")
                with col2:
                    if st.button("🗑️", key=f"remove_shp_{idx}"):
                        st.session_state["shapefile_layers"].pop(idx)
                        st.rerun()
    
    # TAB 3: CSV/TXT
    with tab3:
        st.markdown("""
        **📋 Format requis:**
        - **Extension:** `.csv` ou `.txt`
        - **Séparateur:** virgule (`,`) ou point-virgule (`;`)
        - **Colonnes obligatoires:** longitude et latitude
        - **Format des coordonnées:** décimal (ex: -73.5673, 45.5017)
        
        **📝 Exemple de structure:**
        ```
        nom,longitude,latitude,altitude
        Point A,-73.5673,45.5017,125
        Point B,-73.5680,45.5020,130
        ```
        
        💡 Les points seront affichés sur la carte avec leurs attributs.
        """)
        
        uploaded_csv = st.file_uploader(
            "📤 Téléverser un fichier CSV/TXT",
            type=["csv", "txt"],
            key="upload_csv",
            help="Uploadez un fichier contenant des coordonnées de points"
        )
        
        if uploaded_csv:
            csv_path = save_uploaded_file(uploaded_csv, "uploads/csv")
            
            if csv_path:
                # Prévisualisation
                try:
                    preview_df = pd.read_csv(csv_path, nrows=5)
                    st.markdown("**📊 Prévisualisation (5 premières lignes):**")
                    st.dataframe(preview_df)
                    
                    # Sélection des colonnes
                    columns = preview_df.columns.tolist()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        lon_col = st.selectbox(
                            "Colonne Longitude",
                            columns,
                            index=columns.index('longitude') if 'longitude' in columns else 0,
                            key="csv_lon_col"
                        )
                    with col2:
                        lat_col = st.selectbox(
                            "Colonne Latitude",
                            columns,
                            index=columns.index('latitude') if 'latitude' in columns else 0,
                            key="csv_lat_col"
                        )
                    
                    layer_name = st.text_input(
                        "Nom de la couche",
                        value=uploaded_csv.name.replace('.csv', '').replace('.txt', ''),
                        key="csv_layer_name"
                    )
                    
                    if st.button("✅ Charger les points", key="load_csv_points"):
                        df, error = load_csv_points(csv_path, lon_col, lat_col)
                        
                        if error:
                            st.error(error)
                        else:
                            st.success(f"✅ {len(df)} point(s) chargé(s)")
                            st.session_state["uploaded_csv_data"] = {
                                'df': df,
                                'lon_col': lon_col,
                                'lat_col': lat_col,
                                'name': layer_name
                            }
                            st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Erreur lors de la lecture du CSV: {e}")
        
        # Afficher les données CSV chargées
        if st.session_state.get("uploaded_csv_data"):
            st.markdown("---")
            st.markdown("#### 📍 Points chargés")
            csv_info = st.session_state["uploaded_csv_data"]
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{csv_info['name']}** - {len(csv_info['df'])} point(s)")
            with col2:
                if st.button("🗑️ Supprimer", key="remove_csv"):
                    st.session_state["uploaded_csv_data"] = None
                    st.rerun()

# ============================================================================
# INTERFACE - ANALYSE SPATIALE
# ============================================================================

def run_analysis_spatiale():
    st.markdown("<h1 class='main-header'>🔍 Analyse Spatiale Avancée</h1>", 
                unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        with st.expander("🗺️ Source de Données", expanded=True):
            # Choix entre mosaïque par défaut et TIFF uploadé
            if st.session_state.get("uploaded_tiff_path"):
                use_uploaded = st.checkbox(
                    "Utiliser le TIFF uploadé",
                    value=st.session_state.get("use_uploaded_tiff", False),
                    key="use_uploaded_tiff_checkbox"
                )
                st.session_state["use_uploaded_tiff"] = use_uploaded
                
                if use_uploaded:
                    st.success("✅ Utilisation du TIFF uploadé")
                else:
                    st.info("ℹ️ Utilisation de la mosaïque par défaut")
            else:
                st.info("ℹ️ Mosaïque par défaut")
                st.caption("Uploadez un TIFF dans l'onglet Gestion des Fichiers")
        
        with st.expander("📈 Paramètres des Contours", expanded=True):
            st.session_state["contour_levels"] = st.slider(
                "Nombre de niveaux", 5, 30, 15
            )
            st.session_state["color_scheme"] = st.selectbox(
                "Palette de couleurs", 
                ["terrain", "viridis", "plasma", "inferno", "coolwarm", "rainbow"]
            )
            st.session_state["basemap_opacity"] = st.slider(
                "Opacité fond de carte", 0.0, 1.0, 0.5, 0.1
            )
            st.session_state["show_hillshade"] = st.checkbox("Afficher l'ombrage du relief", False)
        
        with st.expander("📊 Paramètres des Profils"):
            st.session_state["profile_resolution"] = st.slider(
                "Résolution (m)", 10, 200, 50, 10
            )
            st.session_state["show_slope"] = st.checkbox("Afficher les pentes", True)
        
        with st.expander("🗺️ Options de la Carte"):
            show_minimap = st.checkbox("Mini-carte", True)
            show_fullscreen = st.checkbox("Plein écran", True)
    
    # Bouton pour gérer les fichiers
    with st.expander("📂 Gestion des Fichiers", expanded=False):
        run_file_upload_interface()
    
    # Déterminer quel fichier TIFF utiliser
    if st.session_state.get("use_uploaded_tiff") and st.session_state.get("uploaded_tiff_path"):
        mosaic_path = st.session_state["uploaded_tiff_path"]
        st.info("🗺️ Utilisation du TIFF uploadé pour l'analyse")
    else:
        folder_path = "TIFF"
        if not os.path.exists(folder_path):
            st.error("❌ Dossier TIFF introuvable")
            st.info("💡 Créez un dossier nommé 'TIFF' et placez-y vos fichiers GeoTIFF, ou uploadez un fichier personnalisé")
            return
        
        tiff_files = load_tiff_files(folder_path)
        if not tiff_files:
            st.warning("⚠️ Aucun fichier TIFF dans le dossier par défaut")
            st.info("💡 Uploadez un fichier TIFF personnalisé dans la section 'Gestion des Fichiers'")
            return
        
        mosaic_path = build_mosaic(tiff_files)
        if not mosaic_path:
            return
    
    with st.expander("ℹ️ Informations du Fichier", expanded=False):
        info = get_mosaic_info(mosaic_path)
        if info:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Largeur", f"{info['width']} px")
                st.metric("Hauteur", f"{info['height']} px")
            with col2:
                st.metric("Résolution X", f"{info['resolution'][0]:.2f}")
                st.metric("Résolution Y", f"{info['resolution'][1]:.2f}")
            with col3:
                st.metric("Système", info['crs'])
                st.metric("Type", info['dtype'])
    
    map_name = st.text_input("📝 Nom de votre projet", value="Analyse Topographique", 
                            key="analysis_map_name")
    
    st.markdown("### 🗺️ Carte Interactive")
    st.info("🖊️ Utilisez les outils de dessin pour définir vos zones d'analyse")
    
    m = create_advanced_map(mosaic_path, show_minimap, show_fullscreen)
    map_data = st_folium(m, width=None, height=600, key="analysis_map")
    
    if map_data and isinstance(map_data, dict) and "all_drawings" in map_data:
        st.session_state["raw_drawings"] = map_data["all_drawings"]
    
    st.markdown("---")
    st.markdown("### 🎯 Mode d'Analyse")
    
    if st.session_state["analysis_mode"] == "none":
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📈 Générer des Contours", use_container_width=True):
                st.session_state["analysis_mode"] = "contours"
                st.rerun()
        
        with col2:
            if st.button("📊 Tracer des Profils", use_container_width=True):
                st.session_state["analysis_mode"] = "profiles"
                st.rerun()
        
        with col3:
            if st.button("📐 Analyser une Zone", use_container_width=True):
                st.session_state["analysis_mode"] = "zone_analysis"
                st.rerun()
        
        with col4:
            if st.button("📏 Mesures & Stats", use_container_width=True):
                st.session_state["analysis_mode"] = "measurements"
                st.rerun()
    
    elif st.session_state["analysis_mode"] == "contours":
        st.markdown("<div class='sub-header'>📈 Génération de Contours</div>", 
                   unsafe_allow_html=True)
        
        show_basemap = st.checkbox("Afficher le fond de carte", value=True)
        
        drawing_geometries = []
        raw_drawings = st.session_state.get("raw_drawings") or []
        
        for drawing in raw_drawings:
            if isinstance(drawing, dict) and drawing.get("geometry", {}).get("type") == "Polygon":
                drawing_geometries.append(drawing.get("geometry"))
        
        if not drawing_geometries:
            st.warning("⚠️ Dessinez au moins un rectangle/polygone sur la carte")
        else:
            options_list = [f"Zone {i+1}" for i in range(len(drawing_geometries))]
            selected_indices = st.multiselect(
                "Sélectionnez les zones à analyser", 
                options=options_list,
                default=[options_list[0]] if options_list else []
            )
            
            col_btn1, col_btn2 = st.columns([3, 1])
            with col_btn1:
                generate_btn = st.button("🚀 Générer les Contours", 
                                        type="primary", use_container_width=True)
            with col_btn2:
                if st.button("↩️ Retour", use_container_width=True):
                    st.session_state["analysis_mode"] = "none"
                    st.rerun()
            
            if generate_btn and selected_indices:
                progress_bar = st.progress(0)
                for idx, sel in enumerate(selected_indices):
                    zone_idx = int(sel.split()[1]) - 1
                    geometry = drawing_geometries[zone_idx]
                    
                    st.markdown(f"#### Zone {zone_idx + 1}")
                    with st.spinner(f"Génération en cours..."):
                        fig = generate_advanced_contours(
                            mosaic_path, geometry, show_basemap,
                            st.session_state["contour_levels"],
                            st.session_state["color_scheme"],
                            st.session_state["show_hillshade"]
                        )
                        
                        if fig:
                            st.pyplot(fig)
                            store_figure(fig, "contour", 
                                       f"{map_name} - Contours Zone {zone_idx+1}")
                            plt.close(fig)
                            st.success(f"✅ Contours générés pour la Zone {zone_idx+1}")
                            
                            if st.session_state.get("statistics"):
                                stats = st.session_state["statistics"]
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("🔻 Min", f"{stats['min']:.1f} m")
                                col2.metric("🔺 Max", f"{stats['max']:.1f} m")
                                col3.metric("📊 Moyenne", f"{stats['mean']:.1f} m")
                                col4.metric("📏 Dénivelé", f"{stats['range']:.1f} m")
                    
                    progress_bar.progress((idx + 1) / len(selected_indices))
                
                progress_bar.empty()
    
    elif st.session_state["analysis_mode"] == "profiles":
        st.markdown("<div class='sub-header'>📊 Profils d'Élévation</div>", 
                   unsafe_allow_html=True)
        
        raw_drawings = st.session_state.get("raw_drawings") or []
        current_drawings = [
            d for d in raw_drawings 
            if isinstance(d, dict) and d.get("geometry", {}).get("type") == "LineString"
        ]
        
        if not current_drawings:
            st.info("ℹ️ Dessinez des lignes sur la carte pour créer des profils")
        else:
            col_btn1, col_btn2 = st.columns([3, 1])
            with col_btn1:
                generate_all = st.button("🚀 Générer Tous les Profils", 
                                        type="primary", use_container_width=True)
            with col_btn2:
                if st.button("↩️ Retour", use_container_width=True):
                    st.session_state["analysis_mode"] = "none"
                    st.rerun()
            
            if generate_all:
                progress_bar = st.progress(0)
                for i, drawing in enumerate(current_drawings):
                    profile_title = f"{map_name} - Profil {i+1}"
                    st.markdown(f"#### Profil {i+1}")
                    
                    with st.spinner(f"Génération du profil {i+1}..."):
                        fig = generate_advanced_profile(
                            mosaic_path, 
                            drawing["geometry"]["coordinates"],
                            profile_title,
                            st.session_state["show_slope"]
                        )
                        
                        if fig:
                            st.pyplot(fig)
                            store_figure(fig, "profile", profile_title)
                            plt.close(fig)
                            st.success(f"✅ Profil {i+1} généré")
                    
                    progress_bar.progress((i + 1) / len(current_drawings))
                
                progress_bar.empty()
            else:
                st.markdown(f"**{len(current_drawings)} profil(s) détecté(s)**")
                for i in range(len(current_drawings)):
                    st.write(f"• Profil {i+1}")
    
    elif st.session_state["analysis_mode"] == "zone_analysis":
        st.markdown("<div class='sub-header'>📐 Analyse Statistique de Zone</div>", 
                   unsafe_allow_html=True)
        
        raw_drawings = st.session_state.get("raw_drawings") or []
        polygons = [
            d for d in raw_drawings 
            if isinstance(d, dict) and d.get("geometry", {}).get("type") == "Polygon"
        ]
        
        if not polygons:
            st.warning("⚠️ Dessinez un polygone sur la carte")
        else:
            selected_zone = st.selectbox(
                "Sélectionnez une zone",
                [f"Zone {i+1}" for i in range(len(polygons))]
            )
            
            col_btn1, col_btn2 = st.columns([3, 1])
            with col_btn1:
                analyze_btn = st.button("🔍 Analyser", type="primary", use_container_width=True)
            with col_btn2:
                if st.button("↩️ Retour", use_container_width=True):
                    st.session_state["analysis_mode"] = "none"
                    st.rerun()
            
            if analyze_btn:
                zone_idx = int(selected_zone.split()[1]) - 1
                geometry = polygons[zone_idx]["geometry"]
                
                with st.spinner("Analyse en cours..."):
                    analysis_data = analyze_zone(mosaic_path, geometry)
                    
                    if analysis_data:
                        st.success("✅ Analyse terminée")
                        
                        stats = analysis_data["statistics"]
                        col1, col2, col3 = st.columns(3)
                        col1.metric("📐 Surface", f"{analysis_data['area']/10000:.2f} ha")
                        col2.metric("📊 Altitude Moyenne", f"{stats['mean']:.1f} m")
                        col3.metric("📏 Dénivelé Total", f"{stats['range']:.1f} m")
                        
                        fig = create_statistics_chart(analysis_data)
                        st.pyplot(fig)
                        store_figure(fig, "statistics", 
                                   f"{map_name} - Statistiques Zone {zone_idx+1}")
                        plt.close(fig)
    
    elif st.session_state["analysis_mode"] == "measurements":
        st.markdown("<div class='sub-header'>📏 Mesures et Statistiques</div>", 
                   unsafe_allow_html=True)
        
        raw_drawings = st.session_state.get("raw_drawings") or []
        
        if not raw_drawings:
            st.info("ℹ️ Dessinez des éléments sur la carte pour obtenir des mesures")
        else:
            st.markdown("### 📊 Résumé des Dessins")
            
            lines = [d for d in raw_drawings if d.get("geometry", {}).get("type") == "LineString"]
            polygons = [d for d in raw_drawings if d.get("geometry", {}).get("type") == "Polygon"]
            points = [d for d in raw_drawings if d.get("geometry", {}).get("type") == "Point"]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("📍 Points", len(points))
            col2.metric("📏 Lignes", len(lines))
            col3.metric("🔶 Polygones", len(polygons))
            
            if lines or polygons:
                st.markdown("### 📐 Détails des Mesures")
                
                measurements = []
                
                for i, line in enumerate(lines):
                    coords = line["geometry"]["coordinates"]
                    if len(coords) >= 2:
                        distance = sum(
                            haversine(coords[j][0], coords[j][1], coords[j+1][0], coords[j+1][1])
                            for j in range(len(coords)-1)
                        )
                        measurements.append({
                            "Type": "Ligne",
                            "ID": f"L{i+1}",
                            "Longueur (m)": f"{distance:.2f}",
                            "Surface (ha)": "-"
                        })
                
                for i, poly in enumerate(polygons):
                    coords = poly["geometry"]["coordinates"][0]
                    area = calculate_area(coords)
                    perimeter = sum(
                        haversine(coords[j][0], coords[j][1], coords[j+1][0], coords[j+1][1])
                        for j in range(len(coords)-1)
                    )
                    measurements.append({
                        "Type": "Polygone",
                        "ID": f"P{i+1}",
                        "Longueur (m)": f"{perimeter:.2f}",
                        "Surface (ha)": f"{area/10000:.2f}"
                    })
                
                df = pd.DataFrame(measurements)
                st.dataframe(df, use_container_width=True)
        
        if st.button("↩️ Retour", use_container_width=True):
            st.session_state["analysis_mode"] = "none"
            st.rerun()
    
    st.markdown("---")
    
    if st.session_state.get("analysis_results"):
        st.markdown("### 💾 Résultats Sauvegardés")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Total des analyses", len(st.session_state["analysis_results"]))
        
        with col2:
            if st.button("📥 Exporter en ZIP (PNG)", use_container_width=True):
                zip_buffer = export_results("PNG")
                if zip_buffer:
                    st.download_button(
                        "⬇️ Télécharger ZIP",
                        zip_buffer,
                        "analyses_cartographiques.zip",
                        "application/zip"
                    )
        
        with col3:
            if st.button("📄 Exporter en PDF", use_container_width=True):
                pdf_buffer = export_results("PDF")
                if pdf_buffer:
                    st.download_button(
                        "⬇️ Télécharger PDF",
                        pdf_buffer,
                        "rapport_analyses.pdf",
                        "application/pdf"
                    )

# ============================================================================
# INTERFACE - RAPPORT (reste inchangée)
# ============================================================================

def run_report():
    st.markdown("<h1 class='main-header'>📄 Génération de Rapport</h1>", 
                unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### 📝 Métadonnées du Rapport")
        titre = st.text_input("Titre principal", key="rapport_titre", value="RAPPORT CARTOGRAPHIQUE")
        report_id = st.text_input("ID du rapport", key="rapport_id", value=f"CARTO-{datetime.now().strftime('%Y%m%d')}")
        report_date = st.date_input("Date du rapport", date.today(), key="rapport_date")
        report_time = st.time_input("Heure du rapport", datetime.now().time(), key="rapport_time")
        editor = st.text_input("Éditeur", key="rapport_editor", value="CartoTools Pro")
        location = st.text_input("Localisation", key="rapport_location", value="Zone d'étude")
        company = st.text_input("Société", key="rapport_company", value="")
        logo = st.file_uploader("Logo", type=["png", "jpg", "jpeg"], key="rapport_logo")
    
    metadata = {
        'titre': titre,
        'report_id': report_id,
        'date': report_date,
        'time': report_time,
        'editor': editor,
        'location': location,
        'company': company,
        'logo': logo
    }
    
    if "elements" not in st.session_state:
        st.session_state["elements"] = []
    elements = st.session_state["elements"]
    
    tab1, tab2, tab3 = st.tabs(["➕ Ajouter Éléments", "👁️ Aperçu", "📄 Génération PDF"])
    
    with tab1:
        st.markdown("### 📊 Ajouter une carte d'analyse spatiale")
        analysis_card = create_analysis_card_controller()
        if analysis_card:
            if not any(el.get("analysis_ref") == analysis_card.get("analysis_ref") 
                      for el in elements if el["type"] == "Image"):
                elements.append(analysis_card)
                st.session_state["elements"] = elements
                st.success("✅ Carte d'analyse ajoutée avec succès !")
                st.rerun()
            else:
                st.warning("⚠️ Cette carte a déjà été ajoutée")
        
        st.markdown("---")
        st.markdown("### 📝 Ajouter un élément personnalisé")
        new_element = create_element_controller()
        if new_element:
            elements.append(new_element)
            st.session_state["elements"] = elements
            st.success("✅ Élément validé avec succès !")
            st.rerun()
    
    with tab2:
        if elements:
            display_elements_preview(elements)
            
            if st.button("🗑️ Supprimer tous les éléments", type="secondary"):
                st.session_state["elements"] = []
                st.rerun()
        else:
            st.info("ℹ️ Aucun élément ajouté pour le moment")
    
    with tab3:
        if not elements:
            st.warning("⚠️ Ajoutez au moins un élément avant de générer le PDF")
        else:
            st.markdown("### 🎯 Génération du Rapport PDF")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📋 Résumé")
                st.write(f"**Nombre d'éléments:** {len(elements)}")
                images = sum(1 for e in elements if e['type'] == 'Image')
                texts = sum(1 for e in elements if e['type'] == 'Texte')
                st.write(f"**Images:** {images} | **Textes:** {texts}")
            
            with col2:
                st.markdown("#### ℹ️ Métadonnées")
                st.write(f"**ID:** {metadata['report_id']}")
                st.write(f"**Éditeur:** {metadata['editor']}")
                st.write(f"**Date:** {metadata['date'].strftime('%d/%m/%Y')}")
            
            if st.button("🚀 Générer le PDF", type="primary", use_container_width=True):
                with st.spinner("Génération du PDF en cours..."):
                    pdf = generate_pdf(elements, metadata)
                    st.success("✅ Rapport généré avec succès!")
                    
                    st.download_button(
                        "⬇️ Télécharger le PDF",
                        pdf,
                        f"rapport_{metadata['report_id']}.pdf",
                        "application/pdf",
                        use_container_width=True
                    )

# ============================================================================
# INTERFACE - TABLEAU DE BORD (reste inchangée)
# ============================================================================

def run_dashboard():
    st.markdown("<h1 class='main-header'>📊 Tableau de Bord</h1>", 
                unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        analyses_count = len(st.session_state.get("analysis_results", []))
        st.metric("🔍 Analyses réalisées", analyses_count)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        contours = sum(1 for r in st.session_state.get("analysis_results", []) if r.get('type') == 'contour')
        st.metric("📈 Cartes de contours", contours)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        profiles = sum(1 for r in st.session_state.get("analysis_results", []) if r.get('type') == 'profile')
        st.metric("📊 Profils d'élévation", profiles)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        elements_count = len(st.session_state.get("elements", []))
        st.metric("📄 Éléments de rapport", elements_count)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.session_state.get("analysis_results"):
        st.markdown("### 📈 Historique des Analyses")
        
        for i, result in enumerate(st.session_state["analysis_results"][-5:], start=1):
            with st.expander(f"{result['title']} - {result['timestamp'].strftime('%d/%m/%Y %H:%M')}", expanded=False):
                col_img, col_info = st.columns([2, 1])
                
                with col_img:
                    st.image(result['image'], use_container_width=True)
                
                with col_info:
                    st.markdown(f"**Type:** {result['type'].upper()}")
                    st.markdown(f"**Titre:** {result['title']}")
                    st.markdown(f"**Date:** {result['timestamp'].strftime('%d/%m/%Y à %H:%M')}")
                    
                    if result.get('metadata'):
                        st.markdown("**Métadonnées:**")
                        for key, value in result['metadata'].items():
                            st.markdown(f"- {key}: {value}")
    else:
        st.info("ℹ️ Aucune analyse réalisée pour le moment. Rendez-vous dans l'onglet 'Analyse Spatiale' pour commencer.")
    
    st.markdown("---")
    st.markdown("### 🎯 Actions Rapides")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Effacer toutes les analyses", use_container_width=True):
            if st.session_state.get("analysis_results"):
                st.session_state["analysis_results"] = []
                st.success("✅ Analyses effacées")
                st.rerun()
    
    with col2:
        if st.button("🗑️ Effacer les éléments de rapport", use_container_width=True):
            if st.session_state.get("elements"):
                st.session_state["elements"] = []
                st.success("✅ Éléments effacés")
                st.rerun()
    
    with col3:
        if st.button("🔄 Réinitialiser l'application", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("✅ Application réinitialisée")
            st.rerun()

# ============================================================================
# FONCTIONS RAPPORT PDF (reste inchangées)
# ============================================================================

def draw_metadata(c, metadata):
    """Dessine les métadonnées sur le PDF"""
    margin = 40
    x_left = margin
    y_top = PAGE_HEIGHT - margin
    line_height = 16

    logo_drawn = False
    if metadata.get('logo'):
        try:
            if isinstance(metadata['logo'], bytes):
                logo_stream = BytesIO(metadata['logo'])
            else:
                logo_stream = metadata['logo']
            img = ImageReader(logo_stream)
            img_width, img_height = img.getSize()
            aspect = img_height / img_width
            desired_width = 50
            desired_height = desired_width * aspect
            c.drawImage(img, x_left, y_top - desired_height, width=desired_width, 
                       height=desired_height, preserveAspectRatio=True, mask='auto')
            logo_drawn = True
        except Exception as e:
            st.error(f"❌ Erreur logo: {e}")
    
    x_title = x_left + 60 if logo_drawn else x_left
    y_title = y_top - 20
    
    c.setFont("Helvetica-Bold", 22)
    c.setFillColor(colors.HexColor("#1f77b4"))
    if metadata.get('titre'):
        c.drawString(x_title, y_title, metadata['titre'])
    
    c.setFont("Helvetica", 14)
    c.setFillColor(colors.black)
    y_company = y_title - 25
    if metadata.get('company'):
        c.drawString(x_title, y_company, metadata['company'])
    
    y_line = y_company - 10
    c.setStrokeColor(colors.HexColor("#1f77b4"))
    c.setLineWidth(2)
    c.line(x_left, y_line, x_left + 200, y_line)
    c.setLineWidth(1)
    
    y_text = y_line - 20
    infos = [
        ("📋 ID Rapport", metadata.get('report_id', 'N/A')),
        ("📅 Date", metadata['date'].strftime('%d/%m/%Y') if hasattr(metadata.get('date'), "strftime") else str(metadata.get('date', 'N/A'))),
        ("🕐 Heure", metadata['time'].strftime('%H:%M') if hasattr(metadata.get('time'), "strftime") else str(metadata.get('time', 'N/A'))),
        ("👤 Éditeur", metadata.get('editor', 'N/A')),
        ("📍 Localisation", metadata.get('location', 'N/A'))
    ]
    
    value_x_offset = x_left + 90
    for label, value in infos:
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(colors.black)
        c.drawString(x_left, y_text, label)
        c.setFont("Helvetica", 10)
        c.setFillColor(colors.HexColor("#555555"))
        c.drawString(value_x_offset, y_text, str(value))
        y_text -= line_height

def calculate_dimensions(size):
    """Calcule les dimensions selon la taille"""
    dimensions = {
        "Grand": (PAGE_WIDTH, SECTION_HEIGHT),
        "Moyen": (COLUMN_WIDTH, SECTION_HEIGHT),
        "Petit": (COLUMN_WIDTH / 1.5, SECTION_HEIGHT)
    }
    return dimensions.get(size, (PAGE_WIDTH, SECTION_HEIGHT))

def calculate_position(element):
    """Calcule la position d'un élément"""
    vertical_offset = {
        "Haut": 0, 
        "Milieu": SECTION_HEIGHT, 
        "Bas": SECTION_HEIGHT*2
    }[element['v_pos']]
    
    if element['size'] == "Grand":
        return (0, PAGE_HEIGHT - vertical_offset - SECTION_HEIGHT)
    
    if element['h_pos'] == "Gauche":
        x = 0
    elif element['h_pos'] == "Droite":
        x = COLUMN_WIDTH
    else:
        x = COLUMN_WIDTH / 2 - calculate_dimensions(element['size'])[0] / 2
    
    return (x, PAGE_HEIGHT - vertical_offset - SECTION_HEIGHT)

def generate_pdf(elements, metadata):
    """Génère le PDF du rapport"""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    
    c.setAuthor(metadata.get('editor', 'CartoTools Pro'))
    c.setTitle(metadata.get('report_id', 'Rapport Cartographique'))
    
    for element in elements:
        width, height = calculate_dimensions(element['size'])
        x, y = calculate_position(element)
        
        if element['type'] == "Image":
            if element.get("content") is not None:
                try:
                    if isinstance(element["content"], bytes):
                        image_stream = BytesIO(element["content"])
                    else:
                        image_stream = element["content"]
                    img = ImageReader(image_stream)
                    
                    top_margin = 20
                    bottom_margin = 25
                    horizontal_scale = 0.9
                    image_actual_width = width * horizontal_scale
                    image_actual_height = height - top_margin - bottom_margin
                    image_x = x + (width - image_actual_width) / 2
                    image_y = y + bottom_margin
                    
                    c.drawImage(img, image_x, image_y, width=image_actual_width, 
                               height=image_actual_height, preserveAspectRatio=True, mask='auto')
                    
                    if element.get("image_title"):
                        c.setFont("Helvetica-Bold", 12)
                        c.setFillColor(colors.HexColor("#2c3e50"))
                        image_title = element["image_title"].upper()
                        c.drawCentredString(x + width / 2, y + height - top_margin / 2, image_title)
                    
                    if element.get("description"):
                        c.setFont("Helvetica", 9)
                        c.setFillColor(colors.gray)
                        c.drawRightString(x + width - 10, y + bottom_margin / 2, 
                                         element["description"][:100])
                        c.setFillColor(colors.black)
                except Exception as e:
                    st.error(f"❌ Erreur image: {e}")
        else:
            text = element['content']
            style = getSampleStyleSheet()["Normal"]
            style.fontSize = 14 if element['size'] == "Grand" else 12 if element['size'] == "Moyen" else 10
            p = Paragraph(text, style)
            p.wrapOn(c, width - 20, height - 20)
            p.drawOn(c, x + 10, y + 10)
    
    draw_metadata(c, metadata)
    
    c.save()
    buffer.seek(0)
    return buffer

def generate_analysis_report():
    """Génère un rapport automatique des analyses"""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    
    c.setFont("Helvetica-Bold", 24)
    c.setFillColor(colors.HexColor("#1f77b4"))
    c.drawCentredString(PAGE_WIDTH/2, PAGE_HEIGHT - 100, "RAPPORT D'ANALYSE CARTOGRAPHIQUE")
    
    c.setFont("Helvetica", 14)
    c.setFillColor(colors.black)
    c.drawCentredString(PAGE_WIDTH/2, PAGE_HEIGHT - 140, f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}")
    
    c.setFont("Helvetica", 12)
    c.drawCentredString(PAGE_WIDTH/2, PAGE_HEIGHT - 170, "CartoTools Pro v2.0")
    
    y_pos = PAGE_HEIGHT - 220
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y_pos, "📊 Résumé de l'analyse")
    y_pos -= 30
    
    results = st.session_state.get("analysis_results", [])
    c.setFont("Helvetica", 11)
    c.drawString(70, y_pos, f"• Nombre d'analyses réalisées: {len(results)}")
    y_pos -= 20
    
    contours = sum(1 for r in results if r['type'] == 'contour')
    profiles = sum(1 for r in results if r['type'] == 'profile')
    
    c.drawString(70, y_pos, f"• Cartes de contours: {contours}")
    y_pos -= 20
    c.drawString(70, y_pos, f"• Profils d'élévation: {profiles}")
    
    for i, result in enumerate(results):
        c.showPage()
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, PAGE_HEIGHT - 50, f"{i+1}. {result['title']}")
        
        try:
            img = ImageReader(BytesIO(result['image']))
            c.drawImage(img, 50, PAGE_HEIGHT - 550, width=PAGE_WIDTH - 100, 
                       height=400, preserveAspectRatio=True, mask='auto')
        except:
            pass
        
        c.setFont("Helvetica", 10)
        c.drawString(50, PAGE_HEIGHT - 570, 
                    f"Créé le: {result['timestamp'].strftime('%d/%m/%Y à %H:%M')}")
    
    c.save()
    buffer.seek(0)
    return buffer

# ============================================================================
# FONCTIONS RAPPORT - CONTRÔLEURS (reste inchangées)
# ============================================================================

def create_analysis_card_controller():
    """Contrôleur pour ajouter une carte d'analyse spatiale"""
    with st.expander("➕ Ajouter une carte d'analyse spatiale", expanded=True):
        if "analysis_results" not in st.session_state or not st.session_state["analysis_results"]:
            st.info("ℹ️ Aucune carte d'analyse disponible. Générez d'abord des analyses dans l'onglet 'Analyse Spatiale'.")
            return None
        
        options = {f"{i+1} - {res['title']}": i for i, res in enumerate(st.session_state["analysis_results"])}
        chosen = st.selectbox("Choisissez une carte", list(options.keys()), key="analysis_card_select")
        idx = options[chosen]
        
        col1, col2 = st.columns(2)
        with col1:
            size = st.selectbox("Taille", ["Grand", "Moyen", "Petit"], key="analysis_card_size")
        with col2:
            v_pos = st.selectbox("Position verticale", ["Haut", "Milieu", "Bas"], key="analysis_card_v_pos")
            h_pos = st.selectbox("Position horizontale", ["Gauche", "Droite", "Centre"], key="analysis_card_h_pos")
        
        title_input = st.text_input("Titre pour la carte", key="analysis_card_title", 
                                    value=st.session_state["analysis_results"][idx]["title"])
        description_input = st.text_input("Description pour la carte", key="analysis_card_description", 
                                         value="Carte générée depuis l'analyse spatiale")
        
        if st.button("✅ Valider la carte d'analyse", key="validate_analysis_card"):
            return {
                "type": "Image",
                "size": size,
                "v_pos": v_pos,
                "h_pos": h_pos,
                "content": st.session_state["analysis_results"][idx]["image"],
                "image_title": title_input,
                "description": description_input,
                "analysis_ref": idx
            }
    return None

def create_element_controller():
    """Contrôleur pour ajouter un élément personnalisé"""
    with st.expander("➕ Ajouter un élément personnalisé", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            elem_type = st.selectbox("Type", ["Image", "Texte"], key="rapport_elem_type")
            size = st.selectbox("Taille", ["Grand", "Moyen", "Petit"], key="rapport_elem_size")
        with col2:
            vertical_pos = st.selectbox("Position verticale", ["Haut", "Milieu", "Bas"], key="rapport_v_pos")
            horizontal_options = ["Gauche", "Droite", "Centre"] if size == "Petit" else ["Gauche", "Droite"]
            horizontal_pos = st.selectbox("Position horizontale", horizontal_options, key="rapport_h_pos")
        
        if elem_type == "Image":
            content = st.file_uploader("Contenu (image)", type=["png", "jpg", "jpeg"], key="rapport_content_image")
            image_title = st.text_input("Titre de l'image", max_chars=50, key="rapport_image_title")
            description = st.text_input("Description brève (max 100 caractères)", max_chars=100, key="rapport_image_desc")
        else:
            content = st.text_area("Contenu", key="rapport_content_text")
        
        if st.button("✅ Valider l'élément", key="rapport_validate_element"):
            if elem_type == "Image" and content is None:
                st.error("❌ Veuillez charger une image pour cet élément.")
                return None
            element_data = {
                "type": elem_type,
                "size": size,
                "v_pos": vertical_pos,
                "h_pos": horizontal_pos,
                "content": content,
            }
            if elem_type == "Image":
                element_data["image_title"] = image_title
                element_data["description"] = description
            return element_data
    return None

def display_elements_preview(elements):
    """Affiche l'aperçu des éléments validés"""
    st.markdown("### 📋 Aperçu des éléments validés")
    
    if not elements:
        st.info("ℹ️ Aucun élément ajouté pour le moment")
        return
    
    for idx, element in enumerate(elements, start=1):
        with st.expander(f"Élément {idx} - {element['type']} ({element['size']})", expanded=False):
            col1, col2 = st.columns([3, 1])
            with col1:
                if element["type"] == "Image":
                    st.image(element["content"], width=300)
                    if element.get("image_title"):
                        st.markdown(f"**Titre:** {element['image_title']}")
                    if element.get("description"):
                        st.markdown(f"*Description:* {element['description']}")
                else:
                    st.markdown(f"**Texte:** {element['content'][:100]}...")
                
                st.markdown(f"**Position:** {element['v_pos']} - {element['h_pos']}")
            
            with col2:
                if st.button("🗑️ Supprimer", key=f"delete_element_{idx}"):
                    elements.pop(idx - 1)
                    st.session_state["elements"] = elements
                    st.rerun()

# ============================================================================
# APPLICATION PRINCIPALE
# ============================================================================

def main():
    initialize_session_state()
    
    st.sidebar.markdown("# 🗺️ CartoTools Pro")
    st.sidebar.markdown("### Navigation")
    
    menu = st.sidebar.radio(
        "Menu Principal",
        ["📊 Tableau de Bord", "🔍 Analyse Spatiale", "📄 Rapport", "❓ Aide"],
        key="main_menu"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 À propos")
    st.sidebar.info("""
    **CartoTools Pro v2.0**
    
    Application professionnelle d'analyse cartographique et de génération de rapports.
    
    © 2024 CartoTools
    """)
    
    if menu == "📊 Tableau de Bord":
        run_dashboard()
    elif menu == "🔍 Analyse Spatiale":
        run_analysis_spatiale()
    elif menu == "📄 Rapport":
        run_report()
    elif menu == "❓ Aide":
        run_help()

# ============================================================================
# INTERFACE - AIDE (reste inchangée)
# ============================================================================

def run_help():
    st.markdown("<h1 class='main-header'>❓ Aide & Documentation</h1>", 
                unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Démarrage Rapide", "📈 Analyse Spatiale", "📄 Rapports", "⚙️ Configuration"])
    
    with tab1:
        st.markdown("""
        ## 🚀 Démarrage Rapide
        
        ### Préparation des données
        1. **Créez un dossier `TIFF`** à la racine du projet
        2. **Placez vos fichiers GeoTIFF** (.tif, .tiff) dans ce dossier
        3. Les fichiers seront automatiquement fusionnés en mosaïque
        
        ### Première utilisation
        1. Lancez l'application avec `streamlit run app.py`
        2. Accédez à l'onglet **Analyse Spatiale**
        3. La carte interactive se charge avec votre mosaïque
        4. Utilisez les outils de dessin pour créer vos analyses
        
        ### Navigation
        - **📊 Tableau de Bord** : Vue d'ensemble de vos analyses
        - **🔍 Analyse Spatiale** : Création de contours, profils et statistiques
        - **📄 Rapport** : Génération de rapports PDF personnalisés
        - **❓ Aide** : Documentation complète
        """)
    
    with tab2:
        st.markdown("""
        ## 📈 Analyse Spatiale
        
        ### Génération de Contours
        1. **Dessinez un rectangle/polygone** sur la zone d'intérêt
        2. Cliquez sur **"Générer des Contours"**
        3. Sélectionnez les zones à analyser
        4. Configurez les paramètres :
           - Nombre de niveaux (5-30)
           - Palette de couleurs
           - Opacité du fond de carte
           - Ombrage du relief
        
        ### Profils d'Élévation
        1. **Dessinez une ligne** sur votre tracé
        2. Cliquez sur **"Tracer des Profils"**
        3. Les profils incluent :
           - Graphique d'élévation
           - Analyse des pentes
           - Statistiques (min, max, dénivelé)
        
        ### Analyse de Zone
        1. **Dessinez un polygone** sur la zone
        2. Cliquez sur **"Analyser une Zone"**
        3. Obtenez :
           - Surface en hectares
           - Statistiques d'élévation
           - Distribution des altitudes
        
        ### Mesures & Statistiques
        - Vue d'ensemble de tous vos dessins
        - Calcul automatique des longueurs et surfaces
        - Export des données en tableau
        
        ### Gestion des Fichiers
        - **TIFF personnalisés** : Uploadez vos propres fichiers d'élévation
        - **Shapefiles** : Ajoutez des couches vectorielles
        - **CSV/TXT** : Importez des points avec coordonnées
        """)
    
    with tab3:
        st.markdown("""
        ## 📄 Génération de Rapports
        
        ### Métadonnées
        Configurez les informations du rapport dans la barre latérale :
        - Titre principal
        - ID du rapport
        - Date et heure
        - Éditeur
        - Localisation
        - Logo (optionnel)
        
        ### Ajout d'éléments
        
        #### Cartes d'analyse
        1. Générez d'abord des analyses
        2. Dans l'onglet Rapport, sélectionnez une carte
        3. Configurez la taille et position
        4. Ajoutez un titre et description
        
        #### Éléments personnalisés
        - **Images** : Téléversez vos propres images
        - **Textes** : Ajoutez des commentaires
        - Configurez taille et position pour chaque élément
        
        ### Mise en page
        - **Tailles** : Grand (pleine page), Moyen (demi-page), Petit
        - **Positions verticales** : Haut, Milieu, Bas
        - **Positions horizontales** : Gauche, Droite, Centre
        
        ### Génération PDF
        1. Vérifiez l'aperçu des éléments
        2. Cliquez sur **"Générer le PDF"**
        3. Téléchargez votre rapport
        """)
    
    with tab4:
        st.markdown("""
        ## ⚙️ Configuration
        
        ### Paramètres des Contours
        - **Nombre de niveaux** : Plus de niveaux = contours plus détaillés
        - **Palette de couleurs** : Choisissez selon vos besoins
          - `terrain` : Naturel (vert-brun-blanc)
          - `viridis` : Contraste élevé
          - `plasma` : Chaud (violet-orange)
          - `coolwarm` : Bleu-rouge
        - **Opacité fond de carte** : 0 (transparent) à 1 (opaque)
        - **Ombrage du relief** : Ajoute un effet 3D
        
        ### Paramètres des Profils
        - **Résolution** : Distance entre points (10-200m)
          - Basse (10-30m) : Détails précis, calculs longs
          - Moyenne (40-70m) : Bon compromis
          - Haute (80-200m) : Vue d'ensemble rapide
        - **Afficher les pentes** : Graphique des variations
        
        ### Options de la Carte
        - **Mini-carte** : Navigation rapide
        - **Plein écran** : Vue étendue
        - **Mesures** : Outils de mesure intégrés
        
        ### Format d'Export
        - **PNG (ZIP)** : Images individuelles
        - **PDF** : Rapport automatique complet
        """)
    
    st.markdown("---")
    st.markdown("""
    ## 💡 Conseils & Astuces
    
    - **Performance** : Pour de grandes zones, utilisez moins de niveaux de contours
    - **Précision** : Dessinez des zones plus petites pour des analyses détaillées
    - **Organisation** : Nommez clairement vos projets pour les retrouver facilement
    - **Sauvegarde** : Exportez régulièrement vos analyses
    - **Qualité PDF** : Limitez à 10-15 éléments par rapport pour une meilleure lisibilité
    
    ## 🐛 Résolution de problèmes
    
    **La mosaïque ne se charge pas**
    - Vérifiez que le dossier TIFF existe
    - Vérifiez que les fichiers sont au format .tif ou .tiff
    - Vérifiez les droits d'accès au dossier
    
    **Les contours ne s'affichent pas**
    - Dessinez un rectangle/polygone valide
    - Vérifiez que la zone intersecte la mosaïque
    
    **Le PDF ne se génère pas**
    - Vérifiez que tous les éléments sont valides
    - Réduisez le nombre d'éléments si nécessaire
    - Vérifiez les métadonnées (tous les champs requis)
    """)

if __name__ == "__main__":
    main()

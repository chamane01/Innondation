import streamlit as st
import os
import uuid
import json
import base64
import math
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.merge
import rasterio.mask
import rasterio.warp
from rasterio.plot import reshape_as_image
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, transform_bounds, transform_geom
from rasterio.mask import mask
import matplotlib.pyplot as plt
import folium
from folium.plugins import Draw, MeasureControl
from folium import LayerControl
from streamlit_folium import st_folium
from shapely.geometry import shape, Polygon, Point, LineString
from shapely.geometry import LineString as ShapelyLineString
from PIL import Image
from io import BytesIO
from datetime import date, datetime

# ============================
# INITIALISATIONS DE SESSION
# ============================
if "layers" not in st.session_state:
    st.session_state["layers"] = {}           # Couches créées par l'utilisateur
if "uploaded_layers" not in st.session_state:
    st.session_state["uploaded_layers"] = []    # Couches téléversées (TIFF et GeoJSON)
if "new_features" not in st.session_state:
    st.session_state["new_features"] = []       # Entités dessinées temporairement
if "analysis_results" not in st.session_state:
    st.session_state["analysis_results"] = []   # Résultats d'analyse à intégrer au rapport
if 'active_button' not in st.session_state:
    st.session_state['active_button'] = None

# ============================
# FONCTIONS UTILITAIRES (RAPPORT & ANALYSE)
# ============================

# --- Conversion d'images ---
def image_bytes_to_data_url(image_bytes):
    base64_str = base64.b64encode(image_bytes).decode('utf-8')
    return 'data:image/png;base64,' + base64_str

def get_bounds_from_geometry(geometry):
    coords = geometry.get("coordinates")[0]
    lons = [pt[0] for pt in coords]
    lats = [pt[1] for pt in coords]
    return [[min(lats), min(lons)], [max(lats), max(lons)]]

# --- Analyse spatiale & dessin (SIG) ---
def load_tiff_files(folder_path):
    try:
        tiff_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.tif')]
    except Exception as e:
        st.error(f"Erreur lors de la lecture du dossier {folder_path} : {e}")
        return []
    if not tiff_files:
        st.error("Aucun fichier TIFF trouvé dans le dossier.")
        return []
    return [f for f in tiff_files if os.path.exists(f)]

def build_mosaic(tiff_files, mosaic_path="mosaic.tif"):
    try:
        src_files = [rasterio.open(fp) for fp in tiff_files]
        mosaic, out_trans = rasterio.merge.merge(src_files)
        out_meta = src_files[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans
        })
        with rasterio.open(mosaic_path, "w", **out_meta) as dest:
            dest.write(mosaic)
        for src in src_files:
            src.close()
        return mosaic_path
    except Exception as e:
        st.error(f"Erreur lors de la création de la mosaïque : {e}")
        return None

# --- Reprojection et coloration des TIFF ---
def reproject_tiff(input_tiff, target_crs):
    with rasterio.open(input_tiff) as src:
        transform, width, height = calculate_default_transform(src.crs, target_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height,
        })
        unique_id = str(uuid.uuid4())[:8]
        reprojected_tiff = f"reprojected_{unique_id}.tiff"
        with rasterio.open(reprojected_tiff, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest,
                )
    return reprojected_tiff

def apply_color_gradient(tiff_path, output_path):
    with rasterio.open(tiff_path) as src:
        dem_data = src.read(1)
        cmap = plt.get_cmap("terrain")
        norm = plt.Normalize(vmin=dem_data.min(), vmax=dem_data.max())
        colored_image = cmap(norm(dem_data))
        plt.imsave(output_path, colored_image)
        plt.close()

# --- Ajout d'image sur la carte ---
def add_image_overlay(map_object, tiff_path, bounds, name):
    with rasterio.open(tiff_path) as src:
        image = reshape_as_image(src.read())
        folium.raster_layers.ImageOverlay(
            image=image,
            bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
            name=name,
            opacity=0.6,
        ).add_to(map_object)

# --- Fonctions pour GeoJSON et polygones ---
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

def calculate_geojson_bounds(geojson_data):
    gdf = gpd.GeoDataFrame.from_features(geojson_data)
    return gdf.total_bounds

def load_tiff(tiff_path):
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

def validate_projection_and_extent(raster_path, polygons_gdf, target_crs):
    with rasterio.open(raster_path) as src:
        if src.crs != target_crs:
            raise ValueError(f"Le raster {raster_path} n'est pas dans la projection {target_crs}")
        polygons_gdf = polygons_gdf.to_crs(src.crs)
        raster_bounds = src.bounds
        for idx, row in polygons_gdf.iterrows():
            if not row.geometry.intersects(Polygon.from_bounds(*raster_bounds)):
                st.warning(f"Le polygone {idx} est en dehors de l'emprise du raster")
    return polygons_gdf

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

# --- Fonctions pour profils, contours et stockage des résultats ---
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371000
    return c * r

def interpolate_line(coords, step=50):
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

def generate_profile(mosaic_file, coords, profile_title):
    try:
        points, distances = interpolate_line(coords)
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
    except Exception as e:
        st.error(f"Erreur lors de la génération du profil : {e}")
        return None
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(distances, elevations, 'b-', linewidth=1.5)
    ax.set_title(profile_title)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Altitude (m)")
    return fig

def generate_contours(mosaic_file, drawing_geometry):
    try:
        with rasterio.open(mosaic_file) as src:
            geom = drawing_geometry
            if src.crs.to_string() != "EPSG:4326":
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
            if center_lat >= 0:
                utm_crs = f"EPSG:{32600 + utm_zone}"
            else:
                utm_crs = f"EPSG:{32700 + utm_zone}"
            x_flat = X.flatten()
            y_flat = Y.flatten()
            X_utm_flat, Y_utm_flat = transform(src.crs, utm_crs, x_flat, y_flat)
            X_utm = np.array(X_utm_flat).reshape(X.shape)
            Y_utm = np.array(Y_utm_flat).reshape(Y.shape)
            geom_utm = transform_geom("EPSG:4326", utm_crs, drawing_geometry)
            envelope = shape(geom_utm)
            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)
            vmin = np.nanmin(data)
            vmax = np.nanmax(data)
            levels = np.linspace(vmin, vmax, 15)
            fig, ax = plt.subplots(figsize=(8, 6))
            cs = ax.contour(X_utm, Y_utm, data, levels=levels, cmap='terrain', zorder=3)
            ax.clabel(cs, inline=True, fontsize=8)
            ax.set_title("Contours d'élévation (UTM)")
            ax.set_xlabel("UTM Easting")
            ax.set_ylabel("UTM Northing")
            minx, miny, maxx, maxy = envelope.bounds
            dx = (maxx - minx) * 0.05
            dy = (maxy - miny) * 0.05
            ax.set_xlim(minx - dx, maxx + dx)
            ax.set_ylim(miny - dy, maxy + dy)
            import contextily as ctx
            ctx.add_basemap(ax, crs=utm_crs, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.5)
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
                                        ax.plot(x_other, y_other, color='black', linestyle='-', linewidth=2, label=label, zorder=4)
                                    else:
                                        for part in clipped.geoms:
                                            x_other, y_other = part.exterior.xy
                                            ax.plot(x_other, y_other, color='black', linestyle='-', linewidth=2, label=label, zorder=4)
                                elif clipped.geom_type in ["LineString", "MultiLineString"]:
                                    current_profile_label = f"Profil {profile_counter}"
                                    profile_counter += 1
                                    legend_label = "Profil" if not added_profile else "_nolegend_"
                                    if not added_profile:
                                        added_profile = True
                                    if clipped.geom_type == "LineString":
                                        x_other, y_other = clipped.xy
                                        ax.plot(x_other, y_other, color='black', linestyle='-', linewidth=2, label=legend_label, zorder=4)
                                        if len(x_other) >= 2:
                                            dx = x_other[1] - x_other[0]
                                            dy = y_other[1] - y_other[0]
                                            angle = np.degrees(np.arctan2(dy, dx))
                                        else:
                                            angle = 0
                                        centroid = clipped.centroid
                                        ax.text(centroid.x, centroid.y, current_profile_label, fontsize=8, color='black', ha='center', va='center', rotation=angle, zorder=6)
                                    else:
                                        for part in clipped.geoms:
                                            x_other, y_other = part.xy
                                            ax.plot(x_other, y_other, color='black', linestyle='-', linewidth=2, label=legend_label, zorder=4)
                                            if len(x_other) >= 2:
                                                dx = x_other[1] - x_other[0]
                                                dy = y_other[1] - y_other[0]
                                                angle = np.degrees(np.arctan2(dy, dx))
                                            else:
                                                angle = 0
                                            centroid = part.centroid
                                            ax.text(centroid.x, centroid.y, current_profile_label, fontsize=8, color='black', ha='center', va='center', rotation=angle, zorder=6)
                                elif clipped.geom_type in ["Point", "MultiPoint"]:
                                    label = "Point" if not added_point else "_nolegend_"
                                    if not added_point:
                                        added_point = True
                                    if clipped.geom_type == "Point":
                                        ax.plot(clipped.x, clipped.y, 'o', color='black', markersize=8, label=label, zorder=4)
                                    else:
                                        for part in clipped.geoms:
                                            ax.plot(part.x, part.y, 'o', color='black', markersize=8, label=label, zorder=4)
                        except Exception as e:
                            st.error(f"Erreur lors du tracé d'un dessin supplémentaire : {e}")
            x_env, y_env = envelope.exterior.xy
            ax.plot(x_env, y_env, color='red', linewidth=2, label="Zone dessinée", zorder=5)
            leg = ax.legend(loc='lower right', framealpha=1, facecolor='white', fontsize=8)
            for text in leg.get_texts():
                text.set_fontsize(8)
            return fig
    except Exception as e:
        st.error(f"Erreur lors de la génération des contours : {e}")
        return None

def store_figure(fig, result_type, title):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    st.session_state["analysis_results"].append({
        "type": result_type,
        "title": title,
        "image": buf.getvalue()
    })

# --- Fonctions pour le rapport ---
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph
from reportlab.lib import colors

def calculate_dimensions(size):
    dimensions = {
        "Grand": (A4[0], A4[1]/3),
        "Moyen": (A4[0]/2, A4[1]/3),
        "Petit": (A4[0]/3, A4[1]/3)
    }
    return dimensions.get(size, (A4[0], A4[1]/3))

def calculate_position(element):
    vertical_offset = {"Haut": 0, "Milieu": A4[1]/3, "Bas": 2*(A4[1]/3)}[element['v_pos']]
    if element['size'] == "Grand":
        return (0, A4[1] - vertical_offset - A4[1]/3)
    if element['h_pos'] == "Gauche":
        x = 0
    elif element['h_pos'] == "Droite":
        x = A4[0]/2
    else:
        x = A4[0]/4 - calculate_dimensions(element['size'])[0]/2
    return (x, A4[1] - vertical_offset - A4[1]/3)

def draw_metadata(c, metadata):
    margin = 40
    x_left = margin
    y_top = A4[1] - margin
    line_height = 16
    logo_drawn = False
    if metadata['logo']:
        try:
            if isinstance(metadata['logo'], bytes):
                logo_stream = BytesIO(metadata['logo'])
            else:
                logo_stream = metadata['logo']
            img = ImageReader(logo_stream)
            img_width, img_height = img.getSize()
            aspect = img_height / img_width
            desired_width = 40
            desired_height = desired_width * aspect
            c.drawImage(img, x_left, y_top - desired_height, width=desired_width, height=desired_height, preserveAspectRatio=True, mask='auto')
            logo_drawn = True
        except Exception as e:
            st.error(f"Erreur de chargement du logo: {str(e)}")
    if logo_drawn:
        x_title = x_left + 50
        y_title = y_top - 20
    else:
        x_title = x_left
        y_title = y_top - 20
    c.setFont("Helvetica-Bold", 20)
    c.setFillColor(colors.black)
    if metadata.get('titre'):
        c.drawString(x_title, y_title, metadata['titre'])
    c.setFont("Helvetica", 14)
    y_company = y_title - 25
    if metadata.get('company'):
        c.drawString(x_title, y_company, metadata['company'])
    y_line = y_company - 10
    c.setStrokeColor(colors.darkgray)
    c.setLineWidth(2)
    c.line(x_left, y_line, x_left + 150, y_line)
    c.setLineWidth(1)
    y_text = y_line - 20
    infos = [
        ("ID Rapport", metadata['report_id']),
        ("Date", metadata['date'].strftime('%d/%m/%Y') if hasattr(metadata['date'], "strftime") else metadata['date']),
        ("Heure", metadata['time'].strftime('%H:%M') if hasattr(metadata['time'], "strftime") else metadata['time']),
        ("Éditeur", metadata['editor']),
        ("Localisation", metadata['location'])
    ]
    value_x_offset = x_left + 70
    for label, value in infos:
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(colors.black)
        c.drawString(x_left, y_text, label + ":")
        c.setFont("Helvetica", 10)
        c.drawString(value_x_offset, y_text, str(value))
        y_text -= line_height

def generate_pdf(elements, metadata):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setAuthor(metadata['editor'])
    c.setTitle(metadata['report_id'])
    for element in elements:
        width, height = calculate_dimensions(element['size'])
        x, y = calculate_position(element)
        if element['type'] == "Image":
            if element["content"] is not None:
                try:
                    if isinstance(element["content"], bytes):
                        image_stream = BytesIO(element["content"])
                    else:
                        image_stream = element["content"]
                    img = ImageReader(image_stream)
                    top_margin = 20
                    bottom_margin = 20
                    horizontal_scale = 0.9
                    image_actual_width = width * horizontal_scale
                    image_actual_height = height - top_margin - bottom_margin
                    image_x = x + (width - image_actual_width) / 2
                    image_y = y + bottom_margin
                    c.drawImage(img, image_x, image_y, width=image_actual_width, height=image_actual_height, preserveAspectRatio=True, mask='auto')
                    if element.get("image_title"):
                        c.setFont("Helvetica-Bold", 12)
                        image_title = element["image_title"].upper()
                        c.drawCentredString(x + width / 2, y + height - top_margin / 2, image_title)
                    if element.get("description"):
                        c.setFont("Helvetica", 10)
                        c.setFillColor(colors.gray)
                        c.drawRightString(x + width - 10, y + bottom_margin / 2, element["description"][:100])
                        c.setFillColor(colors.black)
                except Exception as e:
                    st.error(f"Erreur d'image: {str(e)}")
            else:
                st.error("Une image validée est introuvable.")
        else:
            text = element['content']
            style = getSampleStyleSheet()["Normal"]
            style.fontSize = 14 if element['size'] == "Grand" else 12 if element['size'] == "Moyen" else 10
            p = Paragraph(text, style)
            p.wrapOn(c, width, height)
            p.drawOn(c, x, y)
    draw_metadata(c, metadata)
    c.save()
    buffer.seek(0)
    return buffer

def display_elements_preview(elements):
    st.markdown("## Aperçu des éléments validés")
    for idx, element in enumerate(elements, start=1):
        st.markdown(f"**Élément {idx}**")
        if element["type"] == "Image":
            st.image(element["content"], width=200)
            if element.get("image_title"):
                st.markdown(f"*Titre de l'image :* **{element['image_title'].upper()}**")
            if element.get("description"):
                st.markdown(f"<span style='color:gray'>*Description :* {element['description']}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"**Texte :** {element['content']}")
        st.markdown("---")

def create_analysis_card_controller():
    with st.expander("➕ Ajouter une carte d'analyse spatiale", expanded=True):
        if not st.session_state["analysis_results"]:
            st.info("Aucune carte d'analyse spatiale n'est disponible pour le moment.")
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
        title_input = st.text_input("Titre pour la carte", key="analysis_card_title", value=st.session_state["analysis_results"][idx]["title"])
        description_input = st.text_input("Description pour la carte", key="analysis_card_description", value="Carte générée depuis l'analyse spatiale")
        if st.button("Valider la carte d'analyse", key="validate_analysis_card"):
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
    with st.expander("➕ Ajouter un élément", expanded=True):
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
        if st.button("Valider l'élément", key="rapport_validate_element"):
            if elem_type == "Image" and content is None:
                st.error("Veuillez charger une image pour cet élément.")
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

# --- Affichage des paramètres d'analyse ---
def display_parameters(button_name):
    if button_name == "Surfaces et volumes":
        st.markdown("### Calcul des volumes et des surfaces")
        method = st.radio("Choisissez la méthode de calcul :", ("Méthode 1 : MNS - MNT", "Méthode 2 : MNS seul"), key="volume_method")
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
                volumes, areas = calculate_volume_and_area_for_each_polygon(mns_utm_path, mnt_utm_path, polygons_gdf_utm)
            else:
                use_average_elevation = st.checkbox("Utiliser la cote moyenne des élévations sur les bords de la polygonale comme référence", value=True, key="use_average_elevation")
                reference_altitude = None
                if not use_average_elevation:
                    reference_altitude = st.number_input("Entrez l'altitude de référence (en mètres) :", value=0.0, step=0.1, key="reference_altitude")
                volumes, areas = calculate_volume_and_area_with_mns_only(mns_utm_path, polygons_gdf_utm, use_average_elevation=use_average_elevation, reference_altitude=reference_altitude)
            global_volume = calculate_global_volume(volumes)
            global_area = calculate_global_area(areas)
            st.write(f"Volume global : {global_volume:.2f} m³")
            st.write(f"Surface globale : {global_area:.2f} m²")
            os.remove(mns_utm_path)
            if method == "Méthode 1 : MNS - MNT":
                os.remove(mnt_utm_path)
        except Exception as e:
            st.error(f"Erreur lors du calcul : {e}")
    elif button_name == "Carte de contours":
        st.markdown("### Générer des contours")
        drawing_geometries = []
        raw_drawings = st.session_state.get("new_features") or []
        for drawing in raw_drawings:
            if isinstance(drawing, dict) and drawing.get("geometry", {}).get("type") == "Polygon":
                drawing_geometries.append(drawing.get("geometry"))
        if not drawing_geometries:
            st.warning("Veuillez dessiner au moins un rectangle sur la carte pour définir une zone.")
        else:
            options_list = [f"Rectangle {i+1}" for i in range(len(drawing_geometries))]
            selected_indices = st.multiselect("Sélectionnez les rectangles pour générer des contours", options=options_list)
            # Pour le calcul, nous utilisons la mosaïque générée à partir des TIFF du dossier "TIFF"
            folder_path = "TIFF"
            tiff_files = load_tiff_files(folder_path)
            mosaic_path = build_mosaic(tiff_files)
            for sel in selected_indices:
                idx = int(sel.split()[1]) - 1
                geometry = drawing_geometries[idx]
                fig = generate_contours(mosaic_path, geometry)
                if fig is not None:
                    st.pyplot(fig)
                    store_figure(fig, "contour", f"Contours - Emprise {idx+1}")
            os.remove(mosaic_path)
    elif button_name == "Tracer des profils":
        st.markdown("### Tracer des profils")
        raw_drawings = st.session_state.get("new_features") or []
        current_drawings = [d for d in raw_drawings if isinstance(d, dict) and d.get("geometry", {}).get("type") == "LineString"]
        if not current_drawings:
            st.info("Aucune ligne tracée pour le moment. Veuillez dessiner une ligne sur la carte.")
        else:
            folder_path = "TIFF"
            tiff_files = load_tiff_files(folder_path)
            mosaic_path = build_mosaic(tiff_files)
            for i, drawing in enumerate(current_drawings):
                profile_title = f"Profil {i+1}"
                st.markdown(f"#### {profile_title}")
                try:
                    fig = generate_profile(mosaic_path, drawing["geometry"]["coordinates"], profile_title)
                    if fig is not None:
                        st.pyplot(fig)
                        store_figure(fig, "profile", profile_title)
                except Exception as e:
                    st.error(f"Erreur de traitement : {e}")
            os.remove(mosaic_path)
    elif button_name == "Trouver un point":
        st.markdown("### Recherche d'un point")
        lon = st.number_input("Longitude", value=0.0, step=0.0001)
        lat = st.number_input("Latitude", value=0.0, step=0.0001)
        st.write(f"Point recherché : ({lat}, {lon})")
    elif button_name == "Télécharger la carte":
        st.markdown("### Téléchargement de la carte")
        st.info("Cliquez sur le bouton de téléchargement ci-dessous pour récupérer la carte sous forme d'image.")
        # Pour simplifier, nous utilisons ici la dernière carte générée enregistrée dans l'analyse des résultats
        if st.session_state["analysis_results"]:
            img_data = st.session_state["analysis_results"][-1]["image"]
            st.download_button("Télécharger la carte", img_data, "carte.png", "image/png")
    elif button_name == "Dessin automatique":
        st.markdown("### Dessin automatique")
        st.info("Cette fonctionnalité n'est pas encore implémentée.")

# ============================
# CRÉATION D'UNE CARTE UNIFIÉE
# ============================
def create_unified_map():
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
    # Ajout des couches dessinées par l'utilisateur
    if st.session_state["layers"]:
        for layer_name, features in st.session_state["layers"].items():
            layer_group = folium.FeatureGroup(name=layer_name, show=True)
            for feature in features:
                feature_type = feature["geometry"]["type"]
                coordinates = feature["geometry"]["coordinates"]
                popup = feature.get("properties", {}).get("name", f"{layer_name} - Entité")
                if feature_type == "Point":
                    lat, lon = coordinates[1], coordinates[0]
                    folium.Marker(location=[lat, lon], popup=popup).add_to(layer_group)
                elif feature_type == "LineString":
                    folium.PolyLine(locations=[(lat, lon) for lon, lat in coordinates], color="blue", popup=popup).add_to(layer_group)
                elif feature_type == "Polygon":
                    folium.Polygon(locations=[(lat, lon) for lon, lat in coordinates[0]], color="green", fill=True, popup=popup).add_to(layer_group)
            layer_group.add_to(m)
    # Ajout des couches téléversées
    if st.session_state["uploaded_layers"]:
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
                    style_function=lambda x, color=color: {"color": color, "weight": 4, "opacity": 0.7}
                ).add_to(m)
    # Ajout des entités dessinées temporairement
    if st.session_state["new_features"]:
        temp_group = folium.FeatureGroup(name="Nouvelles entités", show=True)
        for feature in st.session_state["new_features"]:
            f_type = feature["geometry"]["type"]
            coords = feature["geometry"]["coordinates"]
            if f_type == "Point":
                lat, lon = coords[1], coords[0]
                folium.Marker(location=[lat, lon]).add_to(temp_group)
            elif f_type == "LineString":
                folium.PolyLine(locations=[(lat, lon) for lon, lat in coords], color="blue").add_to(temp_group)
            elif f_type == "Polygon":
                folium.Polygon(locations=[(lat, lon) for lon, lat in coords[0]], color="green", fill=True).add_to(temp_group)
        temp_group.add_to(m)
    # Ajout du contrôle de dessin
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
    return m

# ============================
# RAPPORT (PDF)
# ============================
def run_report():
    st.title("📄 Génération de Rapport")
    with st.sidebar:
        st.header("📝 Métadonnées du Rapport")
        titre = st.text_input("Titre principal", key="rapport_titre")
        report_id = st.text_input("ID du rapport", key="rapport_id")
        report_date = st.date_input("Date du rapport", date.today(), key="rapport_date")
        report_time = st.time_input("Heure du rapport", datetime.now().time(), key="rapport_time")
        editor = st.text_input("Éditeur", key="rapport_editor")
        location = st.text_input("Localisation", key="rapport_location")
        company = st.text_input("Société", key="rapport_company")
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
    st.markdown("### 📌 Ajouter une carte d'analyse spatiale")
    analysis_card = create_analysis_card_controller()
    if analysis_card:
        if not any(el.get("analysis_ref") == analysis_card.get("analysis_ref") for el in elements if el["type"] == "Image"):
            elements.append(analysis_card)
            st.success("Carte d'analyse ajoutée avec succès !")
    st.markdown("### Ajouter d'autres éléments")
    new_element = create_element_controller()
    if new_element:
        elements.append(new_element)
        st.session_state["elements"] = elements
        st.success("Élément validé avec succès !")
    if elements:
        display_elements_preview(elements)
    if elements and st.button("Générer le PDF", key="generate_pdf"):
        pdf = generate_pdf(elements, metadata)
        st.success("✅ Rapport généré avec succès!")
        st.download_button("Télécharger le PDF", pdf, "rapport_structuré.pdf", "application/pdf")

# ============================
# APPLICATION PRINCIPALE UNIFIÉE
# ============================
def main():
    st.set_page_config(page_title="Application d'Analyse Spatiale", layout="wide")
    st.title("Application d'Analyse Spatiale et Topographique")
    st.markdown("""
Cette application centralise toutes les fonctionnalités d’analyse spatiale et topographique.  
Vous pouvez téléverser des fichiers (TIFF, GeoJSON), gérer vos couches, dessiner des entités sur la carte et réaliser des analyses (contours, profils, volumes/surfaces).  
Vous pourrez ensuite générer un rapport PDF regroupant vos analyses.
    """)
    # Barre latérale : Gestion des couches et téléversements
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
                entity_idx = st.selectbox("Sélectionnez une entité à gérer", range(len(st.session_state["layers"][selected_layer])), 
                                            format_func=lambda idx: f"Entité {idx + 1}: {st.session_state['layers'][selected_layer][idx]['geometry']['type']}")
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
        st.markdown("### 2- Téléverser des fichiers")
        tiff_type = st.selectbox("Sélectionnez le type de fichier TIFF", options=["MNT", "MNS", "Orthophoto"],
                                 index=None, placeholder="Veuillez sélectionner", key="tiff_selectbox")
        if tiff_type:
            uploaded_tiff = st.file_uploader(f"Téléverser un fichier TIFF ({tiff_type})", type=["tif", "tiff"], key="tiff_uploader")
            if uploaded_tiff:
                unique_id = str(uuid.uuid4())[:8]
                tiff_path = f"uploaded_{unique_id}.tiff"
                with open(tiff_path, "wb") as f:
                    f.write(uploaded_tiff.read())
                st.write(f"Reprojection du fichier TIFF ({tiff_type})...")
                try:
                    reprojected_tiff = reproject_tiff(tiff_path, "EPSG:4326")
                    with rasterio.open(reprojected_tiff) as src:
                        bounds = src.bounds
                        if not any(layer["name"] == tiff_type and layer["type"] == "TIFF" for layer in st.session_state["uploaded_layers"]):
                            st.session_state["uploaded_layers"].append({"type": "TIFF", "name": tiff_type, "path": reprojected_tiff, "bounds": bounds})
                            st.success(f"Couche {tiff_type} ajoutée à la liste des couches.")
                        else:
                            st.warning(f"La couche {tiff_type} existe déjà.")
                except Exception as e:
                    st.error(f"Erreur lors de la reprojection : {e}")
                finally:
                    os.remove(tiff_path)
        geojson_type = st.selectbox("Sélectionnez le type de fichier GeoJSON", 
                                     options=["Polygonale", "Routes", "Cours d'eau", "Bâtiments", "Pistes", "Plantations",
                                              "Électricité", "Assainissements", "Villages", "Villes", "Chemin de fer", "Parc et réserves"],
                                     index=None, placeholder="Veuillez sélectionner", key="geojson_selectbox")
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
    
    # Onglets unifiés pour l'analyse et le rapport
    tabs = st.tabs(["Analyse", "Rapport"])
    
    with tabs[0]:
        st.subheader("Carte et Analyse Spatiale")
        unified_map = create_unified_map()
        output = st_folium(unified_map, width=800, height=600, returned_objects=["last_active_drawing", "all_drawings"])
        if output and "last_active_drawing" in output and output["last_active_drawing"]:
            new_feature = output["last_active_drawing"]
            if new_feature not in st.session_state["new_features"]:
                st.session_state["new_features"].append(new_feature)
                st.info("Nouvelle entité ajoutée temporairement. Cliquez sur 'Enregistrer les entités' dans la barre latérale pour les ajouter à une couche.")
        st.markdown("### Outils d'analyse")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Surfaces et volumes", key="surfaces_volumes"):
                st.session_state['active_button'] = "Surfaces et volumes"
            if st.button("Carte de contours", key="contours"):
                st.session_state['active_button'] = "Carte de contours"
        with col2:
            if st.button("Tracer des profils", key="tracer_profils"):
                st.session_state['active_button'] = "Tracer des profils"
            if st.button("Trouver un point", key="trouver_point"):
                st.session_state['active_button'] = "Trouver un point"
        with col3:
            if st.button("Télécharger la carte", key="telecharger_carte"):
                st.session_state['active_button'] = "Télécharger la carte"
            if st.button("Dessin automatique", key="dessin_auto"):
                st.session_state['active_button'] = "Dessin automatique"
        parameters_placeholder = st.empty()
        if st.session_state.get('active_button'):
            with parameters_placeholder.container():
                display_parameters(st.session_state['active_button'])
    
    with tabs[1]:
        run_report()

if __name__ == "__main__":
    main()

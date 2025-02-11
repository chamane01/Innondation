import os
import glob
import rasterio
import numpy as np
import folium
import branca.colormap as cm
from rasterio.warp import calculate_default_transform, reproject, Resampling
from folium.raster_layers import ImageOverlay

# Dossiers
TIFF_DIRECTORY = "TIFF"
OUTPUT_DIRECTORY = "TIFF_FIXED"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# Systèmes de coordonnées désirés
TARGET_CRS = "EPSG:4326"  # WGS 84 (latitude/longitude)

# Liste des fichiers TIFF
original_tiff_files = glob.glob(os.path.join(TIFF_DIRECTORY, "*.tif"))
reprojected_files = []

def shorten_filename(filepath):
    """Réduit la longueur des noms de fichiers trop longs."""
    base_name = os.path.basename(filepath)
    parts = base_name.split("_")
    if len(parts) > 3:
        new_name = "_".join(parts[:2]) + "_" + parts[-1]
    else:
        new_name = base_name
    return os.path.join(OUTPUT_DIRECTORY, new_name)

def reproject_tiff(input_tiff, output_tiff, target_crs):
    """Reprojette un fichier TIFF uniquement s'il n'est pas déjà dans le bon CRS."""
    with rasterio.open(input_tiff) as src:
        if src.crs == target_crs:
            print(f"Pas de reprojection nécessaire pour {input_tiff}")
            return input_tiff  # Retourne l'original si pas besoin de reprojection

        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({"crs": target_crs, "transform": transform, "width": width, "height": height})

        with rasterio.open(output_tiff, "w", **kwargs) as dst:
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
    return output_tiff

# Traitement des fichiers TIFF
for tiff_file in original_tiff_files:
    output_tiff = shorten_filename(tiff_file)
    final_tiff = reproject_tiff(tiff_file, output_tiff, TARGET_CRS)
    reprojected_files.append(final_tiff)

# Création d'une carte Folium
m = folium.Map(location=[5.35, -4.03], zoom_start=12)

# Dégradé de couleur
color_map = cm.linear.YlGnBu_09.scale(0, 255)

# Ajout des TIFF sur la carte
for tiff_file in reprojected_files:
    with rasterio.open(tiff_file) as src:
        img_array = src.read(1)
        img_array = np.nan_to_num(img_array)  # Remplace les NaN
        img_min, img_max = np.min(img_array), np.max(img_array)
        if img_max == img_min:
            continue  # Évite d'afficher une image uniforme

        img_normalized = (img_array - img_min) / (img_max - img_min)
        img_colored = color_map(img_normalized)[:, :, :3]  # Enlever alpha

        bounds = [[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]]
        ImageOverlay(image=img_colored, bounds=bounds, opacity=0.7).add_to(m)

# Sauvegarde et affichage
m.save("map.html")
print("Carte générée : ouvrez 'map.html' dans un navigateur.")

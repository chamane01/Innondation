import streamlit as st
import os
import rasterio
import folium
import numpy as np
import matplotlib.pyplot as plt
from rasterio.warp import calculate_default_transform, reproject, Resampling
from streamlit_folium import folium_static
from matplotlib.colors import LinearSegmentedColormap

# Fonction pour vérifier si un fichier a déjà été reprojeté
def is_already_reprojected(file_path):
    return os.path.exists(file_path)

# Fonction pour reprojeter un fichier TIFF
def reproject_tiff(input_path, output_path, dst_crs):
    """Reprojette un fichier TIFF vers le système de coordonnées spécifié."""
    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )

# Fonction pour appliquer un dégradé de couleur à un fichier TIFF
def apply_colormap(tiff_path):
    """Applique un dégradé de couleur à un fichier TIFF."""
    with rasterio.open(tiff_path) as src:
        data = src.read(1)
        data[data == src.nodata] = np.nan  # Gérer les valeurs nodata

        # Créer un dégradé de couleur
        cmap = LinearSegmentedColormap.from_list("elevation", ["green", "yellow", "red"])
        norm_data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))  # Normalisation
        colored_data = cmap(norm_data)

        # Sauvegarder l'image colorée
        output_path = tiff_path.replace(".tif", "_colored.png")
        plt.imsave(output_path, colored_data)
        return output_path

# Fonction pour charger les fichiers TIFF
def load_tiff_files(folder_path):
    """Charge et reprojette les fichiers TIFF contenus dans un dossier."""
    tiff_files = [f for f in os.listdir(folder_path) if f.endswith('.tif') and not f.startswith('reproj_')]
    reproj_files = []
    
    for file in tiff_files:
        input_path = os.path.join(folder_path, file)
        output_path_4326 = os.path.join(folder_path, f"reproj_{file}")
        
        # Reprojeter uniquement si le fichier n'a pas déjà été reprojeté
        if not is_already_reprojected(output_path_4326):
            reproject_tiff(input_path, output_path_4326, 'EPSG:4326')
        
        # Appliquer un dégradé de couleur
        colored_path = apply_colormap(output_path_4326)
        reproj_files.append(colored_path)
    
    return reproj_files

# Fonction pour créer la carte
def create_map(tiff_files):
    """Crée une carte avec une couche OSM et les fichiers TIFF reprojectés."""
    m = folium.Map(location=[0, 0], zoom_start=2)
    
    # Créer une couche commune "Élévation"
    elevation_layer = folium.FeatureGroup(name="Élévation")
    
    for tiff in tiff_files:
        with rasterio.open(tiff.replace("_colored.png", ".tif")) as src:
            bounds = src.bounds
            folium.raster_layers.ImageOverlay(
                image=tiff,
                bounds=[
                    [bounds.bottom, bounds.left],
                    [bounds.top, bounds.right]
                ],
                opacity=0.6,
                name="Élévation"
            ).add_to(elevation_layer)
    
    elevation_layer.add_to(m)
    folium.LayerControl().add_to(m)
    
    return m

# Fonction principale
def main():
    st.title("Carte Dynamique avec Données d'Élévation")
    folder_path = "TIFF"  # Modifier selon l'emplacement réel du dossier
    
    if not os.path.exists(folder_path):
        st.error("Le dossier TIFF n'existe pas.")
        return
    
    st.write("Chargement et reprojection des fichiers TIFF...")
    reproj_files = load_tiff_files(folder_path)
    
    if not reproj_files:
        st.warning("Aucun fichier TIFF trouvé.")
        return
    
    st.write("Création de la carte...")
    map_object = create_map(reproj_files)
    folium_static(map_object)

if __name__ == "__main__":
    main()

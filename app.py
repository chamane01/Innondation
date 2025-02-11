import streamlit as st
import os
import rasterio
import folium
import numpy as np
import matplotlib.pyplot as plt
from rasterio.warp import calculate_default_transform, reproject, Resampling
from streamlit_folium import folium_static

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

def load_tiff_files(folder_path):
    """Charge et reprojette les fichiers TIFF contenus dans un dossier."""
    tiff_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    reproj_files = []
    
    for file in tiff_files:
        input_path = os.path.join(folder_path, file)
        output_path_4326 = os.path.join(folder_path, f"reproj_{file}")
        reproject_tiff(input_path, output_path_4326, 'EPSG:4326')
        reproj_files.append(output_path_4326)
    
    return reproj_files

def apply_color_gradient(tiff_path):
    """Applique un dégradé de couleur aux données TIFF."""
    with rasterio.open(tiff_path) as src:
        data = src.read(1)
        masked_data = np.ma.masked_where(data == src.nodata, data)
        
        # Créer une figure matplotlib avec un dégradé de couleur
        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.imshow(masked_data, cmap='viridis')
        fig.colorbar(cax, orientation='vertical')
        plt.axis('off')
        
        # Sauvegarder l'image en mémoire
        from io import BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close()
        
        return buf

def create_map(tiff_files):
    """Crée une carte avec une couche OSM et les fichiers TIFF reprojectés."""
    m = folium.Map(location=[0, 0], zoom_start=2)
    
    for tiff in tiff_files:
        with rasterio.open(tiff) as src:
            bounds = src.bounds
            color_gradient_image = apply_color_gradient(tiff)
            
            folium.raster_layers.ImageOverlay(
                image=color_gradient_image,
                bounds=[
                    [bounds.bottom, bounds.left],
                    [bounds.top, bounds.right]
                ],
                opacity=0.6,
                interactive=True,
                cross_origin=False
            ).add_to(m)
    
    return m

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

import streamlit as st
import os
import rasterio
import folium
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
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

def create_map(tiff_files):
    """Crée une carte avec les fichiers TIFF affichés en dégradé de couleur."""
    m = folium.Map(location=[0, 0], zoom_start=2)
    
    for tiff in tiff_files:
        with rasterio.open(tiff) as src:
            data = src.read(1)
            
            # Gérer les valeurs nodata
            if src.nodata is not None:
                data = np.ma.masked_where(data == src.nodata, data)
            
            # Normaliser les données et appliquer un colormap
            norm = plt.Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))
            cmap = plt.cm.viridis  # Choix du dégradé de couleur
            image_data = cmap(norm(data))
            
            # Convertir en image PNG
            image_data_uint8 = (image_data * 255).astype(np.uint8)
            img = Image.fromarray(image_data_uint8)
            
            # Sauvegarder l'image dans un buffer
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            # Définir les bornes de l'image
            bounds = [
                [src.bounds.bottom, src.bounds.left],
                [src.bounds.top, src.bounds.right]
            ]
            
            # Ajouter l'image à la carte Folium
            img_overlay = folium.raster_layers.ImageOverlay(
                name=os.path.basename(tiff),
                image=f'data:image/png;base64,{img_base64}',
                bounds=bounds,
                opacity=0.6,
                interactive=True,
                cross_origin=False
            )
            img_overlay.add_to(m)
    
    folium.LayerControl().add_to(m)
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

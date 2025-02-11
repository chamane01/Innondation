import streamlit as st
import os
import rasterio
import folium
import numpy as np
import matplotlib.pyplot as plt
import io
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
    """Crée une carte avec une couche 'Élévation' unifiée."""
    m = folium.Map(location=[0, 0], zoom_start=2)
    
    # Création de la couche d'élévation
    elevation_layer = folium.FeatureGroup(name='Élévation', show=True)
    
    for tiff in tiff_files:
        with rasterio.open(tiff) as src:
            data = src.read(1)
            
            # Gestion des valeurs nodata
            if src.nodata is not None:
                data = np.ma.masked_equal(data, src.nodata)
            
            # Normalisation des valeurs
            valid_data = data[~data.mask] if np.ma.is_masked(data) else data
            min_val, max_val = valid_data.min(), valid_data.max()
            
            if max_val > min_val:
                normalized = (data - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(data, dtype=float)
            
            # Création de l'image colorée
            buffer = io.BytesIO()
            plt.imsave(buffer, normalized, cmap='terrain', format='png', vmin=0, vmax=1)
            buffer.seek(0)
            
            # Définition des limites
            bounds = [[src.bounds.bottom, src.bounds.left], 
                     [src.bounds.top, src.bounds.right]]
            
            # Ajout à la couche d'élévation
            img_overlay = folium.raster_layers.ImageOverlay(
                image=buffer.read(),
                bounds=bounds,
                opacity=0.7,
                interactive=True
            )
            img_overlay.add_to(elevation_layer)
    
    # Ajout des éléments à la carte
    elevation_layer.add_to(m)
    folium.TileLayer('openstreetmap').add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    
    return m

def main():
    st.title("Visualisation des Données d'Élévation")
    folder_path = "TIFF"
    
    if not os.path.exists(folder_path):
        st.error(f"Dossier introuvable : {folder_path}")
        return
    
    st.write("Traitement des fichiers TIFF...")
    reproj_files = load_tiff_files(folder_path)
    
    if not reproj_files:
        st.warning("Aucun fichier TIFF trouvé dans le dossier.")
        return
    
    st.write("Génération de la carte interactive...")
    map_object = create_map(reproj_files)
    folium_static(map_object)

if __name__ == "__main__":
    main()

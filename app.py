import streamlit as st
import os
import rasterio
import folium
from rasterio.warp import calculate_default_transform, reproject, Resampling
from streamlit_folium import folium_static
import numpy as np
import matplotlib.pyplot as plt
from folium.plugins import Draw

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
    """Crée une carte avec une couche OSM et les fichiers TIFF reprojectés."""
    m = folium.Map(location=[0, 0], zoom_start=2)
    
    for tiff in tiff_files:
        with rasterio.open(tiff) as src:
            bounds = src.bounds
            folium.Rectangle(
                bounds=[
                    [bounds.bottom, bounds.left],
                    [bounds.top, bounds.right]
                ],
                color='blue', fill=True, fill_opacity=0.4, tooltip=tiff
            ).add_to(m)
    
    # Ajouter la fonctionnalité de dessin
    Draw(export=True).add_to(m)
    
    return m

def extract_elevation_profile(tiff_files, line_coords):
    """Extrait le profil d'élévation le long d'une ligne."""
    elevations = []
    distances = []
    
    for tiff in tiff_files:
        with rasterio.open(tiff) as src:
            for i in range(len(line_coords) - 1):
                start = line_coords[i]
                end = line_coords[i + 1]
                
                # Interpoler les points le long de la ligne
                num_points = 100
                x = np.linspace(start[1], end[1], num_points)
                y = np.linspace(start[0], end[0], num_points)
                
                for xi, yi in zip(x, y):
                    row, col = src.index(xi, yi)
                    elevation = src.read(1, window=((row, row+1), (col, col+1)))
                    elevations.append(elevation[0][0])
                    distances.append(np.sqrt((xi - start[1])**2 + (yi - start[0])**2))
    
    return distances, elevations

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
    
    # Afficher la carte
    folium_static(map_object)
    
    # Récupérer les coordonnées du trait tracé par l'utilisateur
    if st.button("Extraire le profil d'élévation"):
        if 'last_active_drawing' in st.session_state:
            line_coords = st.session_state['last_active_drawing']['geometry']['coordinates']
            distances, elevations = extract_elevation_profile(reproj_files, line_coords)
            
            # Tracer le profil d'élévation
            plt.figure(figsize=(10, 4))
            plt.plot(distances, elevations, label="Profil d'élévation")
            plt.xlabel("Distance (m)")
            plt.ylabel("Élévation (m)")
            plt.title("Profil d'élévation")
            plt.legend()
            st.pyplot(plt)
        else:
            st.warning("Veuillez tracer un trait sur la carte avant d'extraire le profil d'élévation.")

if __name__ == "__main__":
    main()

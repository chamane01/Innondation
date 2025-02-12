import streamlit as st
import os
import rasterio
import folium
from rasterio.warp import calculate_default_transform, reproject, Resampling
from streamlit_folium import folium_static

def reproject_tiff(input_path, output_path, dst_crs):
    """Reprojette un fichier TIFF vers le système de coordonnées spécifié."""
    try:
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
    except rasterio.errors.RasterioIOError as e:
        st.error(f"Erreur lors de l'ouverture du fichier {input_path}: {e}")
        return None

def load_tiff_files(folder_path):
    """Charge et reprojette les fichiers TIFF contenus dans un dossier."""
    # Supprimer les fichiers reprojetés existants
    for file in os.listdir(folder_path):
        if file.startswith("reproj_") and file.endswith(".tif"):
            os.remove(os.path.join(folder_path, file))
            st.write(f"Fichier supprimé : {file}")

    # Charger uniquement les fichiers originaux (non reprojetés)
    tiff_files = [f for f in os.listdir(folder_path) if f.endswith('.tif') and not f.startswith('reproj_')]
    st.write(f"Fichiers TIFF originaux trouvés : {tiff_files}")  # Debugging statement
    
    reproj_files = []
    
    for file in tiff_files:
        input_path = os.path.join(folder_path, file)
        output_path_4326 = os.path.join(folder_path, f"reproj_{file}")
        if reproject_tiff(input_path, output_path_4326, 'EPSG:4326'):
            reproj_files.append(output_path_4326)
    
    return reproj_files

def create_map(tiff_files):
    """Crée une carte avec une couche OSM et les fichiers TIFF reprojectés."""
    m = folium.Map(location=[0, 0], zoom_start=2)
    
    for tiff in tiff_files:
        try:
            with rasterio.open(tiff) as src:
                bounds = src.bounds
                folium.Rectangle(
                    bounds=[
                        [bounds.bottom, bounds.left],
                        [bounds.top, bounds.right]
                    ],
                    color='blue', fill=True, fill_opacity=0.4, tooltip=tiff
                ).add_to(m)
        except rasterio.errors.RasterioIOError as e:
            st.error(f"Erreur lors de l'ouverture du fichier {tiff}: {e}")
    
    return m

def main():
    st.title("Carte Dynamique avec Données d'Élévation")
    folder_path = "TIFF"  # Modifier selon l'emplacement réel du dossier
    
    if not os.path.exists(folder_path):
        st.error(f"Le dossier TIFF n'existe pas à l'emplacement : {folder_path}")
        return
    
    st.write("Chargement et reprojection des fichiers TIFF...")
    reproj_files = load_tiff_files(folder_path)
    
    if not reproj_files:
        st.warning("Aucun fichier TIFF valide trouvé.")
        return
    
    st.write("Création de la carte...")
    map_object = create_map(reproj_files)
    folium_static(map_object)

if __name__ == "__main__":
    main()

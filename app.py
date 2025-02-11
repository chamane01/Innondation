import streamlit as st
import os
import rasterio
import folium
from rasterio.warp import calculate_default_transform, reproject, Resampling
from streamlit_folium import folium_static
from folium.plugins import Draw
import geopandas as gpd
from shapely.geometry import Polygon

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
    
    # Ajouter des outils de dessin
    Draw(export=True).add_to(m)
    
    return m

def generate_contours(polygon):
    """Génère des contours à partir d'un polygone."""
    # Cette fonction est un exemple, vous pouvez l'adapter pour générer des contours réels
    st.write("Génération des contours pour le polygone sélectionné...")
    st.write(polygon)

def main():
    st.title("Carte Dynamique avec Données d'Élévation")
    
    # Barre latérale à gauche
    st.sidebar.title("Outils d'Analyse Spatiale")
    
    # Menu Streamlit dans la barre latérale
    analysis_tool = st.sidebar.selectbox(
        "Sélectionnez un outil d'analyse",
        ["Aucun", "Générer des contours"]
    )
    
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
    
    # Récupérer les données dessinées sur la carte
    if analysis_tool == "Générer des contours":
        st.sidebar.write("Dessinez un polygone sur la carte pour générer des contours.")
        drawn_data = st.session_state.get("drawn_data", None)
        
        if drawn_data:
            polygon = drawn_data[-1]  # Prendre le dernier polygone dessiné
            generate_contours(polygon)
        else:
            st.sidebar.warning("Veuillez dessiner un polygone sur la carte.")

if __name__ == "__main__":
    main()

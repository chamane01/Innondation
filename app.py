import streamlit as st
import os
import rasterio
import geopandas as gpd
import folium
from rasterio.warp import calculate_default_transform, reproject, Resampling
from streamlit_folium import folium_static
from pyproj import CRS

# Définir un dossier de travail (changer selon l'environnement)
FOLDER_PATH = "TIFF"  # Modifier si besoin
TEMP_FOLDER_PATH = "/tmp/TIFF"  # Pour Streamlit Cloud

# Vérifier si le dossier TIFF est accessible, sinon utiliser un dossier temporaire
if not os.path.exists(FOLDER_PATH):
    st.warning(f"Le dossier {FOLDER_PATH} n'existe pas, utilisation du dossier temporaire.")
    FOLDER_PATH = TEMP_FOLDER_PATH
os.makedirs(FOLDER_PATH, exist_ok=True)  # S'assurer que le dossier existe

def reproject_tiff(input_path, output_path, dst_crs):
    """Reprojette un fichier TIFF vers le système de coordonnées spécifié."""
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Le fichier {input_path} n'existe pas.")

    with rasterio.open(input_path) as src:
        if src.crs is None:
            raise ValueError(f"Le fichier {input_path} n'a pas de CRS défini.")
        
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
    
    if not tiff_files:
        st.warning("Aucun fichier TIFF trouvé dans le dossier.")
        return []
    
    for file in tiff_files:
        input_path = os.path.join(folder_path, file)
        output_path_4326 = os.path.join(folder_path, f"reproj_{file}")
        
        try:
            reproject_tiff(input_path, output_path_4326, 'EPSG:4326')
            reproj_files.append(output_path_4326)
        except Exception as e:
            st.error(f"Erreur lors de la reprojection de {file} : {e}")

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
        except Exception as e:
            st.error(f"Impossible d'afficher {tiff} sur la carte : {e}")
    
    return m

def main():
    st.title("Carte Dynamique avec Données d'Élévation")
    
    # Vérifier les permissions d'écriture dans le dossier
    test_path = os.path.join(FOLDER_PATH, "test.txt")
    try:
        with open(test_path, "w") as f:
            f.write("test")
        os.remove(test_path)
    except Exception as e:
        st.error(f"Problème de permission d'écriture dans {FOLDER_PATH} : {e}")
        return
    
    st.write("Chargement et reprojection des fichiers TIFF...")
    reproj_files = load_tiff_files(FOLDER_PATH)
    
    if not reproj_files:
        st.warning("Aucun fichier TIFF reprojecté disponible.")
        return
    
    st.write("Création de la carte...")
    map_object = create_map(reproj_files)
    folium_static(map_object)

if __name__ == "__main__":
    main()

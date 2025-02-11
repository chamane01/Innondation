import streamlit as st
import os
import rasterio
import folium
from rasterio.warp import calculate_default_transform, reproject, Resampling
from streamlit_folium import folium_static
from rasterio.crs import CRS as RasterioCRS

def is_already_4326(input_path):
    """Vérifie si un fichier est déjà en EPSG:4326."""
    try:
        with rasterio.open(input_path) as src:
            if src.crs is None:
                return False
            return src.crs == RasterioCRS.from_epsg(4326)
    except Exception as e:
        st.error(f"Erreur lors de la vérification du CRS : {str(e)}")
        return False

def reproject_tiff(input_path, output_path, dst_crs):
    """Reprojette un fichier TIFF vers le système de coordonnées spécifié."""
    try:
        dst_crs = RasterioCRS.from_user_input(dst_crs)
        with rasterio.open(input_path) as src:
            if src.crs is None:
                raise ValueError("CRS source non défini")
            
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
            
            kwargs = src.meta.copy()
            kwargs.update({
                'driver': 'GTiff',
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
    except Exception as e:
        st.error(f"Erreur lors de la reprojection : {str(e)}")
        raise

def load_tiff_files(folder_path):
    """Charge et reprojette les fichiers TIFF contenus dans un dossier."""
    try:
        tiff_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
        if not tiff_files:
            return []
        
        # Créer un sous-dossier pour les outputs
        output_dir = os.path.join(folder_path, 'reprojected')
        os.makedirs(output_dir, exist_ok=True)
        
        # Nettoyer les anciens fichiers reprojetés
        for old_file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, old_file))
        
        reproj_files = []
        for file in tiff_files:
            input_path = os.path.join(folder_path, file)
            
            # Vérifier si le fichier est déjà en EPSG:4326
            if is_already_4326(input_path):
                st.info(f"{file} est déjà en EPSG:4326, pas de reprojection nécessaire")
                reproj_files.append(input_path)
                continue
                
            # Générer un nom de fichier court
            output_filename = f"reproj_{file[:20]}.tif"  # Limite à 20 caractères
            output_path_4326 = os.path.join(output_dir, output_filename)
            
            if not os.path.exists(input_path):
                st.error(f"Fichier introuvable : {input_path}")
                continue
                
            try:
                reproject_tiff(input_path, output_path_4326, 'EPSG:4326')
                reproj_files.append(output_path_4326)
            except Exception as e:
                st.error(f"Échec sur {file} : {str(e)}")
                continue
        
        return reproj_files
    except Exception as e:
        st.error(f"Erreur lors du chargement : {str(e)}")
        return []

def create_map(tiff_files):
    """Crée une carte avec les fichiers TIFF."""
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
                    color='blue', 
                    fill=True, 
                    fill_opacity=0.4, 
                    tooltip=os.path.basename(tiff)
                ).add_to(m)
        except Exception as e:
            st.error(f"Erreur avec {tiff} : {str(e)}")
            continue
    
    return m

def main():
    st.title("Carte Dynamique avec Données d'Élévation")
    
    # Chemin relatif corrigé
    folder_path = os.path.join(os.getcwd(), "TIFF")
    
    if not os.path.exists(folder_path):
        st.error("Dossier TIFF introuvable. Structure attendue :")
        st.code(f"""
        {os.getcwd()}/
        ├── app.py
        └── TIFF/
            ├── fichier1.tif
            └── fichier2.tif
        """)
        return
    
    reproj_files = load_tiff_files(folder_path)
    
    if not reproj_files:
        st.warning("Aucun fichier valide traité")
        return
    
    map_object = create_map(reproj_files)
    folium_static(map_object)

if __name__ == "__main__":
    main()

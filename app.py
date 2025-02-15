import streamlit as st 
import os
import rasterio
import rasterio.merge
import rasterio.mask
import folium
import numpy as np
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
from folium.plugins import Draw
from shapely.geometry import shape, mapping  # Pour la fusion des géométries

#########################
# Fonctions utilitaires #
#########################

def load_tiff_files(folder_path):
    """Charge les fichiers TIFF contenus dans un dossier."""
    try:
        tiff_files = [os.path.join(folder_path, f)
                      for f in os.listdir(folder_path) if f.lower().endswith('.tif')]
    except Exception as e:
        st.error(f"Erreur lors de la lecture du dossier {folder_path} : {e}")
        return []
    
    if not tiff_files:
        st.error("Aucun fichier TIFF trouvé dans le dossier.")
        return []
    
    return [f for f in tiff_files if os.path.exists(f)]

def build_mosaic(tiff_files, mosaic_path="mosaic.tif"):
    """Construit une mosaïque à partir de fichiers TIFF."""
    try:
        src_files = [rasterio.open(fp) for fp in tiff_files]
        mosaic, out_trans = rasterio.merge.merge(src_files)
        out_meta = src_files[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans
        })
        with rasterio.open(mosaic_path, "w", **out_meta) as dest:
            dest.write(mosaic)
        for src in src_files:
            src.close()
        return mosaic_path
    except Exception as e:
        st.error(f"Erreur lors de la création de la mosaïque : {e}")
        return None

def create_map(mosaic_file):
    """
    Crée une carte Folium avec :
      - Une couche OSM (50% d'opacité) pour vérifier l'emprise.
      - Un calque indiquant l'emprise de la mosaïque.
      - L'outil de dessin (rectangle) pour sélectionner une zone.
    """
    # Création d'une carte sans fond par défaut
    m = folium.Map(location=[0, 0], zoom_start=2, tiles=None)
    
    # Ajout de la couche OSM avec 50% d'opacité
    folium.TileLayer(
        tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        attr='&copy; OpenStreetMap contributors',
        name='OSM',
        overlay=True,
        control=True,
        opacity=0.5,
        show=True
    ).add_to(m)
    
    # Calque indiquant l'emprise de la mosaïque
    mosaic_group = folium.FeatureGroup(name="Mosaïque")
    try:
        with rasterio.open(mosaic_file) as src:
            bounds = src.bounds
            folium.Rectangle(
                bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                color='blue',
                fill=False,
                tooltip="Emprise de la mosaïque"
            ).add_to(mosaic_group)
    except Exception as e:
        st.error(f"Erreur lors de l'ouverture de la mosaïque : {e}")
    
    mosaic_group.add_to(m)
    
    # Ajout de l'outil de dessin (uniquement pour dessiner des rectangles)
    Draw(
        draw_options={
            'rectangle': True,
            'polyline': False,
            'polygon': False,
            'circle': False,
            'marker': False,
            'circlemarker': False
        },
        edit_options={'edit': True, 'remove': True}
    ).add_to(m)
    
    folium.LayerControl().add_to(m)
    
    return m

def generate_contours(mosaic_file, drawing_geometry=None, title="Contours d'élévation"):
    """
    Génère et affiche les courbes de niveau (contours) à partir du fichier TIFF.
    Si drawing_geometry est fourni (GeoJSON d'une zone dessinée), on ne
    génère les contours que sur cette zone.
    Le titre du graphique est personnalisé via le paramètre title.
    """
    try:
        with rasterio.open(mosaic_file) as src:
            if drawing_geometry is not None:
                # Découper la mosaïque selon l'emprise dessinée
                out_image, out_transform = rasterio.mask.mask(src, [drawing_geometry], crop=True)
                data = out_image[0]
            else:
                data = src.read(1)
                out_transform = src.transform
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier TIFF : {e}")
        return
    
    # Masquer les valeurs nodata
    nodata = src.nodata
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)
    
    # Création d'une grille de coordonnées en se basant sur la transformation affine
    nrows, ncols = data.shape
    x_coords = np.arange(ncols) * out_transform.a + out_transform.c + out_transform.a/2
    y_coords = np.arange(nrows) * out_transform.e + out_transform.f + out_transform.e/2
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Détermination automatique des niveaux de contour
    try:
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
    except Exception as e:
        st.error(f"Erreur lors du calcul des valeurs min et max : {e}")
        return
    
    levels = np.linspace(vmin, vmax, 15)
    
    # Tracé des courbes de niveau
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contour(X, Y, data, levels=levels, cmap='terrain')
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    st.pyplot(fig)

##################
# Fonction main  #
##################

def main():
    st.title("Génération de Contours à partir d'un TIFF")
    
    # Saisie du nom de la carte
    map_name = st.text_input("Nom de votre carte", value="Ma Carte")
    
    # Option pour fusionner les profils dessinés sur la même carte
    merge_profiles = st.checkbox("Fusionner les profils dessinés sur la même carte", value=False)
    
    # Chargement et construction de la mosaïque
    folder_path = "TIFF"
    if not os.path.exists(folder_path):
        st.error("Dossier TIFF introuvable")
        return

    tiff_files = load_tiff_files(folder_path)
    if not tiff_files:
        return

    mosaic_path = build_mosaic(tiff_files)
    if not mosaic_path:
        return

    # Création de la carte interactive avec l'outil de dessin
    m = create_map(mosaic_path)
    st.write("**Utilisez l'outil de dessin pour sélectionner une zone (rectangle) sur la carte.**")
    map_data = st_folium(m, width=700, height=500)
    
    # Récupération des emprises dessinées
    drawing_geometries = []
    if isinstance(map_data, dict):
        raw_drawings = map_data.get("all_drawings", [])
        if raw_drawings:
            for drawing in raw_drawings:
                if isinstance(drawing, dict) and drawing.get("geometry", {}).get("type") == "Polygon":
                    drawing_geometries.append(drawing.get("geometry"))
    
    if not drawing_geometries:
        st.warning("Veuillez dessiner une emprise")
    else:
        # Si fusion des profils est activée, on calcule l'union des géométries
        if merge_profiles:
            union_geom = None
            for geom in drawing_geometries:
                poly = shape(geom)
                if union_geom is None:
                    union_geom = poly
                else:
                    union_geom = union_geom.union(poly)
            union_geojson = mapping(union_geom)
            st.subheader(f"Contours fusionnés pour la carte : {map_name}")
            generate_contours(mosaic_path, union_geojson, title=f"Contours fusionnés pour {map_name}")
        else:
            # Sinon, afficher les contours pour chaque emprise dessinée séparément
            for i, geom in enumerate(drawing_geometries, start=1):
                st.subheader(f"Contours d'élévation - Emprise {i} pour {map_name}")
                generate_contours(mosaic_path, geom, title=f"Contours d'élévation - Emprise {i} pour {map_name}")
    
if __name__ == "__main__":
    main()

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
    Crée une carte Folium avec l'outil de dessin (rectangle pour sélectionner une emprise)
    et intègre un calque indiquant l'emprise de la mosaïque.
    """
    m = folium.Map(location=[0, 0], zoom_start=2)
    
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

def generate_contours(mosaic_file, drawing_geometry=None):
    """
    Génère et affiche les courbes de niveau (contours) à partir du fichier TIFF.
    Si drawing_geometry est fourni (GeoJSON d'un rectangle dessiné), on ne
    génère les contours que sur cette zone.
    """
    try:
        with rasterio.open(mosaic_file) as src:
            if drawing_geometry is not None:
                # Découpe de la mosaïque à partir de l'emprise dessinée
                out_image, out_transform = rasterio.mask.mask(src, [drawing_geometry], crop=True)
                data = out_image[0]
            else:
                data = src.read(1)
                out_transform = src.transform
            nodata = src.nodata
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier TIFF : {e}")
        return
    
    # Masquer les valeurs nodata
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)
    
    nrows, ncols = data.shape
    # Calcul des coordonnées des centres de pixels
    x_coords = np.arange(ncols) * out_transform.a + out_transform.c + out_transform.a/2
    y_coords = np.arange(nrows) * out_transform.e + out_transform.f + out_transform.e/2
    X, Y = np.meshgrid(x_coords, y_coords)
    
    try:
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
    except Exception as e:
        st.error(f"Erreur lors du calcul des valeurs min et max : {e}")
        return
    
    levels = np.linspace(vmin, vmax, 15)
    
    # Tracé des courbes de niveau avec légende d'élévation
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contour(X, Y, data, levels=levels, cmap='terrain')
    ax.clabel(contour, inline=True, fontsize=8)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Élévation")
    ax.set_title("Contours d'élévation")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    st.pyplot(fig)

##################
# Fonction main  #
##################

def main():
    st.title("Génération de Contours à partir d'un TIFF")
    
    # Saisie du nom de la carte (affiché en titre, par exemple)
    map_name = st.text_input("Nom de votre carte", value="Ma Carte")
    
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

    # Création de la carte interactive avec outil de dessin pour sélectionner une emprise
    m = create_map(mosaic_path)
    st.write("**Utilisez l'outil de dessin pour sélectionner une zone (rectangle) sur la carte.**")
    map_data = st_folium(m, width=700, height=500)
    
    # Récupération sécurisée des dessins (pour éviter une erreur si aucun dessin n'est présent)
    raw_drawings = map_data.get("all_drawings")
    if raw_drawings is None:
        raw_drawings = []

    # Si des rectangles ont été dessinés, on génère un graphique de contours par zone
    if raw_drawings:
        for i, drawing in enumerate(raw_drawings):
            if isinstance(drawing, dict) and drawing.get("geometry", {}).get("type") == "Polygon":
                st.subheader(f"Contours pour la zone {i+1}")
                generate_contours(mosaic_path, drawing.get("geometry"))
    else:
        st.info("Aucune zone dessinée, génération des contours pour la mosaïque complète.")
        st.subheader("Contours pour la mosaïque complète")
        generate_contours(mosaic_path, None)
    
if __name__ == "__main__":
    main()

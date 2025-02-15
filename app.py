import streamlit as st
import os
import rasterio
import rasterio.merge
from rasterio.transform import rowcol
from rasterio.windows import Window
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
        tiff_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith('.tif')
        ]
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
    
    # Ajout de l'outil de dessin (seulement pour dessiner des rectangles)
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

def generate_contour_figure(mosaic_file, drawing_geometry=None, title="Contours d'élévation"):
    """
    Génère une figure matplotlib avec les courbes de niveau (contours) à partir du fichier TIFF.
    Si drawing_geometry est fourni (un rectangle dessiné), la zone est découpée en utilisant sa bounding box.
    """
    try:
        with rasterio.open(mosaic_file) as src:
            if drawing_geometry is not None:
                # Récupération de la bounding box du rectangle dessiné
                coords = drawing_geometry.get("coordinates", [[]])[0]
                xs = [pt[0] for pt in coords]
                ys = [pt[1] for pt in coords]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                
                # Conversion des coordonnées géospatiales en indices de pixels
                # (rowcol renvoie row puis col)
                row_min, col_min = rowcol(src.transform, min_x, max_y)
                row_max, col_max = rowcol(src.transform, max_x, min_y)
                # S'assurer que les indices sont dans l'ordre (top-left et bottom-right)
                if row_min > row_max:
                    row_min, row_max = row_max, row_min
                if col_min > col_max:
                    col_min, col_max = col_max, col_min
                
                width = col_max - col_min + 1
                height = row_max - row_min + 1
                window = Window(col_off=col_min, row_off=row_min, width=width, height=height)
                data = src.read(1, window=window)
                new_transform = rasterio.windows.transform(window, src.transform)
            else:
                data = src.read(1)
                new_transform = src.transform
            nodata = src.nodata
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier TIFF : {e}")
        return None
    
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)
    
    # Création d'une grille de coordonnées en se basant sur la transformation affine
    nrows, ncols = data.shape
    x_coords = np.arange(ncols) * new_transform.a + new_transform.c + new_transform.a / 2
    y_coords = np.arange(nrows) * new_transform.e + new_transform.f + new_transform.e / 2
    X, Y = np.meshgrid(x_coords, y_coords)
    
    try:
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
    except Exception as e:
        st.error(f"Erreur lors du calcul des valeurs min et max : {e}")
        return None
    
    levels = np.linspace(vmin, vmax, 15)
    
    # Tracé des courbes de niveau avec légende (colorbar)
    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(X, Y, data, levels=levels, cmap='terrain', alpha=0.8)
    cs = ax.contour(X, Y, data, levels=levels, colors='black', linewidths=0.5)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.0f")
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(cf, ax=ax, label="Élévation (m)")
    
    return fig

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

    # Création de la carte interactive avec outil de dessin pour sélectionner l'emprise
    m = create_map(mosaic_path)
    st.write("**Utilisez l'outil de dessin pour sélectionner une ou plusieurs zones (rectangles) sur la carte.**")
    map_data = st_folium(m, width=700, height=500)
    
    # Extraction des zones dessinées (vérifier que raw_drawings est une liste)
    drawing_geometries = []
    if isinstance(map_data, dict):
        raw_drawings = map_data.get("all_drawings")
        if raw_drawings is None or not isinstance(raw_drawings, list):
            raw_drawings = []
        for drawing in raw_drawings:
            if isinstance(drawing, dict) and drawing.get("geometry", {}).get("type") == "Polygon":
                drawing_geometries.append(drawing.get("geometry"))
    
    st.subheader("Résultat des contours")
    if drawing_geometries:
        # Pour chaque zone dessinée, générer une carte de contours
        for i, geom in enumerate(drawing_geometries):
            fig = generate_contour_figure(
                mosaic_path, 
                drawing_geometry=geom, 
                title=f"{map_name} - Contours zone {i+1}"
            )
            if fig:
                st.pyplot(fig)
    else:
        # Si aucune zone n'est dessinée, générer les contours sur l'ensemble de la mosaïque
        fig = generate_contour_figure(
            mosaic_path, 
            drawing_geometry=None, 
            title=f"{map_name} - Contours (ensemble de la mosaïque)"
        )
        if fig:
            st.pyplot(fig)

if __name__ == "__main__":
    main()

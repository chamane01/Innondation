import streamlit as st
import rasterio
import rasterio.warp
import folium
from folium.plugins import MeasureControl, Draw
from rasterio.plot import reshape_as_image
import numpy as np
from sklearn.cluster import DBSCAN
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import os


# Reprojection function
def reproject_tiff(input_tiff, target_crs):
    with rasterio.open(input_tiff) as src:
        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height,
        })

        reprojected_tiff = "reprojected.tiff"
        with rasterio.open(reprojected_tiff, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=rasterio.warp.Resampling.nearest,
                )
    return reprojected_tiff

# Function to apply color gradient to a DEM TIFF
def apply_color_gradient(tiff_path, output_path):
    with rasterio.open(tiff_path) as src:
        # Read the DEM data
        dem_data = src.read(1)
        
        # Create a color map using matplotlib
        cmap = plt.get_cmap("terrain")
        norm = plt.Normalize(vmin=dem_data.min(), vmax=dem_data.max())
        
        # Apply the colormap
        colored_image = cmap(norm(dem_data))
        
        # Save the colored image as PNG
        plt.imsave(output_path, colored_image)
        plt.close()

# Overlay function for TIFF images
def add_image_overlay(map_object, tiff_path, bounds, name):
    with rasterio.open(tiff_path) as src:
        image = reshape_as_image(src.read())
        folium.raster_layers.ImageOverlay(
            image=image,
            bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
            name=name,
            opacity=0.6,
        ).add_to(map_object)

# Function to count trees based on MNT and MNS
def analyse_arbre(mns_data, mnt_data):
    """Fonction pour l'analyse DBSCAN et le comptage des arbres"""
    # Calcul de la différence MNS - MNT pour obtenir les hauteurs relatives
    difference = mns_data - mnt_data

    # Appliquer DBSCAN pour identifier les clusters (arbres)
    dbscan = DBSCAN(eps=0.5, min_samples=10)
    clusters = dbscan.fit_predict(difference.reshape(-1, 1))  # A adapter selon les données

    # Compter les arbres (clusters distincts)
    unique_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)  # exclure le bruit (-1)
    return unique_clusters

# Main application
def main():
    st.title("DESSINER une CARTE ")

    # Initialiser les variables de données
    mnt_data = None
    mns_data = None

    # Initialiser la carte
    fmap = folium.Map(location=[0, 0], zoom_start=2)
    fmap.add_child(MeasureControl(position="topleft"))
    draw = Draw(
        position="topleft",
        export=True,
        draw_options={
            "polyline": {"shapeOptions": {"color": "orange", "weight": 4, "opacity": 0.7}},
            "polygon": {"shapeOptions": {"color": "green", "weight": 4, "opacity": 0.7}},
            "rectangle": {"shapeOptions": {"color": "red", "weight": 4, "opacity": 0.7}},
            "circle": {"shapeOptions": {"color": "purple", "weight": 4, "opacity": 0.7}},
        },
        edit_options={"edit": True},
    )
    fmap.add_child(draw)

    # Téléversement du fichier MNT (Modèle Numérique de Terrain)
    uploaded_mnt = st.file_uploader("Téléverser un fichier MNT (TIFF)", type=["tif", "tiff"])
    if uploaded_mnt:
        mnt_path = uploaded_mnt.name
        with open(mnt_path, "wb") as f:
            f.write(uploaded_mnt.read())

        st.write("Reprojection du fichier MNT...")
        try:
            mnt_data = load_tiff_data(mnt_path)
            # Traitement de l'image colorée du MNT
            temp_png_path = "mnt_colored.png"
            apply_color_gradient(mnt_path, temp_png_path)
            with rasterio.open(mnt_path) as src:
                bounds = src.bounds
                add_image_overlay(fmap, temp_png_path, bounds, "MNT")
            os.remove(temp_png_path)
        except Exception as e:
            st.error(f"Erreur lors de la reprojection du MNT : {e}")

    # Téléversement du fichier MNS (Modèle Numérique de Surface)
    uploaded_mns = st.file_uploader("Téléverser un fichier MNS (TIFF)", type=["tif", "tiff"])
    if uploaded_mns:
        mns_path = uploaded_mns.name
        with open(mns_path, "wb") as f:
            f.write(uploaded_mns.read())

        st.write("Reprojection du fichier MNS...")
        try:
            mns_data = load_tiff_data(mns_path)
            # Traitement de l'image colorée du MNS
            temp_png_path = "mns_colored.png"
            apply_color_gradient(mns_path, temp_png_path)
            with rasterio.open(mns_path) as src:
                bounds = src.bounds
                add_image_overlay(fmap, temp_png_path, bounds, "MNS")
            os.remove(temp_png_path)
        except Exception as e:
            st.error(f"Erreur lors de la reprojection du MNS : {e}")

    # Ajout d'un bouton pour compter les arbres
    if mnt_data is not None and mns_data is not None:
        if st.button("Compter arbres"):
            st.write("Analyse des données MNS - MNT en cours...")
            try:
                arbres_count = analyse_arbre(mns_data, mnt_data)
                st.write(f"Le nombre d'arbres détectés est : {arbres_count}")
            except Exception as e:
                st.error(f"Erreur lors du comptage des arbres : {e}")

    # Affichage de la carte
    folium_static(fmap, width=700, height=500)

# Fonction pour charger les données TIFF et les convertir en numpy array
def load_tiff_data(tiff_path):
    with rasterio.open(tiff_path) as src:
        data = src.read(1)  # Lire la première bande
        data = data.astype(np.float32)  # Assurer que les données sont de type float32
        return data

if __name__ == "__main__":
    main()

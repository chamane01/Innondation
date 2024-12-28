import streamlit as st
import rasterio
import numpy as np
import folium
from folium import plugins
from sklearn.cluster import DBSCAN
from rasterio.warp import calculate_default_transform, reproject, Resampling
from streamlit_folium import folium_static

# Fonction pour reprojeter les fichiers TIFF en WGS84
def reproject_to_wgs84(src_path, dst_path):
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": "EPSG:4326",
            "transform": transform,
            "width": width,
            "height": height,
        })

        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs="EPSG:4326",
                    resampling=Resampling.nearest,
                )
    return dst_path

# Charger un fichier TIFF et ses métadonnées
def load_tiff(file):
    with rasterio.open(file) as src:
        data = src.read(1)
        bounds = src.bounds
        transform = src.transform
    return data, bounds, transform

# Calcul des hauteurs à partir des données MNT et MNS
def calculate_heights(mnt, mns):
    return mns - mnt

# Détection d'arbres avec DBSCAN
def detect_trees(heights, height_threshold, eps, min_samples):
    coords = np.column_stack(np.where(heights > height_threshold))
    if coords.size == 0:
        return coords, np.array([])
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    return coords, db.labels_

# Afficher la carte interactive
def display_map(mnt_data, mns_data, bounds, coords=None, labels=None):
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    # Overlay pour MNT
    folium.raster_layers.ImageOverlay(
        image=mnt_data,
        bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
        opacity=0.5,
        name="MNT"
    ).add_to(fmap)

    # Overlay pour MNS
    folium.raster_layers.ImageOverlay(
        image=mns_data,
        bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
        opacity=0.5,
        name="MNS"
    ).add_to(fmap)

    # Ajouter les points détectés
    if coords is not None and labels is not None:
        for (x, y), label in zip(coords, labels):
            if label != -1:
                folium.CircleMarker(
                    location=[x, y],
                    radius=2,
                    color="red",
                    fill=True
                ).add_to(fmap)

    folium.LayerControl().add_to(fmap)
    folium_static(fmap, width=800, height=600)

# Application principale
def main():
    st.title("Analyse de données MNT et MNS")
    
    # Téléversement des fichiers
    uploaded_mnt = st.file_uploader("Téléversez un fichier MNT (TIFF)", type=["tif", "tiff"])
    uploaded_mns = st.file_uploader("Téléversez un fichier MNS (TIFF)", type=["tif", "tiff"])
    
    # Stocker les fichiers dans session_state
    if uploaded_mnt and uploaded_mns:
        st.session_state["mnt_file"] = uploaded_mnt
        st.session_state["mns_file"] = uploaded_mns
        st.success("Fichiers téléchargés et stockés dans la session.")

    # Initialiser les paramètres de la détection
    if "height_threshold" not in st.session_state:
        st.session_state["height_threshold"] = 0.1
    if "eps" not in st.session_state:
        st.session_state["eps"] = 3
    if "min_samples" not in st.session_state:
        st.session_state["min_samples"] = 3

    # Vérification des fichiers dans la session
    if "mnt_file" in st.session_state and "mns_file" in st.session_state:
        mnt_file = st.session_state["mnt_file"]
        mns_file = st.session_state["mns_file"]

        # Reprojection vers WGS84
        mnt_reprojected = reproject_to_wgs84(mnt_file.name, "mnt_wgs84.tif")
        mns_reprojected = reproject_to_wgs84(mns_file.name, "mns_wgs84.tif")

        # Chargement des fichiers reprojectés
        mnt, mnt_bounds, _ = load_tiff(mnt_reprojected)
        mns, _, _ = load_tiff(mns_reprojected)

        # Calcul des hauteurs
        heights = calculate_heights(mnt, mns)

        # Affichage de la carte initiale
        st.subheader("Carte initiale avec MNT et MNS")
        display_map(mnt, mns, mnt_bounds)

        # Ajouter le bouton pour lancer la détection dans la sidebar
        with st.sidebar:
            st.subheader("Paramètres de détection")
            st.session_state["height_threshold"] = st.slider(
                "Seuil de hauteur (m)", 0.1, 20.0, st.session_state["height_threshold"], step=0.1
            )
            st.session_state["eps"] = st.slider(
                "Rayon de voisinage (m)", 1, 10, st.session_state["eps"]
            )
            st.session_state["min_samples"] = st.slider(
                "Nombre minimum de points par cluster", 1, 10, st.session_state["min_samples"]
            )
            
            if st.button("Lancer la détection"):
                coords, labels = detect_trees(
                    heights, 
                    st.session_state["height_threshold"], 
                    st.session_state["eps"], 
                    st.session_state["min_samples"]
                )
                if coords.size == 0:
                    st.warning("Aucun arbre détecté.")
                else:
                    st.success(f"{len(set(labels)) - (1 if -1 in labels else 0)} arbres détectés.")
                    display_map(mnt, mns, mnt_bounds, coords, labels)

if __name__ == "__main__":
    main()

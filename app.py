import streamlit as st
import rasterio
import numpy as np
import folium
from folium import plugins
from sklearn.cluster import DBSCAN
from shapely.geometry import Point
import matplotlib.pyplot as plt
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling

# Fonction pour téléverser un fichier
def upload_file(file, key):
    if file:
        st.session_state[key] = file.name
        with open(file.name, "wb") as f:
            f.write(file.read())
        st.success(f"Fichier {key} téléversé avec succès !")

# Charger un fichier TIFF et ses métadonnées
def load_tiff(file):
    with rasterio.open(file) as src:
        data = src.read(1)
        bounds = src.bounds  # Récupérer les bornes géographiques du fichier
    return data, bounds

# Calcul des hauteurs à partir des données MNS et MNT
def calculate_heights(mns, mnt):
    return mnt - mns  # Différence de hauteur

# Fonction pour détecter les arbres avec DBSCAN
def detect_trees(heights, height_threshold, eps, min_samples):
    # Appliquer DBSCAN
    coords = np.column_stack(np.where(heights > height_threshold))  # Coordonnées où la hauteur est supérieure au seuil
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    return coords, db.labels_

# Fonction pour calculer les centroïdes des clusters
def calculate_cluster_centroids(coords, labels):
    unique_labels = set(labels)
    centroids = []
    for label in unique_labels:
        if label != -1:  # Exclure le bruit
            cluster_coords = coords[labels == label]
            centroid = np.mean(cluster_coords, axis=0)
            centroids.append(centroid)
    return centroids

# Ajouter des centroids des arbres sur la carte
def add_tree_centroids_layer(map_object, centroids, bounds, shape, name):
    for centroid in centroids:
        lat, lon = centroid[0], centroid[1]
        folium.CircleMarker(location=[lat, lon], radius=5, color='red', fill=True).add_to(map_object)

# Fonction principale de l'application
def main():
    st.title("Application Cartographique")

    # Initialisation de la session
    if "files" not in st.session_state:
        st.session_state["files"] = {}

    # Téléversement des fichiers
    st.write("**Téléversement des fichiers**")
    uploaded_mnt = st.file_uploader("Téléverser un fichier MNT (TIFF)", type=["tif", "tiff"])
    upload_file(uploaded_mnt, "mnt")  # Sauvegarde dans session_state

    uploaded_mns = st.file_uploader("Téléverser un fichier MNS (TIFF)", type=["tif", "tiff"])
    upload_file(uploaded_mns, "mns")  # Sauvegarde dans session_state

    # Lorsque les deux fichiers sont téléversés
    if "mnt" in st.session_state and "mns" in st.session_state:
        mnt_file = st.session_state["mnt"]
        mns_file = st.session_state["mns"]

        mnt, mnt_bounds = load_tiff(mnt_file)
        mns, mns_bounds = load_tiff(mns_file)

        if mnt is None or mns is None:
            st.sidebar.error("Erreur lors du chargement des fichiers.")
        elif mnt_bounds != mns_bounds:
            st.sidebar.error("Les fichiers doivent avoir les mêmes bornes géographiques.")
        else:
            # Calcul des hauteurs
            heights = calculate_heights(mns, mnt)

            # Paramètres de détection
            height_threshold = st.sidebar.slider("Seuil de hauteur", 0.1, 20.0, 2.0, 0.1)
            eps = st.sidebar.slider("Rayon de voisinage", 0.1, 10.0, 2.0, 0.1)
            min_samples = st.sidebar.slider("Min. points pour un cluster", 1, 10, 5, 1)

            # Détection et visualisation
            if st.sidebar.button("Lancer la détection"):
                coords, tree_clusters = detect_trees(heights, height_threshold, eps, min_samples)
                num_trees = len(set(tree_clusters)) - (1 if -1 in tree_clusters else 0)
                st.sidebar.write(f"Nombre d'arbres détectés : {num_trees}")

                # Calcul des centroïdes des clusters
                centroids = calculate_cluster_centroids(coords, tree_clusters)

                # Mise à jour de la carte
                center_lat = (mnt_bounds[1] + mnt_bounds[3]) / 2
                center_lon = (mnt_bounds[0] + mnt_bounds[2]) / 2
                fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12)

                # Ajouter l'overlay du MNT à la carte
                folium.raster_layers.ImageOverlay(
                    image=mnt,
                    bounds=[[mnt_bounds[1], mnt_bounds[0]], [mnt_bounds[3], mnt_bounds[2]]],
                    opacity=0.5,
                    name="MNT"
                ).add_to(fmap)

                # Ajouter les centroids des arbres
                add_tree_centroids_layer(fmap, centroids, mnt_bounds, mnt.shape, "Arbres")

                folium_static(fmap, width=700, height=500)

    # Boutons d'actions
    st.write("### Actions disponibles")
    if st.button("Réinitialiser les fichiers"):
        st.session_state["files"] = {}
        st.success("Tous les fichiers ont été réinitialisés.")

if __name__ == "__main__":
    main()




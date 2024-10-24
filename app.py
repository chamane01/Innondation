import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
from pyproj import Transformer

# Exemple de données de coordonnées (latitude, longitude)
# Remplace ces données par tes propres données
grid_X = np.array([-74.006, -74.001, -74.005])  # Longitudes
grid_Y = np.array([40.7128, 40.7130, 40.7135])  # Latitudes

# Titre de l'application Streamlit
st.title("Visualisation de la Carte avec Matplotlib")

# Transformer pour passer de EPSG:4326 (lat/lon) à EPSG:3857 (Web Mercator)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

# Transformation des coordonnées en EPSG:3857 (Web Mercator)
grid_X_3857, grid_Y_3857 = transformer.transform(grid_X, grid_Y)

# Vérification des NaN et des Inf
valid_mask = np.isfinite(grid_X_3857) & np.isfinite(grid_Y_3857)

# Appliquer le masque pour ne conserver que les valeurs valides
grid_X_3857_valid = grid_X_3857[valid_mask]
grid_Y_3857_valid = grid_Y_3857[valid_mask]

# Vérification que les données filtrées ne sont pas vides
if len(grid_X_3857_valid) == 0 or len(grid_Y_3857_valid) == 0:
    st.error("Erreur : données transformées invalides (NaN ou Inf).")
else:
    # Création de la figure et des axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Ajouter des points (ou d'autres éléments) sur la carte
    ax.scatter(grid_X_3857_valid, grid_Y_3857_valid, color='red', s=100, label='Points de données')

    # Définir les limites des axes en fonction des valeurs valides
    ax.set_xlim(grid_X_3857_valid.min(), grid_X_3857_valid.max())
    ax.set_ylim(grid_Y_3857_valid.min(), grid_Y_3857_valid.max())

    # Ajouter le fond de carte avec les bonnes limites
    ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.OpenStreetMap.Mapnik)

    # Ajouter des labels et un titre
    ax.set_title("Carte avec Points de Données")
    ax.set_xlabel("Longitude (EPSG:3857)")
    ax.set_ylabel("Latitude (EPSG:3857)")
    ax.legend()

    # Afficher la carte dans Streamlit
    st.pyplot(fig)

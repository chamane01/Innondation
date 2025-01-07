import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Titre de l'application
st.title("Calculateur de volumes à partir d'un DEM")

# Importer un fichier DEM
uploaded_file = st.file_uploader("Téléchargez un fichier DEM (GeoTIFF uniquement) :", type=["tif", "tiff"])

if uploaded_file:
    # Charger le DEM
    with rasterio.open(uploaded_file) as src:
        dem_data = src.read(1)  # Lire la première bande
        dem_data[dem_data == src.nodata] = np.nan  # Gérer les valeurs no_data
        profile = src.profile

    st.success("Fichier DEM chargé avec succès !")
    
    # Affichage des métadonnées
    st.write("**Dimensions du DEM :**", dem_data.shape)
    st.write("**Résolution :**", profile["transform"][0], "unités par pixel")

    # Afficher le DEM sous forme d'image
    st.subheader("Aperçu du DEM")
    fig, ax = plt.subplots()
    cax = ax.imshow(dem_data, cmap="terrain")
    fig.colorbar(cax, ax=ax, label="Altitude (mètres)")
    st.pyplot(fig)

    # Choisir une altitude de référence
    st.subheader("Calcul du volume")
    reference_altitude = st.number_input(
        "Altitude de référence (mètres) :", value=0.0, step=0.1, format="%.1f"
    )

    if st.button("Calculer le volume"):
        # Calculer les volumes
        cell_area = profile["transform"][0] * abs(profile["transform"][4])  # Surface d'une cellule
        above_reference = np.nansum((dem_data - reference_altitude)[dem_data > reference_altitude]) * cell_area
        below_reference = np.nansum((reference_altitude - dem_data)[dem_data < reference_altitude]) * cell_area

        # Résultats
        st.write(f"**Volume au-dessus de l'altitude de référence :** {above_reference:.2f} m³")
        st.write(f"**Volume en dessous de l'altitude de référence :** {below_reference:.2f} m³")
        st.write(
            f"**Volume net (différence) :** {above_reference - below_reference:.2f} m³"
        )

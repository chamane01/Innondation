
# Importer les bibliothèques nécessaires
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import Polygon, box
from shapely.geometry import MultiPolygon
import contextily as ctx
import ezdxf  # Bibliothèque pour créer des fichiers DXF
from datetime import datetime
import rasterio

import streamlit as st
import numpy as np
import rasterio
import folium
from streamlit_folium import st_folium
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import os


# Fonction pour charger un fichier TIFF
def charger_tiff(fichier_tiff):
    try:
        with rasterio.open(fichier_tiff) as src:
            data = src.read(1)  # Lire la première bande
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
            return data, transform, crs, bounds
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier GeoTIFF : {e}")
        return None, None, None, None


# Fonction pour générer une carte de profondeur et sauvegarder comme image temporaire
def generer_image_profondeur(data_tiff, bounds_tiff, output_path):
    extent = [bounds_tiff[0], bounds_tiff[2], bounds_tiff[1], bounds_tiff[3]]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data_tiff, cmap='terrain', extent=extent)
    fig.colorbar(im, ax=ax, label="Altitude (m)")

    ax.set_title("Carte de profondeur (terrain)", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Sauvegarder l'image
    plt.savefig(output_path, format='png', bbox_inches='tight')
    plt.close(fig)


# Fonction pour créer une carte Folium avec superposition
def creer_carte_osm(data_tiff, bounds_tiff, niveau_inondation=None):
    try:
        lat_min, lon_min = bounds_tiff[1], bounds_tiff[0]
        lat_max, lon_max = bounds_tiff[3], bounds_tiff[2]
        center = [(lat_min + lat_max) / 2, (lon_min + lon_max) / 2]

        # Créer une carte Folium
        m = folium.Map(location=center, zoom_start=13, control_scale=True)

        # Générer une image temporaire pour la carte de profondeur
        depth_map_path = "temp_depth_map.png"
        generer_image_profondeur(data_tiff, bounds_tiff, depth_map_path)

        # Ajouter la superposition de la carte de profondeur
        img_overlay = folium.raster_layers.ImageOverlay(
            image=depth_map_path,
            bounds=[[lat_min, lon_min], [lat_max, lon_max]],
            opacity=0.7,
            interactive=True
        )
        img_overlay.add_to(m)

        # Gestion du niveau d'inondation si défini
        if niveau_inondation is not None:
            inondation_mask = data_tiff <= niveau_inondation
            zone_inondee = np.zeros_like(data_tiff, dtype=np.uint8)
            zone_inondee[inondation_mask] = 255

            # Générer une image temporaire pour les zones inondées
            flood_map_path = "temp_flood_map.png"
            fig, ax = plt.subplots(figsize=(8, 6))
            extent = [lon_min, lon_max, lat_min, lat_max]
            ax.imshow(zone_inondee, cmap=ListedColormap(['none', 'magenta']), extent=extent, alpha=0.5)
            plt.axis('off')
            plt.savefig(flood_map_path, format='png', bbox_inches='tight', transparent=True)
            plt.close(fig)

            flood_overlay = folium.raster_layers.ImageOverlay(
                image=flood_map_path,
                bounds=[[lat_min, lon_min], [lat_max, lon_max]],
                opacity=0.6,
                interactive=True
            )
            flood_overlay.add_to(m)

        folium.LayerControl().add_to(m)
        return m
    except Exception as e:
        st.error(f"Erreur lors de la création de la carte : {e}")
        return None


# Fonction pour générer une carte statique combinée
def generer_carte_combinee(data_tiff, bounds_tiff, niveau_inondation, output_path):
    # Extraire les limites des coordonnées
    lon_min, lat_min, lon_max, lat_max = bounds_tiff[0], bounds_tiff[1], bounds_tiff[2], bounds_tiff[3]

    extent = [lon_min, lon_max, lat_min, lat_max]

    # Masque des zones inondées
    inondation_mask = data_tiff <= niveau_inondation

    fig, ax = plt.subplots(figsize=(8, 6))

    # Afficher la carte de profondeur
    im = ax.imshow(data_tiff, cmap='terrain', extent=extent)
    cbar = fig.colorbar(im, ax=ax, label="Altitude (m)")

    # Superposer les zones inondées
    zone_inondee = np.zeros_like(data_tiff, dtype=np.uint8)
    zone_inondee[inondation_mask] = 1
    ax.imshow(zone_inondee, cmap=ListedColormap(["none", "magenta"]), extent=extent, alpha=0.5)

    # Ajouter des titres et des axes
    ax.set_title("Carte combinée : Profondeur et Zones inondées", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Sauvegarder l'image
    plt.savefig(output_path, format='png', bbox_inches='tight')
    plt.close(fig)


# Interface principale Streamlit
def main():
    st.title("Analyse des zones inondées")
    st.markdown("### Téléchargez un fichier GeoTIFF pour analyser les zones inondées.")

    fichier_tiff = st.file_uploader("Téléchargez un fichier GeoTIFF", type=["tif"], key="file_uploader")

    if fichier_tiff is not None:
        data_tiff, transform_tiff, crs_tiff, bounds_tiff = charger_tiff(fichier_tiff)

        if data_tiff is not None:
            st.write(f"Dimensions : {data_tiff.shape}")
            st.write(f"Altitude min : {data_tiff.min()}, max : {data_tiff.max()}")

            st.write("### Carte de profondeur avec OSM")
            m = creer_carte_osm(data_tiff, bounds_tiff)
            st_folium(m, width=700, height=500, key="osm_map")

            niveau_inondation = st.slider(
                "Choisissez le niveau d'inondation",
                float(data_tiff.min()),
                float(data_tiff.max()),
                float(np.percentile(data_tiff, 50)),
                step=0.1,
                key="niveau_inondation"
            )

            if st.button("Afficher la zone inondée", key="btn_zone_inondee"):
                st.write(f"### Zone inondée pour une altitude de {niveau_inondation:.2f} m")
                m = creer_carte_osm(data_tiff, bounds_tiff, niveau_inondation=niveau_inondation)
                st_folium(m, width=700, height=500, key="flood_map")

            # Bouton pour créer une carte statique
            if st.button("Créer une carte statique", key="btn_carte_statique"):
                carte_statique_path = "carte_combinee.png"
                generer_carte_combinee(data_tiff, bounds_tiff, niveau_inondation, carte_statique_path)
                st.image(carte_statique_path, caption="Carte statique combinée", use_column_width=True)

                # Supprimer l'image après affichage
                if os.path.exists(carte_statique_path):
                    os.remove(carte_statique_path)

            # Supprimer les fichiers temporaires après usage
            if os.path.exists("temp_depth_map.png"):
                os.remove("temp_depth_map.png")
            if os.path.exists("temp_flood_map.png"):
                os.remove("temp_flood_map.png")


if __name__ == "__main__":
    main()








import streamlit as st
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Fonction pour charger le fichier TIFF
def charger_tiff(fichier_tiff):
    try:
        with rasterio.open(fichier_tiff) as src:
            data = src.read(1)  # Lire la première bande
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
            return data, transform, crs, bounds
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier GeoTIFF : {e}")
        return None, None, None, None

# Fonction pour afficher la carte de profondeur
def afficher_carte_profondeur(data_tiff, bounds_tiff):
    # Étendue géographique (extent)
    extent = [bounds_tiff[0], bounds_tiff[2], bounds_tiff[1], bounds_tiff[3]]

    # Créer la figure
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data_tiff, cmap='terrain', extent=extent)
    cbar = fig.colorbar(im, ax=ax, label="Altitude (m)")

    # Titre et axes
    ax.set_title("Carte de profondeur (terrain)", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    st.pyplot(fig)

# Fonction pour afficher la zone inondée en magenta
def afficher_zone_inondee(data_tiff, niveau_inondation, bounds_tiff):
    # Étendue géographique (extent)
    extent = [bounds_tiff[0], bounds_tiff[2], bounds_tiff[1], bounds_tiff[3]]

    # Créer un masque des pixels inondés
    inondation_mask = data_tiff <= niveau_inondation
    nb_pixels_inondes = np.sum(inondation_mask)

    # Créer une nouvelle couche pour la zone inondée (valeurs 1 pour inondées, 0 sinon)
    zone_inondee = np.zeros_like(data_tiff, dtype=np.uint8)
    zone_inondee[inondation_mask] = 1

    # Créer la figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Afficher l'image de fond (altitudes)
    im = ax.imshow(data_tiff, cmap='terrain', extent=extent)
    cbar = fig.colorbar(im, ax=ax, label="Altitude (m)")

    # Superposer la couche des zones inondées en magenta
    cmap = ListedColormap(["none", "red"])
    ax.imshow(
        zone_inondee,
        cmap=cmap,
        alpha=0.5,
        extent=extent
    )

    # Ajouter une légende manuelle pour les zones inondées
    legend_elements = [
        Patch(facecolor='red', edgecolor='none', label='Zone inondée (red)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Titre et axes
    ax.set_title(f"Zone inondée pour une cote de {niveau_inondation:.2f} m", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Afficher le nombre de pixels inondés
    st.write(f"**Nombre de pixels inondés :** {nb_pixels_inondes}")
    st.pyplot(fig)

# Interface Streamlit
def main():
    st.title("Analyse des zones inondées")
    st.markdown("### Téléchargez un fichier GeoTIFF pour analyser les zones inondées.")

    # Téléversement du fichier GeoTIFF
    fichier_tiff = st.file_uploader("Téléchargez un fichier GeoTIFF", type=["tif"])

    if fichier_tiff is not None:
        # Charger le fichier TIFF
        data_tiff, transform_tiff, crs_tiff, bounds_tiff = charger_tiff(fichier_tiff)

        if data_tiff is not None:
            # Afficher les informations de base
            st.write(f"Dimensions : {data_tiff.shape}")
            st.write(f"Altitude min : {data_tiff.min()}, max : {data_tiff.max()}")

            # Afficher la carte de profondeur
            if st.checkbox("Afficher la carte de profondeur"):
                afficher_carte_profondeur(data_tiff, bounds_tiff)

            # Sélectionner le niveau d'inondation
            niveau_inondation = st.slider(
                "Choisissez le niveau d'inondation",
                float(data_tiff.min()),
                float(data_tiff.max()),
                float(np.percentile(data_tiff, 50)),  # Par défaut, la médiane
                step=0.1
            )

            # Bouton pour afficher la zone inondée
            if st.button("Afficher la zone inondée"):
                afficher_zone_inondee(data_tiff, niveau_inondation, bounds_tiff)

if __name__ == "__main__":
    main()

















# Streamlit - Titre de l'application avec deux logos centrés
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.image("POPOPO.jpg", width=150)
with col2:
    st.image("logo.png", width=150)
with col3:
    st.write("")  # Cette colonne est laissée vide pour centrer les logos

st.title("Carte des zones inondées avec niveaux d'eau et surface")

# Initialiser session_state pour stocker les données d'inondation
if 'flood_data' not in st.session_state:
    st.session_state.flood_data = {
        'surface_bleu': None,  
        'volume_eau': None,
        'niveau_inondation': 0.0
    }

# Étape 1 : Sélectionner un site ou téléverser un fichier
st.markdown("## Sélectionner un site ou téléverser un fichier GeoTIFF")
uploaded_tiff_file = st.file_uploader("Téléversez un fichier GeoTIFF (.tif)", type=["tif"])

# Charger les données depuis un fichier GeoTIFF
def charger_tiff(fichier_tiff):
    try:
        with rasterio.open(fichier_tiff) as src:
            # Lire les métadonnées et les données raster
            data = src.read(1)  # Lire la première bande
            transform = src.transform  # Transformation spatiale
            crs = src.crs  # Système de coordonnées
            return data, transform, crs
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier GeoTIFF : {e}")
        return None, None, None

# Si un fichier GeoTIFF est téléversé
if uploaded_tiff_file is not None:
    data_tiff, transform_tiff, crs_tiff = charger_tiff(uploaded_tiff_file)

    if data_tiff is not None:
        st.write("**Informations sur le fichier GeoTIFF :**")
        st.write(f"Dimensions : {data_tiff.shape}")
        st.write(f"Valeurs min : {data_tiff.min()}, max : {data_tiff.max()}")
        st.write(f"Système de coordonnées : {crs_tiff}")

        # Afficher les données raster sous forme d'image
        fig, ax = plt.subplots(figsize=(8, 6))
        extent = (
            transform_tiff[2],  # Min X
            transform_tiff[2] + transform_tiff[0] * data_tiff.shape[1],  # Max X
            transform_tiff[5] + transform_tiff[4] * data_tiff.shape[0],  # Min Y
            transform_tiff[5]  # Max Y
        )
        cax = ax.imshow(data_tiff, cmap='terrain', extent=extent)
        fig.colorbar(cax, ax=ax, label="Altitude (m)")
        ax.set_title("Carte d'altitude (GeoTIFF)")
        st.pyplot(fig)

        # Niveau d'eau et analyse
        st.session_state.flood_data['niveau_inondation'] = st.number_input(
            "Entrez le niveau d'eau (mètres)", min_value=float(data_tiff.min()), max_value=float(data_tiff.max()), step=0.1
        )

        if st.button("Calculer et afficher la zone inondée"):
            # Calculer la zone inondée
            inondation_mask = data_tiff <= st.session_state.flood_data['niveau_inondation']
            surface_inondee = np.sum(inondation_mask) * (transform_tiff[0] * transform_tiff[4]) / 10_000  # En hectares
            st.session_state.flood_data['surface_bleu'] = surface_inondee
            st.write(f"**Surface inondée :** {surface_inondee:.2f} hectares")

            # Afficher la carte de l'inondation
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(data_tiff, cmap='terrain', extent=extent)
            ax.imshow(inondation_mask, cmap='Blues', alpha=0.5, extent=extent)
            ax.set_title("Zone inondée (en bleu)")
            fig.colorbar(cax, ax=ax, label="Altitude (m)")
            st.pyplot(fig)


st.title("Carte des zones inondées avec niveaux d'eau et surface")

# Initialiser session_state pour stocker les données d'inondation
if 'flood_data' not in st.session_state:
    st.session_state.flood_data = {
        'surface_bleu': None,  
        'volume_eau': None,
        'niveau_inondation': 0.0
    }

# Étape 1 : Sélectionner un site ou téléverser un fichier
st.markdown("## Sélectionner un site ou téléverser un fichier")
option_site = st.selectbox("Sélectionnez un site", ("Aucun", "AYAME 1", "AYAME 2"))
uploaded_file = st.file_uploader("Téléversez un fichier Excel ou TXT", type=["xlsx", "txt"])

# Charger les données en fonction de l'option sélectionnée
def charger_fichier(fichier, is_uploaded=False):
    try:
        if is_uploaded:
            if fichier.name.endswith('.xlsx'):
                df = pd.read_excel(fichier)
            elif fichier.name.endswith('.txt'):
                df = pd.read_csv(fichier, sep=",", header=None, names=["X", "Y", "Z"])
        else:
            df = pd.read_csv(fichier, sep=",", header=None, names=["X", "Y", "Z"])
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {e}")
        return None

if option_site == "AYAME 1":
    df = charger_fichier('AYAME1.txt')
elif option_site == "AYAME 2":
    df = charger_fichier('AYAME2.txt')
elif uploaded_file is not None:
    df = charger_fichier(uploaded_file, is_uploaded=True)
else:
    st.warning("Veuillez sélectionner un site ou téléverser un fichier pour démarrer.")
    df = None




uploaded_geojson_file = st.file_uploader("Téléversez un fichier GeoJSON pour les routes", type=["geojson"])
def charger_geojson(fichier):
    try:
        gdf = gpd.read_file(fichier)
        return gdf
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier GeoJSON : {e}")
        return None

# Charger les données du fichier GeoJSON des routes
routes_gdf = None
if uploaded_geojson_file is not None:
    routes_gdf = charger_geojson(uploaded_geojson_file)

# Charger et filtrer les bâtiments dans l'emprise de la carte
try:
    batiments_gdf = gpd.read_file("batiments2.geojson")
    if df is not None:
        emprise = box(df['X'].min(), df['Y'].min(), df['X'].max(), df['Y'].max())
        batiments_gdf = batiments_gdf.to_crs(epsg=32630)
        batiments_dans_emprise = batiments_gdf[batiments_gdf.intersects(emprise)]
    else:
        batiments_dans_emprise = None
except Exception as e:
    st.error(f"Erreur lors du chargement des bâtiments : {e}")
    batiments_dans_emprise = None

# Traitement des données si le fichier est chargé
if df is not None:
    st.markdown("---")

    # Vérification du fichier : colonnes X, Y, Z
    if 'X' not in df.columns or 'Y' not in df.columns or 'Z' not in df.columns:
        st.error("Erreur : colonnes 'X', 'Y' et 'Z' manquantes.")
    else:
        st.session_state.flood_data['niveau_inondation'] = st.number_input("Entrez le niveau d'eau (mètres)", min_value=0.0, step=0.1)
        interpolation_method = st.selectbox("Méthode d'interpolation", ['linear', 'nearest'])

        X_min, X_max = df['X'].min(), df['X'].max()
        Y_min, Y_max = df['Y'].min(), df['Y'].max()
        resolution = st.number_input("Résolution de la grille", value=300, min_value=100, max_value=1000)
        grid_X, grid_Y = np.mgrid[X_min:X_max:resolution*1j, Y_min:Y_max:resolution*1j]
        grid_Z = griddata((df['X'], df['Y']), df['Z'], (grid_X, grid_Y), method=interpolation_method)

        def calculer_surface_bleue(niveau_inondation):
            return np.sum((grid_Z <= niveau_inondation)) * (grid_X[1, 0] - grid_X[0, 0]) * (grid_Y[0, 1] - grid_Y[0, 0]) / 10000

        def calculer_volume(surface_bleue):
            return surface_bleue * st.session_state.flood_data['niveau_inondation'] * 10000

        if st.button("Afficher la carte d'inondation"):
            surface_bleue = calculer_surface_bleue(st.session_state.flood_data['niveau_inondation'])
            volume_eau = calculer_volume(surface_bleue)
            st.session_state.flood_data['surface_bleu'] = surface_bleue
            st.session_state.flood_data['volume_eau'] = volume_eau

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlim(X_min, X_max)
            ax.set_ylim(Y_min, Y_max)
            ctx.add_basemap(ax, crs="EPSG:32630", source=ctx.providers.OpenStreetMap.Mapnik)
            # Ajouter des coordonnées sur les quatre côtés
            ax.tick_params(axis='both', which='both', direction='in', length=6, width=1, color='black', labelsize=10)
            ax.set_xticks(np.linspace(X_min, X_max, num=5))# Coordonnées sur l'axe X
            ax.set_yticks(np.linspace(Y_min, Y_max, num=5))# Coordonnées sur l'axe Y
            ax.xaxis.set_tick_params(labeltop=True)# Affiche les labels sur le haut
            ax.yaxis.set_tick_params(labelright=True)# Affiche les labels à droite
            
            # Ajouter les lignes pour relier les tirets (lignes horizontales et verticales)
            # Lignes verticales (de haut en bas)
            for x in np.linspace(X_min, X_max, num=5):
                ax.axvline(x, color='black', linewidth=0.5, linestyle='--',alpha=0.2)
            # Lignes horizontales (de gauche à droite)
            for y in np.linspace(Y_min, Y_max, num=5):
                ax.axhline(y, color='black', linewidth=0.5, linestyle='--',alpha=0.2)

            # Ajouter les croisillons aux intersections avec opacité à 100%
            # Déterminer les positions d'intersection
            intersections_x = np.linspace(X_min, X_max, num=5)
            intersections_y = np.linspace(Y_min, Y_max, num=5)
            # Tracer les croisillons aux intersections avec opacité à 100%
            for x in intersections_x:
                for y in intersections_y:
                    ax.plot(x, y, 'k+', markersize=7, alpha=1.0) # 'k+' : plus noire, alpha=1 pour opacité 100%
                    


            # Tracer la zone inondée avec les contours
            contours_inondation = ax.contour(grid_X, grid_Y, grid_Z, levels=[st.session_state.flood_data['niveau_inondation']], colors='red', linewidths=1)
            ax.clabel(contours_inondation, inline=True, fontsize=10, fmt='%1.1f m')
            ax.contourf(grid_X, grid_Y, grid_Z, levels=[-np.inf, st.session_state.flood_data['niveau_inondation']], colors='#007FFF', alpha=0.5)

            # Transformer les contours en polygones pour analyser les bâtiments
            contour_paths = [Polygon(path.vertices) for collection in contours_inondation.collections for path in collection.get_paths()]
            zone_inondee = gpd.GeoDataFrame(geometry=[MultiPolygon(contour_paths)], crs="EPSG:32630")

            # Filtrer et afficher tous les bâtiments
            if batiments_dans_emprise is not None:
                batiments_dans_emprise.plot(ax=ax, facecolor='grey', edgecolor='black', linewidth=0.5, alpha=0.6, label="Bâtiments non inondés")
                
                # Séparer les bâtiments inondés
                batiments_inondes = batiments_dans_emprise[batiments_dans_emprise.intersects(zone_inondee.unary_union)]
                nombre_batiments_inondes = len(batiments_inondes)

                # Afficher les bâtiments inondés en rouge
                batiments_inondes.plot(ax=ax, facecolor='red', edgecolor='red', linewidth=1, alpha=0.8, label="Bâtiments inondés")

                st.write(f"Nombre de bâtiments dans la zone inondée : {nombre_batiments_inondes}")
                ax.legend()
            else:
                st.write("Aucun bâtiment à analyser dans cette zone.")

            if routes_gdf is not None:
                routes_gdf = routes_gdf.to_crs(epsg=32630)  # Reprojeter les données si nécessaire
                routes_gdf.plot(ax=ax, color='orange', linewidth=2, label="Routes")
                st.write(f"**Nombre de routes affichées :** {len(routes_gdf)}")

            

            

            st.pyplot(fig)

            # Enregistrer les contours en fichier DXF
            doc = ezdxf.new(dxfversion='R2010')
            msp = doc.modelspace()
            for collection in contours_inondation.collections:
                for path in collection.get_paths():
                    points = path.vertices
                    for i in range(len(points)-1):
                        msp.add_line(points[i], points[i+1])

            dxf_file = "contours_inondation.dxf"
            doc.saveas(dxf_file)
            carte_file = "carte_inondation.png"
            fig.savefig(carte_file)

            with open(carte_file, "rb") as carte:
                st.download_button(label="Télécharger la carte", data=carte, file_name=carte_file, mime="image/png")

            with open(dxf_file, "rb") as dxf:
                st.download_button(label="Télécharger le fichier DXF", data=dxf, file_name=dxf_file, mime="application/dxf")

            # Afficher les résultats
            now = datetime.now()
            st.markdown("## Résultats")
            st.write(f"**Surface inondée :** {surface_bleue:.2f} hectares")
            st.write(f"**Volume d'eau :** {volume_eau:.2f} m³")
            st.write(f"**Niveau d'eau :** {st.session_state.flood_data['niveau_inondation']} m")
            st.write(f"**Nombre de bâtiments inondés :** {nombre_batiments_inondes}")
            st.write(f"**Date :** {now.strftime('%Y-%m-%d')}")
            st.write(f"**Heure :** {now.strftime('%H:%M:%S')}")
            st.write(f"**Système de projection :** EPSG:32630")

# Fonction pour générer la carte de profondeur avec dégradé de couleurs
def generate_depth_map(label_rotation_x=0, label_rotation_y=0):

    # Détection des bas-fonds
    def detecter_bas_fonds(grid_Z, seuil_rel_bas_fond=1.5):
        """
        Détermine les bas-fonds en fonction de la profondeur Z relative.
        Bas-fond = Z < moyenne(Z) - seuil_rel_bas_fond * std(Z)
        """
        moyenne_Z = np.mean(grid_Z)
        ecart_type_Z = np.std(grid_Z)
        seuil_bas_fond = moyenne_Z - seuil_rel_bas_fond * ecart_type_Z
        bas_fonds = grid_Z < seuil_bas_fond
        return bas_fonds, seuil_bas_fond

    # Calcul des surfaces des bas-fonds
    def calculer_surface_bas_fond(bas_fonds, grid_X, grid_Y):
        """
        Calcule la surface des bas-fonds en hectares.
        """
        resolution = (grid_X[1, 0] - grid_X[0, 0]) * (grid_Y[0, 1] - grid_Y[0, 0]) / 10000  # Résolution en hectares
        surface_bas_fond = np.sum(bas_fonds) * resolution
        return surface_bas_fond

    bas_fonds, seuil_bas_fond = detecter_bas_fonds(grid_Z)
    surface_bas_fond = calculer_surface_bas_fond(bas_fonds, grid_X, grid_Y)

    
    # Appliquer un dégradé de couleurs sur la profondeur (niveau de Z)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(X_min, X_max)
    ax.set_ylim(Y_min, Y_max)
    ctx.add_basemap(ax, crs="EPSG:32630", source=ctx.providers.OpenStreetMap.Mapnik)
    ax.tick_params(axis='both', which='both', direction='in', length=6, width=1, color='black', labelsize=10)
    ax.set_xticks(np.linspace(X_min, X_max, num=5))
    ax.set_yticks(np.linspace(Y_min, Y_max, num=5))
    ax.xaxis.set_tick_params(labeltop=True)
    ax.yaxis.set_tick_params(labelright=True)

     # Masquer les coordonnées aux extrémités
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    ax.set_xticklabels(
         ["" if x == X_min or x == X_max else f"{int(x)}" for x in xticks],
        rotation=label_rotation_x,
    )
    ax.set_yticklabels(
        ["" if y == Y_min or y == Y_max else f"{int(y)}" for y in yticks],
        rotation=label_rotation_y,
        va="center"  # Alignement vertical des étiquettes Y
    )
    #modifier rotation
    for label in ax.get_xticklabels():
        label.set_rotation(label_rotation_x)

    for label in ax.get_yticklabels():
        label.set_rotation(label_rotation_y)

    

    # Ajouter les contours pour la profondeur
    depth_levels = np.linspace(grid_Z.min(), grid_Z.max(), 100)
    cmap = plt.cm.plasma  # Couleurs allant de bleu à jaune
    cont = ax.contourf(grid_X, grid_Y, grid_Z, levels=depth_levels, cmap=cmap)
    cbar = plt.colorbar(cont, ax=ax)
    cbar.set_label('Profondeur (m)', rotation=270)

    # Ajouter les bas-fonds en cyan
    ax.contourf(grid_X, grid_Y, bas_fonds, levels=[0.5, 1], colors='cyan', alpha=0.4, label='Bas-fonds')
    
    # Ajouter une ligne de contour autour des bas-fonds
    contour_lines = ax.contour(
        grid_X, grid_Y, grid_Z,
        levels=[seuil_bas_fond],  # Niveau correspondant au seuil des bas-fonds
        colors='black',  # Couleur des contours
        linewidths=1.5,
        linestyles='solid',# Épaisseur de la ligne
    )
    # Ajouter des labels pour les contours
    ax.clabel(contour_lines,
        inline=True,
        fmt={seuil_bas_fond: f"{seuil_bas_fond:.2f} m"},  # Format du label
        fontsize=12
    )


    # Ajouter des lignes pour relier les tirets
    for x in np.linspace(X_min, X_max, num=5):
        ax.axvline(x, color='black', linewidth=0.5, linestyle='--', alpha=0.2)
    for y in np.linspace(Y_min, Y_max, num=5):
        ax.axhline(y, color='black', linewidth=0.5, linestyle='--', alpha=0.2)
#croisillon 
    intersections_x = np.linspace(X_min, X_max, num=5)
    intersections_y = np.linspace(Y_min, Y_max, num=5)
    for x in intersections_x:
        for y in intersections_y:
            ax.plot(x, y, 'k+', markersize=7, alpha=1.0)

    


    # Ajouter les bâtiments
    if batiments_dans_emprise is not None:
        batiments_dans_emprise.plot(ax=ax, facecolor='grey', edgecolor='black', linewidth=0.5, alpha=0.6)

    # Affichage de la carte de profondeur
    st.pyplot(fig)
    # Afficher les surfaces calculées
    st.write(f"**Surface des bas-fonds** : {surface_bas_fond:.2f} hectares")

# Ajouter un bouton pour générer la carte de profondeur
if st.button("Générer la carte de profondeur avec bas-fonds"):
    generate_depth_map(label_rotation_x=0, label_rotation_y=-90)







# Fonction pour charger les polygones
def charger_polygones(uploaded_file):
    try:
        if uploaded_file is not None:
            # Lire le fichier GeoJSON téléchargé
            polygones_gdf = gpd.read_file(uploaded_file)
            
            # Convertir le GeoDataFrame au CRS EPSG:32630
            polygones_gdf = polygones_gdf.to_crs(epsg=32630)
            
            # Créer une emprise (bounding box) basée sur les données
            if 'X' in df.columns and 'Y' in df.columns:
                emprise = box(df['X'].min(), df['Y'].min(), df['X'].max(), df['Y'].max())
                polygones_dans_emprise = polygones_gdf[polygones_gdf.intersects(emprise)]  # Filtrer les polygones dans l'emprise
            else:
                polygones_dans_emprise = polygones_gdf  # Si pas de colonne X/Y dans df, prendre tous les polygones
        else:
            polygones_dans_emprise = None
    except Exception as e:
        st.error(f"Erreur lors du chargement des polygones : {e}")
        polygones_dans_emprise = None

    return polygones_dans_emprise

# Fonction pour afficher les polygones
def afficher_polygones(ax, gdf_polygones, edgecolor='white', linewidth=1.0):
    if gdf_polygones is not None and not gdf_polygones.empty:
        gdf_polygones.plot(ax=ax, facecolor='none', edgecolor=edgecolor, linewidth=linewidth)
    else:
        st.warning("Aucun polygone à afficher dans l'emprise.")

# Exemple d'appel dans l'interface Streamlit
st.title("Affichage des Polygones et Profondeur")

# Téléchargement du fichier GeoJSON pour les polygones
uploaded_file = st.file_uploader("Téléverser un fichier GeoJSON", type="geojson")



def calculer_surface_bas_fonds_polygones(polygones, bas_fonds, grid_X, grid_Y):
    try:
        # Conversion des bas-fonds en GeoDataFrame
        resolution = (grid_X[1, 0] - grid_X[0, 0]) * (grid_Y[0, 1] - grid_Y[0, 0])
        bas_fonds_coords = [
            Polygon([
                (grid_X[i, j], grid_Y[i, j]),
                (grid_X[i + 1, j], grid_Y[i + 1, j]),
                (grid_X[i + 1, j + 1], grid_Y[i + 1, j + 1]),
                (grid_X[i, j + 1], grid_Y[i, j + 1])
            ])
            for i in range(grid_X.shape[0] - 1)
            for j in range(grid_X.shape[1] - 1)
            if bas_fonds[i, j]
        ]
        bas_fonds_gdf = gpd.GeoDataFrame(geometry=bas_fonds_coords, crs="EPSG:32630")

        # Intersection entre bas-fonds et polygones
        intersection = gpd.overlay(polygones, bas_fonds_gdf, how="intersection")
        
        # Calcul de la surface totale
        surface_totale = intersection.area.sum() / 10_000  # Convertir en hectares
        return surface_totale
    except Exception as e:
        st.error(f"Erreur dans le calcul de la surface des bas-fonds : {e}")
        return 0


# Définir la fonction detecter_bas_fonds en dehors de generate_depth_map
def detecter_bas_fonds(grid_Z, seuil_rel_bas_fond=1.5):
    moyenne_Z = np.mean(grid_Z)
    ecart_type_Z = np.std(grid_Z)
    seuil_bas_fond = moyenne_Z - seuil_rel_bas_fond * ecart_type_Z
    bas_fonds = grid_Z < seuil_bas_fond
    return bas_fonds, seuil_bas_fond

# Définir la fonction calculer_surface_bas_fond en dehors de generate_depth_map
def calculer_surface_bas_fond(bas_fonds, grid_X, grid_Y):
    resolution = (grid_X[1, 0] - grid_X[0, 0]) * (grid_Y[0, 1] - grid_Y[0, 0]) / 10000  # Résolution en hectares
    surface_bas_fond = np.sum(bas_fonds) * resolution
    return surface_bas_fond

# Fonction pour générer la carte de profondeur
def generate_depth_map(ax, grid_Z, grid_X, grid_Y, X_min, X_max, Y_min, Y_max, label_rotation_x=0, label_rotation_y=0):
    # Appliquer un dégradé de couleurs sur la profondeur (niveau de Z)
    ax.set_xlim(X_min, X_max)
    ax.set_ylim(Y_min, Y_max)

    # Afficher la carte de fond OpenStreetMap en EPSG:32630
    ctx.add_basemap(ax, crs="EPSG:32630", source=ctx.providers.OpenStreetMap.Mapnik)

    ax.tick_params(axis='both', which='both', direction='in', length=6, width=1, color='black', labelsize=10)
    ax.set_xticks(np.linspace(X_min, X_max, num=5))
    ax.set_yticks(np.linspace(Y_min, Y_max, num=5))
    ax.xaxis.set_tick_params(labeltop=True)
    ax.yaxis.set_tick_params(labelright=True)

    # Masquer les coordonnées aux extrémités
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    ax.set_xticklabels(
        ["" if x == X_min or x == X_max else f"{int(x)}" for x in xticks],
        rotation=label_rotation_x,
    )
    ax.set_yticklabels(
        ["" if y == Y_min or y == Y_max else f"{int(y)}" for y in yticks],
        rotation=label_rotation_y,
        va="center"  # Alignement vertical des étiquettes Y
    )

    # Modifier rotation
    for label in ax.get_xticklabels():
        label.set_rotation(label_rotation_x)

    for label in ax.get_yticklabels():
        label.set_rotation(label_rotation_y)

    # Ajouter les contours pour la profondeur et Barre verticale
    depth_levels = np.linspace(grid_Z.min(), grid_Z.max(), 100)
    cmap = plt.cm.plasma  # Couleurs allant de bleu à jaune
    cont = ax.contourf(grid_X, grid_Y, grid_Z, levels=depth_levels, cmap=cmap)
    cbar = plt.colorbar(cont, ax=ax)
    cbar.set_label('Profondeur (m)', rotation=270, labelpad=20)

    # Ajouter les bas-fonds en cyan
    bas_fonds, seuil_bas_fond = detecter_bas_fonds(grid_Z)  # Appel à la fonction externe
    ax.contourf(grid_X, grid_Y, bas_fonds, levels=[0.5, 1], colors='cyan', alpha=0.4, label='Bas-fonds')

    # Ajouter une ligne de contour autour des bas-fonds
    contour_lines = ax.contour(
        grid_X, grid_Y, grid_Z,
        levels=[seuil_bas_fond],  # Niveau correspondant au seuil des bas-fonds
        colors='black',  # Couleur des contours
        linewidths=1.5,
        linestyles='solid',
    )
    intersections_x = np.linspace(X_min, X_max, num=5)
    intersections_y = np.linspace(Y_min, Y_max, num=5)
    for x in intersections_x:
        for y in intersections_y:
            ax.plot(x, y, 'k+', markersize=7, alpha=1.0)

    # Ajouter des labels pour les contours
    ax.clabel(contour_lines, inline=True, fmt={seuil_bas_fond: f"{seuil_bas_fond:.2f} m"}, fontsize=12, colors='white')

    # Ajouter des lignes pour relier les tirets
    for x in np.linspace(X_min, X_max, num=5):
        ax.axvline(x, color='black', linewidth=0.5, linestyle='--', alpha=0.2)
    for y in np.linspace(Y_min, Y_max, num=5):
        ax.axhline(y, color='black', linewidth=0.5, linestyle='--', alpha=0.2)

    # Affichage de la carte de profondeur
    surface_bas_fond = calculer_surface_bas_fond(bas_fonds, grid_X, grid_Y)
    st.write(f"**Surface des bas-fonds** : {surface_bas_fond:.2f} hectares")
    # Afficher la surface des bas-fonds dans les polygones
    st.write(f"**Surface des bas-fonds dans les polygones** : {surface_bas_fond_polygones:.2f} hectares")

    
    # Ajouter des labels sous l'emprise de la carte de profondeur
    label_y_position = Y_min - (Y_max - Y_min) * 0.10
    ax.text(
        X_min + (X_max - X_min) * 0,  # Position horizontale (10% de la largeur)
        label_y_position,
        f"Surface des bas-fonds :",
        fontsize=12,
        color="black",
        ha="left",  # Aligné à gauche
        va="top",# Aligné en haut
        fontweight='bold',
    )
    ax.text(
        X_min + (X_max - X_min) * 0.37,  # Position horizontale (10% de la largeur)
        label_y_position - (Y_max - Y_min) * 0,  # Légèrement plus bas
        f"{surface_bas_fond:.2f} hectares",
        fontsize=12,
        color="black",
        ha="left",  # Aligné à gauche
        va="top",   # Aligné en haut
    )
    
    ax.text(
        X_min + (X_max - X_min) * 0,  # Position horizontale (10% de la largeur)
        label_y_position - (Y_max - Y_min) * 0.10,  # Légèrement plus bas
        f"Surface des bas-fonds dans les polygones :",
        fontsize=12,
        color="black",
        ha="left",  # Aligné à gauche
        va="top",
        fontweight='bold',# Aligné en haut
    )
    ax.text(
        X_min + (X_max - X_min) * 0.67,  # Position horizontale (10% de la largeur)
        label_y_position - (Y_max - Y_min) * 0.10,  # Légèrement plus bas
       f"{surface_bas_fond_polygones:.2f} hectares",
        fontsize=12,
        color="black",
        ha="left",  # Aligné à gauche
        va="top",# Aligné en haut
    )
    ax.text(
        X_min + (X_max - X_min) * 0,  # Position horizontale (10% de la largeur)
        label_y_position - (Y_max - Y_min) * 0.20,  # Légèrement plus bas
        f"Cote du bafond :",
        fontsize=12,
        color="black",
        ha="left",  # Aligné à gauche
        va="top",
        fontweight='bold',# Aligné en haut
    )
    ax.text(
        X_min + (X_max - X_min) * 0.26,  # Position horizontale (10% de la largeur)
        label_y_position - (Y_max - Y_min) * 0.20,  # Légèrement plus bas
        f"{seuil_bas_fond:.2f} m",
        fontsize=12,
        color="black",
        ha="left",  # Aligné à gauche
        va="top",# Aligné en haut
    )
    


# Ajouter les polygones sur la carte
if st.button("Afficher les polygones"):
    # Charger les polygones
    polygones_dans_emprise = charger_polygones(uploaded_file)

    # Si des polygones sont chargés, utiliser leur emprise pour ajuster les limites
    if polygones_dans_emprise is not None:
        # Calculer les limites du polygone
        X_min_polygone, Y_min_polygone, X_max_polygone, Y_max_polygone = polygones_dans_emprise.total_bounds
        
        # Calculer les limites de la carte de profondeur
        X_min_depth, Y_min_depth, X_max_depth, Y_max_depth = grid_X.min(), grid_Y.min(), grid_X.max(), grid_Y.max()

        # Vérifier si l'emprise de la carte de profondeur couvre celle des polygones
        if (X_min_depth <= X_min_polygone and X_max_depth >= X_max_polygone and
            Y_min_depth <= Y_min_polygone and Y_max_depth >= Y_max_polygone):
            X_min, Y_min, X_max, Y_max = X_min_depth, Y_min_depth, X_max_depth, Y_max_depth
        else:
            marge = 0.1
            X_range = X_max_polygone - X_min_polygone
            Y_range = Y_max_polygone - Y_min_polygone
            
            X_min = min(X_min_depth, X_min_polygone - X_range * marge)
            Y_min = min(Y_min_depth, Y_min_polygone - Y_range * marge)
            X_max = max(X_max_depth, X_max_polygone + X_range * marge)
            Y_max = max(Y_max_depth, Y_max_polygone + Y_range * marge)

        # Calculer les bas-fonds
        bas_fonds, _ = detecter_bas_fonds(grid_Z)

        # Calculer la surface des bas-fonds à l'intérieur des polygones
        surface_bas_fond_polygones = calculer_surface_bas_fonds_polygones(
            polygones_dans_emprise, bas_fonds, grid_X, grid_Y
        )

        # Affichage de la carte
        fig, ax = plt.subplots(figsize=(10, 10))
        generate_depth_map(ax, grid_Z, grid_X, grid_Y, X_min, X_max, Y_min, Y_max, label_rotation_x=0, label_rotation_y=-90)
        afficher_polygones(ax, polygones_dans_emprise)
        st.pyplot(fig)

        
        

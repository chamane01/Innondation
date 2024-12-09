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
import rasterio
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box, Polygon, MultiPolygon
from scipy.interpolate import griddata
import contextily as ctx

# Streamlit - Titre de l'application avec deux logos centrés
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.image("POPOPO.jpg", width=150)
with col2:
    st.image("logo.png", width=150)
with col3:
    st.write("")  # Colonne vide pour centrer les logos

st.title("Carte des zones inondées avec niveaux d'eau et surface")

# Initialisation des données
if "flood_data" not in st.session_state:
    st.session_state.flood_data = {
        "surface_bleue": None,
        "volume_eau": None,
        "niveau_inondation": 0.0,
    }

# Téléversement des fichiers GeoTIFF et GeoJSON
st.markdown("## Téléversez un fichier GeoTIFF et GeoJSON pour les analyses")
uploaded_tiff_file = st.file_uploader("Téléversez un fichier GeoTIFF (.tif)", type=["tif"])

# Fonction pour charger un GeoTIFF
def charger_tiff(fichier_tiff):
    try:
        with rasterio.open(fichier_tiff) as src:
            data = src.read(1)
            transform = src.transform
            crs = src.crs
            return data, transform, crs
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier GeoTIFF : {e}")
        return None, None, None


# Traitement des données GeoTIFF
if uploaded_tiff_file is not None:
    data_tiff, transform_tiff, crs_tiff = charger_tiff(uploaded_tiff_file)

    if data_tiff is not None:
        st.write("### Informations sur le fichier GeoTIFF :")
        st.write(f"Dimensions : {data_tiff.shape}")
        st.write(f"Valeurs min : {np.min(data_tiff)}, max : {np.max(data_tiff)}")
        st.write(f"Système de coordonnées : {crs_tiff}")

        # Affichage des données raster
        fig, ax = plt.subplots(figsize=(8, 6))
        extent = (
            transform_tiff[2], 
            transform_tiff[2] + transform_tiff[0] * data_tiff.shape[1],
            transform_tiff[5] + transform_tiff[4] * data_tiff.shape[0], 
            transform_tiff[5]
        )
        cax = ax.imshow(data_tiff, cmap="terrain", extent=extent)
        fig.colorbar(cax, ax=ax, label="Altitude (m)")
        ax.set_title("Carte d'altitude (GeoTIFF)")
        st.pyplot(fig)

        # Définir le niveau d'eau
        niveau_eau = st.number_input(
            "Entrez le niveau d'eau (mètres)",
            min_value=float(np.min(data_tiff)),
            max_value=float(np.max(data_tiff)),
            step=0.1,
        )
        st.session_state.flood_data["niveau_inondation"] = niveau_eau

        # Calcul de la zone inondée
        mask_inondee = data_tiff <= niveau_eau
        surface_inondee = np.sum(mask_inondee) * (transform_tiff[0] * abs(transform_tiff[4])) / 1e4  # En hectares

        st.write(f"Surface inondée : **{surface_inondee:.2f} hectares**")

        # Carte des zones inondées
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(mask_inondee, cmap="Blues", extent=extent, alpha=0.6)
        ax.set_title("Carte des zones inondées")
        ctx.add_basemap(ax, crs=crs_tiff.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
        st.pyplot(fig)

# Gestion des bâtiments
try:
    batiments_gdf = gpd.read_file("batiments2.geojson")
    if uploaded_tiff_file is not None:
        emprise = box(
            transform_tiff[2],
            transform_tiff[5] + transform_tiff[4] * data_tiff.shape[0],
            transform_tiff[2] + transform_tiff[0] * data_tiff.shape[1],
            transform_tiff[5],
        )
        batiments_dans_emprise = batiments_gdf[batiments_gdf.intersects(emprise)]
        st.write("### Bâtiments dans la zone analysée")
        st.map(batiments_dans_emprise)
except Exception as e:
    st.error(f"Erreur lors du chargement des bâtiments : {e}")

# (Code suivant déjà fourni et analysé dans la description.)


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

        
        

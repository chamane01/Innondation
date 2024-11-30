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
        return bas_fonds

    # Calcul des surfaces des bas-fonds
    def calculer_surface_bas_fond(bas_fonds, grid_X, grid_Y):
        """
        Calcule la surface des bas-fonds en hectares.
        """
        resolution = (grid_X[1, 0] - grid_X[0, 0]) * (grid_Y[0, 1] - grid_Y[0, 0]) / 10000  # Résolution en hectares
        surface_bas_fond = np.sum(bas_fonds) * resolution
        return surface_bas_fond

    bas_fonds = detecter_bas_fonds(grid_Z)
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


    # Ajouter des lignes pour relier les tirets
    for x in np.linspace(X_min, X_max, num=5):
        ax.axvline(x, color='black', linewidth=0.5, linestyle='--', alpha=0.2)
    for y in np.linspace(Y_min, Y_max, num=5):
        ax.axhline(y, color='black', linewidth=0.5, linestyle='--', alpha=0.2)

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
    legend_handles = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='cyan', markersize=10)]
    ax.legend(handles=legend_handles, labels=[f"Bas-fonds: {surface_bas_fond:.2f} ha"], loc='lower right', fontsize=10)
    # Afficher les surfaces calculées
    
# Ajouter un bouton pour générer la carte de profondeur
if st.button("Générer la carte de profondeur avec bas-fonds"):
    generate_depth_map(label_rotation_x=0, label_rotation_y=-90)


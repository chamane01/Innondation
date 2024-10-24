# Importer les bibliothèques nécessaires
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import Polygon
import contextily as ctx

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
        'surface_inondee': None,
        'volume_eau': None,
        'niveau_inondation': 0.0
    }

# Étape 1 : Sélectionner un site ou téléverser un fichier
st.markdown("## Sélectionner un site ou téléverser un fichier")

# Ajouter une option pour sélectionner parmi des fichiers CSV existants (AYAME 1 et AYAME 2)
option_site = st.selectbox(
    "Sélectionnez un site",
    ("Aucun", "AYAME 1", "AYAME 2")
)

# Téléverser un fichier Excel ou TXT
uploaded_file = st.file_uploader("Téléversez un fichier Excel ou TXT", type=["xlsx", "txt"])

# Fonction pour charger le fichier (identique pour les fichiers prédéfinis et téléversés)
def charger_fichier(fichier, is_uploaded=False):
    try:
        if is_uploaded:
            # Si le fichier est téléversé, vérifier son type
            if fichier.name.endswith('.xlsx'):
                df = pd.read_excel(fichier)
            elif fichier.name.endswith('.txt'):
                df = pd.read_csv(fichier, sep=",", header=None, names=["X", "Y", "Z"])
        else:
            # Si le fichier est prédéfini (site), il est déjà connu
            df = pd.read_csv(fichier, sep=",", header=None, names=["X", "Y", "Z"])
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {e}")
        return None

# Charger les fichiers selon la sélection
if option_site == "AYAME 1":
    df = charger_fichier('AYAME1.txt')
elif option_site == "AYAME 2":
    df = charger_fichier('AYAME2.csv')
elif uploaded_file is not None:
    df = charger_fichier(uploaded_file, is_uploaded=True)
else:
    st.warning("Veuillez sélectionner un site ou téléverser un fichier pour démarrer.")
    df = None

# Traitement des données si le fichier est chargé
if df is not None:
    # Séparateur pour organiser l'affichage
    st.markdown("---")  # Ligne de séparation

    # Vérification du fichier : s'assurer que les colonnes X, Y, Z sont présentes
    if 'X' not in df.columns or 'Y' not in df.columns or 'Z' not in df.columns:
        st.error("Erreur : colonnes 'X', 'Y' et 'Z' manquantes.")
    else:
        # Étape 5 : Paramètres du niveau d'inondation
        st.session_state.flood_data['niveau_inondation'] = st.number_input("Entrez le niveau d'eau (mètres)", min_value=0.0, step=0.1)
        interpolation_method = st.selectbox("Méthode d'interpolation", ['linear', 'nearest'])

        # Étape 6 : Création de la grille
        X_min, X_max = df['X'].min(), df['X'].max()
        Y_min, Y_max = df['Y'].min(), df['Y'].max()

        resolution = st.number_input("Résolution de la grille", value=300, min_value=100, max_value=1000)
        grid_X, grid_Y = np.mgrid[X_min:X_max:resolution*1j, Y_min:Y_max:resolution*1j]
        grid_Z = griddata((df['X'], df['Y']), df['Z'], (grid_X, grid_Y), method=interpolation_method)

        # Étape 7 : Calcul de la surface inondée
        def calculer_surface(niveau_inondation):
            contours = []
            for x in range(grid_X.shape[0]):
                for y in range(grid_Y.shape[1]):
                    if grid_Z[x, y] <= niveau_inondation:
                        contours.append((grid_X[x, y], grid_Y[x, y]))

            # Convertir les contours en un polygone
            if contours:
                polygon = Polygon(contours)
                return polygon, polygon.area / 10000  # Retourne le polygone et la surface en hectares
            return None, 0.0

        # Étape 8 : Calcul du volume d'eau
        def calculer_volume(surface_inondee):
            volume = surface_inondee * st.session_state.flood_data['niveau_inondation'] * 10000  # Conversion en m³ (1 hectare = 10,000 m²)
            return volume

        if st.button("Afficher la carte d'inondation"):
            # Étape 9 : Calcul de la surface et volume
            polygon_inonde, surface_inondee = calculer_surface(st.session_state.flood_data['niveau_inondation'])
            volume_eau = calculer_volume(surface_inondee)

            # Stocker les résultats dans session_state
            st.session_state.flood_data['surface_inondee'] = surface_inondee
            st.session_state.flood_data['volume_eau'] = volume_eau

            # Tracer la carte de profondeur
            fig, ax = plt.subplots(figsize=(8, 6))

            # Tracer le fond OpenStreetMap
            ax.set_xlim(X_min, X_max)
            ax.set_ylim(Y_min, Y_max)
            ctx.add_basemap(ax, crs="EPSG:32630", source=ctx.providers.OpenStreetMap.Mapnik)

            # Tracer la carte de profondeur
           # contourf = ax.contourf(grid_X, grid_Y, grid_Z, levels=100, cmap='viridis', alpha=0.5)
           # plt.colorbar(contourf, label='Profondeur (mètres)')

            # Tracer le contour du niveau d'inondation
            contours_inondation = ax.contour(grid_X, grid_Y, grid_Z, levels=[st.session_state.flood_data['niveau_inondation']], colors='red', linewidths=1)
            ax.clabel(contours_inondation, inline=True, fontsize=10, fmt='%1.1f m')
            # Tracé des hachures pour la zone inondée
            contourf_filled = ax.contourf(grid_X, grid_Y, grid_Z, 
                               levels=[-np.inf, st.session_state.flood_data['niveau_inondation']], 
                               colors='#007FFF', alpha=0.5)  # Couleur bleue semi-transparente


            # Tracer la zone inondée
          #  if polygon_inonde:
                #x_poly, y_poly = polygon_inonde.exterior.xy
                #ax.fill(x_poly, y_poly, alpha=0.5, fc='cyan', ec='black', lw=1, label='Zone inondée')  # Couleur cyan pour la zone inondée

            #ax.set_title("Carte des zones inondées")
           # ax.set_xlabel("Coordonnée X")
            #ax.set_ylabel("Coordonnée Y")
            #ax.legend()

            # Affichage de la carte
            st.pyplot(fig)

            # Affichage des résultats à droite de la carte
            col1, col2 = st.columns([3, 1])  # Créer deux colonnes
            with col2:
                st.write(f"**Surface inondée :** {st.session_state.flood_data['surface_inondee']:.2f} hectares")
                st.write(f"**Volume d'eau :** {st.session_state.flood_data['volume_eau']:.2f} m³")

# Etape 1 : Fonction pour charger les coordonnées locales et transformer les contours en polygones
def charger_coordonnees_locales(fichier_local):
    try:
        if fichier_local.name.endswith('.txt'):
            df_local = pd.read_csv(fichier_local, sep=",", header=None, names=["Local_X", "Local_Y", "Local_Z"])
        elif fichier_local.name.endswith('.csv'):
            df_local = pd.read_csv(fichier_local)
        else:
            st.error("Format de fichier non supporté. Veuillez téléverser un fichier TXT ou CSV.")
            return None
        return df_local
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier local : {e}")
        return None

# Etape 2 : Transformer les contours de la première carte en polygones
def transformer_contours_en_polygones(contours):
    polygons = []
    for contour in contours.collections:
        for path in contour.get_paths():
            vertices = path.vertices
            polygons.append(Polygon(vertices))
    return polygons

# Téléverser le fichier avec les coordonnées locales pour la deuxième carte
uploaded_local_file = st.file_uploader("Téléversez un fichier avec les coordonnées locales (TXT ou CSV)", type=["txt", "csv"])

# Si le fichier local est chargé, créer la deuxième carte
if uploaded_local_file is not None:
    df_local = charger_coordonnees_locales(uploaded_local_file)
    
    if df_local is not None and df is not None:
        st.markdown("---")  # Ligne de séparation

        # Etape 3 : Création de la deuxième carte avec les coordonnées locales
        if st.button("Afficher la deuxième carte avec coordonnées locales"):
            fig, ax = plt.subplots(figsize=(8, 6))

            # Extraire les contours de la première carte
            contours_inondation = ax.contour(grid_X, grid_Y, grid_Z, levels=[st.session_state.flood_data['niveau_inondation']], colors='red', linewidths=1)
            
            # Transformer les contours en polygones
            polygons = transformer_contours_en_polygones(contours_inondation)
            
            # Affichage des polygones sur la carte des coordonnées locales
            for polygon in polygons:
                if polygon is not None:
                    x_poly, y_poly = polygon.exterior.xy
                    ax.fill(x_poly, y_poly, alpha=0.5, fc='orange', ec='black', lw=1, label='Polygone extrait')

            # Tracer les coordonnées locales (si nécessaires)
            ax.scatter(df_local["Local_X"], df_local["Local_Y"], c='blue', marker='x', label="Coordonnées locales")

            ax.set_title("Deuxième carte avec contours transformés en polygonales")
            ax.set_xlabel("Coordonnée X locale")
            ax.set_ylabel("Coordonnée Y locale")
            ax.legend()

            # Afficher la carte finale
            st.pyplot(fig)


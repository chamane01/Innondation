# Importer les bibliothèques nécessaires
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import contextily as ctx
from shapely.geometry import Polygon, Point

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
    df = charger_fichier('AYAME2.txt')
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
        def calculer_surface_inondee(niveau_inondation):
            # Création des contours pour le niveau d'inondation donné
            contours_inondation = plt.contour(grid_X, grid_Y, grid_Z, levels=[niveau_inondation])

            # Récupérer les points des contours
            for contour in contours_inondation.collections:
                for path in contour.get_paths():
                    # Obtenir les points du contour
                    v = path.vertices
                    polygon_contour = Polygon(v)

                    # Masque pour les zones à l'intérieur du contour et en dessous du niveau d'inondation
                    masque_inonde = (grid_Z <= niveau_inondation)

                    # Filtrer les points inondés
                    points_inondes = np.column_stack((grid_X[masque_inonde], grid_Y[masque_inonde]))
                    points_in_polygon = [Point(pt).within(polygon_contour) for pt in points_inondes]
                    points_final = points_inondes[points_in_polygon]

                    if len(points_final) > 0:
                        # Créer un polygone de la zone inondée
                        polygone_inonde = Polygon(points_final).convex_hull

                        # Calcul de la surface en hectares
                        surface_inondee = polygone_inonde.area / 10000  # Conversion en hectares
                        return surface_inondee, polygone_inonde
            return 0.0, None

        # Étape 8 : Calcul du volume d'eau
        def calculer_volume(surface_inondee):
            # Volume = Surface x Hauteur (niveau d'eau)
            volume = surface_inondee * st.session_state.flood_data['niveau_inondation'] * 10000  # Conversion en m³ (1 hectare = 10,000 m²)
            return volume

        if st.button("Afficher la carte d'inondation"):
            # Étape 9 : Calcul de la surface et du volume
            surface_inondee, polygone_inonde = calculer_surface_inondee(st.session_state.flood_data['niveau_inondation'])
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

            # Tracer le contour du niveau d'inondation
            contours_inondation = ax.contour(grid_X, grid_Y, grid_Z, levels=[st.session_state.flood_data['niveau_inondation']], colors='red', linewidths=1)
            ax.clabel(contours_inondation, inline=True, fontsize=10, fmt='%1.1f m')

            # Tracé des hachures pour la zone inondée
            ax.contourf(grid_X, grid_Y, grid_Z, levels=[-np.inf, st.session_state.flood_data['niveau_inondation']], colors='#007FFF', alpha=0.5)

            # Tracer la zone inondée si le polygone est valide
            if polygone_inonde:
                x_poly, y_poly = polygone_inonde.exterior.xy
                ax.fill(x_poly, y_poly, alpha=0.5, fc='cyan', ec='black', lw=1, label='Zone inondée')

            ax.set_title("Carte des zones inondées")
            ax.set_xlabel("Coordonnée X")
            ax.set_ylabel("Coordonnée Y")
            ax.legend()

            # Affichage de la carte
            st.pyplot(fig)

            # Affichage des résultats à droite de la carte
            col1, col2 = st.columns([3, 1])  # Créer deux colonnes
            with col2:
                st.write(f"**Surface inondée :** {st.session_state.flood_data['surface_inondee']:.2f} ha")
                st.write(f"**Volume d'eau :** {st.session_state.flood_data['volume_eau']:.2f} m³")


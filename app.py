# Importer les bibliothèques nécessaires
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import Polygon
import contextily as ctx

# Charger les fichiers txt prédéfinis
def load_predefined_data(file_path):
    df = pd.read_csv(file_path, sep=",", header=None, names=["X", "Y", "Z"])
    return df

# Chemins vers les fichiers CSV prédéfinis dans votre repo
predefined_files = {
    "AYAME 1": "path/to/your/repo/AYAME1.csv",  # Mettez à jour le chemin réel
    "AYAME 2": "path/to/your/repo/AYAME22.csv"  # Mettez à jour le chemin réel
}

# Streamlit - Titre de l'application avec le logo centré
st.image("logo.png", width=150)
st.title("Carte des zones inondées avec niveaux d'eau et surface")

# Initialiser session_state pour stocker les données d'inondation
if 'flood_data' not in st.session_state:
    st.session_state.flood_data = {
        'surface_inondee': None,
        'volume_eau': None,
        'niveau_inondation': 0.0
    }

# Étape 1 : Sélectionner une carte existante ou téléverser un fichier
st.markdown("### Sélectionner une carte existante")

# Barre de sélection pour les cartes existantes
carte_selectionnee = st.selectbox("Choisir une carte existante", ["", "AYAME 1", "AYAME 2"])

# Charger les données de la carte sélectionnée ou option de téléversement
if carte_selectionnee:
    # Charger la carte prédéfinie à partir du dépôt
    df = load_predefined_data(predefined_files[carte_selectionnee])
    st.success(f"Carte '{carte_selectionnee}' chargée avec succès.")
else:
    # Option de téléversement d'un fichier
    st.markdown("---")  # Ligne de séparation
    st.markdown("### Ou téléversez un fichier")

    st.markdown(
        """
        <style>
        .stFileUploader label {
            color: white;
            background-color: #1E90FF; /* Bleu élégant */
            padding: 5px;
            border-radius: 5px;
        }
        .stSelectbox .st-bq {
            background: linear-gradient(135deg, #1E90FF, #87CEEB); /* Dégradé bleu */
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Téléversez un fichier Excel ou CSV", type=["xlsx", "csv"])

    if uploaded_file is not None:
        # Identifier le type de fichier et charger les données en fonction
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, sep=",", header=None, names=["X", "Y", "Z"])

# Vérification que le DataFrame est chargé
if 'df' in locals():
    # Étape 3 : Vérification du fichier
    if 'X' not in df.columns or 'Y' not in df.columns or 'Z' not in df.columns:
        st.error("Erreur : colonnes 'X', 'Y' et 'Z' manquantes.")
    else:
        # Étape 5 : Paramètres du niveau d'inondation
        st.session_state.flood_data['niveau_inondation'] = st.number_input("Entrez le niveau d'eau (mètres)", min_value=0.0, step=0.1)
        interpolation_method = st.selectbox("Méthode d'interpolation", ['linear', 'nearest'])

        # Étape 6 : Création de la grille
        try:
            # Définir les limites
            X_min, X_max = float(df['X'].min()), float(df['X'].max())
            Y_min, Y_max = float(df['Y'].min()), float(df['Y'].max())

            # Vérification des valeurs
            st.write(f"Valeurs X: min={X_min}, max={X_max}")
            st.write(f"Valeurs Y: min={Y_min}, max={Y_max}")

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
                contourf = ax.contourf(grid_X, grid_Y, grid_Z, levels=100, cmap='viridis', alpha=0.5)
                plt.colorbar(contourf, label='Profondeur (mètres)')

                # Tracer le contour du niveau d'inondation
                contours_inondation = ax.contour(grid_X, grid_Y, grid_Z, levels=[st.session_state.flood_data['niveau_inondation']], colors='red', linewidths=1)
                ax.clabel(contours_inondation, inline=True, fontsize=10, fmt='%1.1f m')

                # Tracer la zone inondée
                if polygon_inonde:
                    x_poly, y_poly = polygon_inonde.exterior.xy
                    ax.fill(x_poly, y_poly, alpha=0.5, fc='cyan', ec='black', lw=1, label='Zone inondée')  # Couleur cyan pour la zone inondée

                ax.set_title("Carte des zones inondées")
                ax.set_xlabel("Coordonnée X")
                ax.set_ylabel("Coordonnée Y")
                ax.legend()

                # Affichage de la carte
                st.pyplot(fig)

                # Affichage des résultats à droite de la carte
                col1, col2 = st.columns([3, 1])  # Créer deux colonnes
                with col2:
                    st.write(f"**Surface inondée :** {surface_inondee:.2f} hectares")
                    st.write(f"**Volume d'eau :** {volume_eau:.2f} m³")
        except Exception as e:
            st.error(f"Erreur lors de la création de la grille : {e}")
else:
    st.warning("Veuillez choisir une carte ou téléverser un fichier pour continuer.")

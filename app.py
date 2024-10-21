# Importer les bibliothèques nécessaires
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from shapely.geometry import Polygon
import plotly.graph_objects as go

# Streamlit - Titre de l'application
st.title("Carte des zones inondées avec contours fermés et calcul de surface")

# Étape 1 : Téléverser le fichier Excel ou TXT
uploaded_file = st.file_uploader("Téléversez un fichier Excel ou TXT", type=["xlsx", "txt"])

if uploaded_file is not None:
    # Étape 2 : Identifier le type de fichier et charger les données en fonction
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.txt'):
        df = pd.read_csv(uploaded_file, sep=",", header=None, names=["X", "Y", "Z"])

    # Étape 3 : Vérification du fichier
    if 'X' not in df.columns or 'Y' not in df.columns or 'Z' not in df.columns:
        st.error("Erreur : colonnes 'X', 'Y' et 'Z' manquantes.")
    else:
        # Affichage de la disposition des points dans une vue de haut (XY)
        st.subheader("Vue de haut des points du fichier CAD")
        
        fig_view = go.Figure(data=go.Scatter(x=df['X'], y=df['Y'], mode='markers', marker=dict(color='blue')))
        fig_view.update_layout(title="Vue de haut des points (XY)", xaxis_title="Coordonnée X", yaxis_title="Coordonnée Y")
        st.plotly_chart(fig_view)

        # Étape 5 : Paramètres du niveau d'inondation
        niveau_inondation = st.number_input("Entrez le niveau d'eau (mètres)", min_value=0.0, step=0.1)
        interpolation_method = st.selectbox("Méthode d'interpolation", ['linear', 'nearest'])

        # Étape 6 : Création de la grille
        X_min, X_max = df['X'].min(), df['X'].max()
        Y_min, Y_max = df['Y'].min(), df['Y'].max()

        resolution = st.number_input("Résolution de la grille", value=300, min_value=100, max_value=1000)
        grid_X, grid_Y = np.mgrid[X_min:X_max:resolution*1j, Y_min:Y_max:resolution*1j]
        grid_Z = griddata((df['X'], df['Y']), df['Z'], (grid_X, grid_Y), method=interpolation_method)

        # Étape 7 : Génération des contours fermés
        def generer_polygones_fermes(niveau_inondation):
            contour = plt.contour(grid_X, grid_Y, grid_Z, levels=[niveau_inondation], colors='none')
            paths = contour.collections[0].get_paths()

            polygones = []
            for path in paths:
                # Extraire les points du contour
                vertices = path.vertices
                if len(vertices) > 2:
                    # Si les polygones ne sont pas fermés, on ferme manuellement
                    if not np.array_equal(vertices[0], vertices[-1]):
                        vertices = np.vstack([vertices, vertices[0]])
                    polygon = Polygon(vertices)
                    if polygon.is_valid and polygon.area > 0:
                        polygones.append(polygon)

            return polygones

        polygones_inondes = generer_polygones_fermes(niveau_inondation)

        # Étape 8 : Affichage de la carte avec les contours fermés et hachures
        def plot_map_with_hatching(niveau_inondation):
            plt.close('all')

            # Taille ajustée pour la carte
            fig, ax_map = plt.subplots(figsize=(8, 6))

            # Tracé de la carte de profondeur
            contour = ax_map.contourf(grid_X, grid_Y, grid_Z, cmap='viridis', levels=100)
            cbar = fig.colorbar(contour, ax=ax_map)
            cbar.set_label('Profondeur (mètres)')

            # Tracé du contour actuel du niveau d'inondation
            contours_inondation = ax_map.contour(grid_X, grid_Y, grid_Z, levels=[niveau_inondation], colors='red', linewidths=2)
            ax_map.clabel(contours_inondation, inline=True, fontsize=10, fmt='%1.1f m')

            # Tracé des hachures pour la zone inondée
            ax_map.contourf(grid_X, grid_Y, grid_Z, levels=[-np.inf, niveau_inondation], colors='none', hatches=['///'], alpha=0)

            ax_map.set_title("Carte des zones inondées avec hachures")
            ax_map.set_xlabel("Coordonnée X")
            ax_map.set_ylabel("Coordonnée Y")

            # Affichage
            st.pyplot(fig)

        if st.button("Afficher la carte"):
            plot_map_with_hatching(niveau_inondation)

        # Étape 9 : Extraction des coordonnées XY pour chaque polygonale
        def extraire_coordonnees_XY(polygones):
            tables_coordonnees = []
            for polygon in polygones:
                coords = np.array(polygon.exterior.coords)
                table = pd.DataFrame({"X": coords[:, 0], "Y": coords[:, 1]})
                tables_coordonnees.append(table)
            return tables_coordonnees

        tables_coordonnees = extraire_coordonnees_XY(polygones_inondes)

        # Affichage des tableaux de coordonnées pour chaque polygonale sur demande
        if st.checkbox("Afficher les tableaux de coordonnées des polygonales"):
            st.subheader("Coordonnées des polygonales")
            for i, table in enumerate(tables_coordonnees):
                st.write(f"Polygonale {i+1}")
                st.dataframe(table)

        # Étape 10 : Calcul de la surface en utilisant les coordonnées projetées
        def calcul_surface_topographique(table):
            x = table['X'].values
            y = table['Y'].values
            n = len(x)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += x[i] * y[j] - x[j] * y[i]
            return abs(area) / 2.0

        surfaces_polygonales = [calcul_surface_topographique(table) for table in tables_coordonnees]

        # Étape 11 : Affichage des surfaces calculées
        st.subheader("Surface des zones inondées (formule topographique)")
        for i, surface in enumerate(surfaces_polygonales):
            st.write(f"Surface de la polygonale {i+1} : {surface:.2f} mètres carrés")
        
        surface_totale = sum(surfaces_polygonales)
        st.write(f"Surface totale : {surface_totale / 10000:.2f} hectares")

else:
    st.warning("Veuillez téléverser un fichier pour démarrer.")

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import Polygon
import leafmap.foliumap as leafmap  # Bibliothèque pour la carte dynamique

# Chemin de l'image du logo
logo_path = "POPOPO.jpg"  # Assurez-vous que l'image est bien dans le même répertoire que ce fichier app.py

# Fonction pour afficher le logo
def afficher_logo():
    try:
        logo_image = Image.open(logo_path)
        st.image(logo_image, width=200)  # Ajustez la largeur selon vos besoins
    except FileNotFoundError:
        st.error("Le fichier du logo n'a pas été trouvé. Veuillez vérifier le chemin.")

# Fonction pour tracer la carte avec contours actuels et hachures
def plot_map_with_hatching(niveau_inondation, surface_inondee, volume_eau, previous_contours=None, rapport_comparatif=None):
    plt.close('all')

    # Taille ajustée pour la carte
    fig, (ax_map, ax_report) = plt.subplots(1, 2, figsize=(14, 8))

    # Tracé de la carte de profondeur
    contour = ax_map.contourf(grid_X, grid_Y, grid_Z, cmap='viridis', levels=100)
    cbar = fig.colorbar(contour, ax=ax_map)
    cbar.set_label('Profondeur (mètres)')

    # Tracé du contour actuel du niveau d'inondation
    contours_inondation = ax_map.contour(grid_X, grid_Y, grid_Z, levels=[niveau_inondation], colors='red', linewidths=2)
    ax_map.clabel(contours_inondation, inline=True, fontsize=10, fmt='%1.1f m')

    # Tracé des hachures pour la zone inondée
    ax_map.contourf(grid_X, grid_Y, grid_Z, levels=[-np.inf, niveau_inondation], colors='none', hatches=['///'], alpha=0)

    # Tracé du contour précédent (avec opacité réduite)
    if previous_contours is not None:
        ax_map.contour(grid_X, grid_Y, grid_Z, levels=[previous_contours], colors='blue', linewidths=2, alpha=0.5)

    ax_map.set_title("Carte des zones inondées avec hachures")
    ax_map.set_xlabel("Coordonnée X")
    ax_map.set_ylabel("Coordonnée Y")

    # Affichage du rapport dans un cadre
    rapport = (f"Niveau d'inondation : {niveau_inondation} m\n"
               f"Surface inondée : {surface_inondee:.2f} hectares\n"
               f"Volume d'eau : {volume_eau:.2f} m³")
    if rapport_comparatif:
        rapport += f"\n\nComparaison :\n{rapport_comparatif}"

    # Positionnement du texte en haut à gauche
    ax_report.text(0.01, 0.99, rapport, fontsize=12, ha='left', va='top', transform=ax_report.transAxes)

    # Ajout de la légende
    ax_report.text(0.01, 0.60, "Légende:\n"
                     "- Hachures: Zone inondée\n"
                     "- Ligne rouge: Niveau d'inondation", fontsize=10, ha='left', va='top', transform=ax_report.transAxes)

    ax_report.axis('off')

    plt.tight_layout()
    st.pyplot(fig)  # Utilisation de Streamlit pour afficher la carte

# Carte dynamique avec différents fonds de carte
def afficher_carte_dynamique(df):
    st.subheader("Carte dynamique")
    m = leafmap.Map(center=[df['Y'].mean(), df['X'].mean()], zoom=10)

    # Ajout des couches de base
    m.add_basemap("SATELLITE")  # Carte satellite
    m.add_basemap("OSM")        # OpenStreetMap
    m.add_basemap("HYBRID")     # Carte hybride

    # Ajout des points depuis le dataframe
    for _, row in df.iterrows():
        m.add_marker([row['Y'], row['X']], popup=f"Point Z: {row['Z']}")

    m.to_streamlit(height=500)

# Fonction principale de l'application
def main():
    st.title("Application de Gestion des Inondations")
    
    # Afficher le logo
    afficher_logo()
    
    # Étape 1 : Téléverser le fichier Excel ou TXT
    uploaded_file = st.file_uploader("Téléverser un fichier Excel (.xlsx) ou TXT (.txt)", type=['xlsx', 'txt'])

    if uploaded_file is not None:
        # Étape 2 : Charger les données en fonction du type de fichier
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.txt'):
            df = pd.read_csv(uploaded_file, sep=",", header=None, names=["X", "Y", "Z"])

        # Vérifier la présence des colonnes nécessaires
        if 'X' in df.columns and 'Y' in df.columns and 'Z' in df.columns:
            st.success("Données chargées avec succès !")

            # Afficher les premières lignes du dataframe
            st.write(df.head())

            # Étape 5 : Paramètres du niveau d'inondation
            niveau_inondation = st.number_input("Entrez le niveau d'eau (en mètres)", min_value=float(df['Z'].min()), max_value=float(df['Z'].max()), value=float(df['Z'].mean()))

            # Étape 6 : Créer la grille
            X_min, X_max = df['X'].min(), df['X'].max()
            Y_min, Y_max = df['Y'].min(), df['Y'].max()

            resolution = 300  # Résolution fixe

            global grid_X, grid_Y, grid_Z  # Global pour être utilisé dans d'autres fonctions
            grid_X, grid_Y = np.mgrid[X_min:X_max:resolution*1j, Y_min:Y_max:resolution*1j]
            grid_Z = griddata((df['X'], df['Y']), df['Z'], (grid_X, grid_Y), method='linear')

            # Étape 7 : Calculer la surface inondée
            def calculer_surface(niveau_inondation):
                contour = plt.contour(grid_X, grid_Y, grid_Z, levels=[niveau_inondation], colors='none')
                paths = contour.collections[0].get_paths()
                surfaces = [Polygon(path.vertices).area for path in paths]
                return sum(surfaces) / 10000  # Retourne en hectares

            # Étape 8 : Calcul du volume d'eau
            def calculer_volume(niveau_inondation, surface_inondee):
                volume = surface_inondee * niveau_inondation * 10000  # Conversion en m³ (1 hectare = 10,000 m²)
                return volume

            surface_inondee = calculer_surface(niveau_inondation)
            volume_eau = calculer_volume(niveau_inondation, surface_inondee)

            # Étape 9 : Affichage de la carte avec hachures
            plot_map_with_hatching(niveau_inondation, surface_inondee, volume_eau)

            # Carte dynamique avec les points
            afficher_carte_dynamique(df)

        else:
            st.error("Le fichier doit contenir les colonnes 'X', 'Y' et 'Z'.")
    else:
        st.warning("Veuillez téléverser un fichier.")

if __name__ == "__main__":
    main()

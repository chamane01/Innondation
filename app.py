import streamlit as st
import os
import rasterio
import rasterio.merge
import folium
import math
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
from folium.plugins import Draw
from folium import MacroElement
from jinja2 import Template

def load_tiff_files(folder_path):
    """Charge les fichiers TIFF contenus dans un dossier."""
    try:
        tiff_files = [os.path.join(folder_path, f) 
                      for f in os.listdir(folder_path) if f.lower().endswith('.tif')]
    except Exception as e:
        st.error(f"Erreur lors de la lecture du dossier {folder_path}: {e}")
        return []
    
    if not tiff_files:
        st.error("Aucun fichier TIFF trouvé dans le dossier.")
        return []
    
    return [f for f in tiff_files if os.path.exists(f)]

def build_mosaic(tiff_files, mosaic_path="mosaic.tif"):
    """Construit une mosaïque à partir de fichiers TIFF."""
    try:
        src_files = [rasterio.open(fp) for fp in tiff_files]
        mosaic, out_trans = rasterio.merge.merge(src_files)
        out_meta = src_files[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans
        })
        with rasterio.open(mosaic_path, "w", **out_meta) as dest:
            dest.write(mosaic)
        for src in src_files:
            src.close()
        return mosaic_path
    except Exception as e:
        st.error(f"Erreur lors de la création de la mosaïque : {e}")
        return None

class VertexMarker(MacroElement):
    """
    MacroElement qui injecte un script JS afin d'ajouter des marqueurs
    sur les points de cassure (sommets) des polylignes dessinées.
    """
    _template = Template("""
        {% macro script(this, kwargs) %}
        // Ajout de marqueurs sur les sommets des polylignes dès leur création
        {{this._parent.get_name()}}.on('draw:created', function(e) {
            var layer = e.layer;
            if (layer instanceof L.Polyline) {
                var latlngs = layer.getLatLngs();
                for (var i = 0; i < latlngs.length; i++) {
                    L.marker(latlngs[i]).addTo({{this._parent.get_name()}});
                }
            }
        });
        {% endmacro %}
    """)

    def render(self, **kwargs):
        super().render(**kwargs)

def create_map(mosaic_file):
    """Crée une carte Folium avec :
      - Un groupe de couches nommé 'élevation cote d'ivoire' affichant l'emprise de la mosaïque,
      - L'outil de dessin limité aux polylignes,
      - Le MacroElement pour marquer les sommets des polylignes,
      - Le gestionnaire de couches.
    """
    m = folium.Map(location=[0, 0], zoom_start=2)
    
    # Groupe de couches pour la mosaïque
    mosaic_group = folium.FeatureGroup(name="élevation cote d'ivoire")
    try:
        with rasterio.open(mosaic_file) as src:
            bounds = src.bounds
            folium.Rectangle(
                bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                color='blue', 
                fill=False,
                tooltip="Emprise de la mosaïque"
            ).add_to(mosaic_group)
    except Exception as e:
        st.error(f"Erreur lors de l'ouverture de la mosaïque : {e}")
    mosaic_group.add_to(m)
    
    # Ajout de l'outil de dessin (uniquement pour les polylignes)
    Draw(
        draw_options={
            'polyline': {'allowIntersection': False},
            'polygon': False,
            'circle': False,
            'marker': False,
            'circlemarker': False,
            'rectangle': False
        },
        edit_options={'edit': True, 'remove': True}
    ).add_to(m)
    
    # Ajout du MacroElement pour marquer les points de cassure des polylignes
    m.add_child(VertexMarker())
    
    # Ajout du gestionnaire de couches
    folium.LayerControl().add_to(m)
    
    return m

def haversine(lon1, lat1, lon2, lat2):
    """Calcule la distance en mètres entre deux points GPS."""
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371000  # Rayon de la Terre en mètres
    return c * r

def interpolate_line(coords, step=50):
    """Interpole des points sur une ligne."""
    if len(coords) < 2:
        return coords, [0]
    sampled_points = [coords[0]]
    cumulative_dist = [0]
    for i in range(len(coords)-1):
        start = coords[i]
        end = coords[i+1]
        seg_distance = haversine(start[0], start[1], end[0], end[1])
        num_steps = max(int(seg_distance // step), 1)
        for j in range(1, num_steps+1):
            fraction = j / num_steps
            lon = start[0] + fraction * (end[0] - start[0])
            lat = start[1] + fraction * (end[1] - start[1])
            dist = haversine(sampled_points[-1][0], sampled_points[-1][1], lon, lat)
            sampled_points.append([lon, lat])
            cumulative_dist.append(cumulative_dist[-1] + dist)
    return sampled_points, cumulative_dist

def main():
    st.title("Carte Dynamique avec Profils d'Élévation")
    
    # Création du menu latéral (sidebar) avec différentes options
    option = st.sidebar.selectbox("Choisissez une option", 
                                  ["Tracer des profils", "Option 2 (en cours de développement)", "Option 3 (en cours de développement)"])
    
    if option == "Tracer des profils":
        # Configuration du nom de la carte
        map_name = st.text_input("Nom de votre carte", value="Ma Carte")
    
        # Gestion des fichiers TIFF
        folder_path = "TIFF"
        if not os.path.exists(folder_path):
            st.error("Dossier TIFF introuvable")
            return
    
        tiff_files = load_tiff_files(folder_path)
        if not tiff_files:
            return
    
        mosaic_path = build_mosaic(tiff_files)
        if not mosaic_path:
            return
    
        # Création de la carte interactive
        m = create_map(mosaic_path)
        st.write("**Dessinez des polylignes pour générer des profils d'élévation. Les sommets (cassures) seront automatiquement marqués.**")
    
        # Affichage de la carte et récupération des dessins
        map_data = st_folium(m, width=700, height=500)
    
        # Initialisation des profils
        if "profiles" not in st.session_state:
            st.session_state.profiles = []
    
        # Extraction des dessins réalisés (uniquement les polylignes)
        current_drawings = []
        if isinstance(map_data, dict):
            raw_drawings = map_data.get("all_drawings", [])
            if isinstance(raw_drawings, list):
                current_drawings = [
                    d for d in raw_drawings 
                    if isinstance(d, dict) and d.get("geometry", {}).get("type") == "LineString"
                ]
    
        # Mise à jour des profils en fonction des dessins
        st.session_state.profiles = [{
            "coords": d["geometry"]["coordinates"],
            "name": f"{map_name} - Ligne {i+1}"
        } for i, d in enumerate(current_drawings)]
    
        # Affichage des profils générés avec leur graphique d'élévation
        if st.session_state.profiles:
            st.subheader("Profils générés")
            for i, profile in enumerate(st.session_state.profiles):
                col1, col2 = st.columns([1, 4])
                with col1:
                    new_name = st.text_input(
                        "Nom du profil", 
                        value=profile["name"],
                        key=f"profile_name_{i}"
                    )
                    profile["name"] = new_name
                with col2:
                    try:
                        # Calcul des élévations
                        points, distances = interpolate_line(profile["coords"])
                        with rasterio.open(mosaic_path) as src:
                            elevations = [list(src.sample([p]))[0][0] for p in points]
                        
                        # Création du graphique du profil d'élévation
                        fig, ax = plt.subplots(figsize=(8, 3))
                        ax.plot(distances, elevations, 'b-', linewidth=1.5)
                        ax.set_title(profile["name"])
                        ax.set_xlabel("Distance (m)")
                        ax.set_ylabel("Altitude (m)")
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Erreur de traitement : {e}")
    else:
        st.info("Cette option est en cours de développement.")

if __name__ == "__main__":
    main()

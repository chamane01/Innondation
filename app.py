# Mode "Tracer des profils"
if st.session_state.mode == "profiles":
    st.subheader("Tracer des profils")
    
    # Extraction des dessins de type ligne sur la carte
    current_drawings = []
    if isinstance(map_data, dict):
        raw_drawings = map_data.get("all_drawings", [])
        # Assurez-vous que raw_drawings est une liste
        if not isinstance(raw_drawings, list):
            raw_drawings = []
        current_drawings = [
            d for d in raw_drawings 
            if isinstance(d, dict) and d.get("geometry", {}).get("type") == "LineString"
        ]
    else:
        st.info("Aucun dessin n'a été détecté. Utilisez l'outil de dessin sur la carte pour tracer une ligne.")
    
    # Transformation des lignes dessinées en profils
    profiles = []
    for i, d in enumerate(current_drawings):
        profiles.append({
            "coords": d["geometry"]["coordinates"],
            "name": f"{map_name} - Profil {i+1}"
        })
    
    # Affichage des profils générés
    if profiles:
        st.write("Les profils dessinés sont affichés ci-dessous. Toute nouvelle droite dessinée sera prise en compte automatiquement.")
        for i, profile in enumerate(profiles):
            st.markdown(f"#### {profile['name']}")
            col_a, col_b = st.columns([1, 4])
            with col_a:
                # Permet de modifier le nom du profil
                new_name = st.text_input(
                    "Nom du profil", 
                    value=profile["name"],
                    key=f"profile_name_{i}"
                )
                profile["name"] = new_name
                
                # Option de présentation du profil
                presentation_mode = st.radio(
                    "Mode de présentation",
                    ("Automatique", "Manuel"),
                    key=f"presentation_mode_{i}"
                )
                manual_options = {}
                if presentation_mode == "Manuel":
                    manual_options["ecart_distance"] = st.number_input(
                        "Ecart distance (m)",
                        min_value=1.0,
                        value=50.0,
                        step=1.0,
                        key=f"ecart_distance_{i}"
                    )
                    manual_options["ecart_altitude"] = st.number_input(
                        "Ecart altitude (m)",
                        min_value=1.0,
                        value=10.0,
                        step=1.0,
                        key=f"ecart_altitude_{i}"
                    )
            with col_b:
                try:
                    points, distances = interpolate_line(profile["coords"])
                    with rasterio.open(mosaic_path) as src:
                        elevations = [list(src.sample([p]))[0][0] for p in points]
                    
                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.plot(distances, elevations, 'b-', linewidth=1.5)
                    ax.set_title(profile["name"])
                    ax.set_xlabel("Distance (m)")
                    ax.set_ylabel("Altitude (m)")
                    
                    # En mode manuel, on fixe les ticks en fonction des écarts indiqués
                    if presentation_mode == "Manuel":
                        ecart_distance = manual_options.get("ecart_distance", 50.0)
                        ecart_altitude = manual_options.get("ecart_altitude", 10.0)
                        xticks = np.arange(0, max(distances) + ecart_distance, ecart_distance)
                        yticks = np.arange(min(elevations), max(elevations) + ecart_altitude, ecart_altitude)
                        ax.set_xticks(xticks)
                        ax.set_yticks(yticks)
                    
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Erreur de traitement : {e}")
    else:
        st.info("Aucun profil dessiné pour le moment. Utilisez l'outil de dessin sur la carte pour tracer une ligne.")
    
    # Bouton "Retour" pour revenir au menu principal
    if st.button("Retour", key="retour_profiles"):
        st.session_state.mode = "none"

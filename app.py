import streamlit as st

# Titre de l'application
st.title("Tableau de Bord Ageroute Côte d'Ivoire")

# Section 1: Carte Interactive des Routes
st.header("Carte Interactive de Toutes les Routes")
st.write("En cours de développement")  # Placeholder pour la carte interactive

# Section 2: Statistiques des Défauts
st.header("Statistiques des Défauts")
selected_month = st.selectbox("Sélectionner un mois", ["Jan", "Fév", "Mar", "Avr"])
if selected_month:
    st.write(f"Statistiques pour {selected_month}:")
    st.write("- Fissures: En cours de développement")
    st.write("- Nids-de-poule: En cours de développement")
    st.write("- Usures: En cours de développement")

# Section 3: Voir Toutes les Routes
st.header("Voir Toutes les Routes")
if st.button("Afficher toutes les routes"):
    st.write("Liste des routes:")
    st.write("- A1 - Autoroute du Nord (Abidjan-Yamoussoukro)")
    st.write("- A3 - Autoroute Abidjan-Grand Bassam")
    st.write("- Nationale A1 (Yamoussoukro-Bouaké)")
    st.write("- Nationale A3 (Abidjan-Adzopé)")
    st.write("- Pont HKB")

# Section 4: Inspections Réalisées
st.header("Inspections Réalisées")
st.write("Statistiques des 5 derniers mois: En cours de développement")

# Section 5: Génération de Rapports
st.header("Générer des Rapports")
report_type = st.radio("Type de rapport", ["Journalier", "Mensuel", "Annuel"])
if st.button("Générer le rapport"):
    st.write(f"Génération du rapport {report_type.lower()} en cours de développement")

# Section 6: Alertes
st.header("Alertes")
st.write("Nid-de-poule dangereux détecté sur l’Autoroute du Nord (Section Yamoussoukro-Bouaké)")
st.write("Fissures multiples sur le Pont HKB à Abidjan")
st.write("Usures importantes sur la Nationale A3 (Abidjan-Adzopé)")

# Section 7: Inspections par Date
st.header("Inspections par Date")
selected_date = st.date_input("Sélectionner une date")
if st.button("Voir les inspections pour cette date"):
    st.write(f"Inspections pour {selected_date}: En cours de développement")

# Section 8: Options de Rapport
st.header("Options de Rapport")
if st.button("Voir statistiques par route"):
    st.write("Statistiques par route: En cours de développement")
if st.button("Voir toutes les statistiques d’inspection"):
    st.write("Toutes les statistiques d’inspection: En cours de développement")

# Pied de page
st.sidebar.image("logo_gouvernement.png", width=100)
st.sidebar.write("Gouvernement")
st.sidebar.write("Agéroute")

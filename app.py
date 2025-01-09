import streamlit as st
import pandas as pd

# Titre de l'application
st.title("Tableau de Bord Ageroute Côte d'Ivoire")

# Section 1: Carte des Défauts
st.header("Carte des Défauts")
st.subheader("Carte Interactive de Toutes les routes")
st.write("Carte par Routes")
st.write("Voir Toutes les Routes")

# Section 2: Statistiques des Défauts
st.header("Statistiques des Défauts")
st.subheader("Mois")

# Données des statistiques
data = {
    "Mois": ["Jan", "Fev", "Mar", "Avr"],
    "Fissures": [40, 30, 20, 27],
    "Nids-de-poule": [24, 13, 98, 39],
    "Usures": [24, 22, 22, 20]
}

df = pd.DataFrame(data)
st.dataframe(df)

# Section 3: Inspections Réalisées
st.header("Inspections Réalisées")
st.subheader("Statistiques des 5 derniers mois")

# Données des inspections
inspections_data = {
    "Mois": ["Mai", "Juin", "Juliet", "Aout", "Septembre"],
    "Inspections": [55, 180, 220, 150, 200]
}

inspections_df = pd.DataFrame(inspections_data)
st.dataframe(inspections_df)

# Section 4: Alertes
st.header("Alertes")
st.write("Nid-de-poule dangereux détecté sur l’Autoroute du Nord (Section Yamoussoukro-Bouaké)")
st.write("Fissures multiples sur le Pont HKB à Abidjan")
st.write("Usures importantes sur la Nationale A3 (Abidjan-Adzopé)")

# Section 5: Rapports
st.header("Rapports")
st.write("Générer rapport journalier")
st.write("Générer rapport mensuel")
st.write("Générer rapport annuel")

# Section 6: Toutes les Routes
st.header("Toutes les routes")
st.write("A1 - Autoroute du Nord (Abidjan-Yamoussoukro)")
st.write("A3 - Autoroute Abidjan-Grand Bassam")
st.write("Nationale A1 (Yamoussoukro-Bouaké)")
st.write("Nationale A3 (Abidjan-Adzopé)")
st.write("Pont HKB")

# Section 7: Statistiques par Routes
st.header("Statistiques par Routes")
st.write("Rapport par Routes")

# Footer
st.write("Logo Gouvernement")
st.write("Agéroute")

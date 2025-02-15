import streamlit as st
import pandas as pd
from io import BytesIO

# Configuration de la page
st.set_page_config(page_title="Générateur de Rapports", layout="wide")

# Fonction pour convertir le DataFrame en fichier Excel
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()
    processed_data = output.getvalue()
    return processed_data

# Interface utilisateur
st.title("📊 Générateur de Rapports Automatisés")

# Téléchargement du fichier
uploaded_file = st.file_uploader("Importer un fichier (CSV ou Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Lecture des données
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Sélection des colonnes
    selected_columns = st.multiselect("Sélectionner les colonnes à inclure", df.columns)
    
    if selected_columns:
        df_preview = df[selected_columns]
        
        # Section de configuration du rapport
        with st.expander("⚙️ Configuration du rapport"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Sélection des colonnes de regroupement
                groupby_columns = st.multiselect("Regrouper par", selected_columns)
            
            with col2:
                # Sélection des colonnes d'agrégation
                numeric_columns = df_preview.select_dtypes(include=['number']).columns.tolist()
                aggregation_columns = st.multiselect("Colonnes à analyser", numeric_columns)
                
                # Sélection des fonctions d'agrégation
                agg_functions = {
                    'Somme': 'sum',
                    'Moyenne': 'mean',
                    'Minimum': 'min',
                    'Maximum': 'max',
                    'Count': 'count'
                }
                selected_aggs = st.multiselect("Fonctions d'agrégation", list(agg_functions.keys()))
        
        # Génération du rapport
        if st.button("🔄 Générer le rapport"):
            try:
                if groupby_columns and aggregation_columns and selected_aggs:
                    # Création des agrégations
                    aggregation = {}
                    for col in aggregation_columns:
                        aggregation[col] = [agg_functions[agg] for agg in selected_aggs]
                    
                    # Génération du rapport
                    report = df_preview.groupby(groupby_columns).agg(aggregation)
                else:
                    report = df_preview.describe().T
                
                # Prévisualisation
                st.subheader("📄 Prévisualisation du rapport")
                st.dataframe(report.style.background_gradient(cmap='Blues'), use_container_width=True)
                
                # Exportation
                st.subheader("📤 Exporter le rapport")
                export_format = st.radio("Format d'export", ['CSV', 'Excel'])
                
                if export_format == 'CSV':
                    csv = report.to_csv(index=True).encode('utf-8')
                    st.download_button(
                        label="Télécharger CSV",
                        data=csv,
                        file_name='rapport.csv',
                        mime='text/csv'
                    )
                else:
                    excel_data = to_excel(report)
                    st.download_button(
                        label="Télécharger Excel",
                        data=excel_data,
                        file_name='rapport.xlsx',
                        mime='application/vnd.ms-excel'
                    )
                
            except Exception as e:
                st.error(f"Une erreur est survenue : {str(e)}")

    # Aperçu des données brutes
    with st.expander("👀 Aperçu des données brutes"):
        st.dataframe(df_preview.head(10), use_container_width=True)
else:
    st.info("Veuillez importer un fichier pour commencer")

# Instructions
st.markdown("---")
st.markdown("""
### Instructions d'utilisation :
1. Importer un fichier CSV ou Excel
2. Sélectionner les colonnes à inclure dans le rapport
3. Configurer les regroupements et agrégations souhaités
4. Prévisualiser le rapport généré
5. Exporter dans le format souhaité
""")

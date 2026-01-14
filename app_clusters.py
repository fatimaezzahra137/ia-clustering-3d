import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="IA Collecte de Données", layout="wide")
st.title("🛍️ Analyseur avec Collecte des Tests")

# 1. Chargement du modèle
model = joblib.load("modele_clusters_3d.pkl")

# 2. INITIALISATION DE LA MÉMOIRE (Session State)
# Cela permet de garder les données même quand la page se rafraîchit
if 'historique_tests' not in st.session_state:
    # On commence avec les données de base
    data_depart = {
        'depenses': [120, 3000, 600, 5000],
        'visites': [1, 10, 5, 20],
        'anciennete': [2, 24, 12, 48]
    }
    st.session_state.historique_tests = pd.DataFrame(data_depart)

# 3. Barre latérale : Nouveau Test
st.sidebar.header("🔍 Saisie Nouveau Client")
dep = st.sidebar.number_input("Dépenses (€)", 0, 10000, 500)
vis = st.sidebar.number_input("Visites / mois", 0, 30, 5)
anc = st.sidebar.number_input("Ancienneté (Mois)", 0, 120, 12)

if st.sidebar.button("Déterminer le Segment"):
    # A. Prédiction
    nouveau_client = pd.DataFrame([[dep, vis, anc]], columns=['depenses', 'visites', 'anciennete'])
    res = model.predict(nouveau_client)[0]
    
    # B. AJOUT À LA MÉMOIRE
    # On ajoute le nouveau test à notre tableau stocké dans la session
    st.session_state.historique_tests = pd.concat([st.session_state.historique_tests, nouveau_client], ignore_index=True)
    
    noms = {1: "Économe 📉", 0: "Standard 📊", 2: "VIP 🏆"}
    st.sidebar.success(f"Résultat : **{noms.get(res)}** (Ajouté au tableau !)")

# 4. Calcul des segments pour TOUT le tableau (Base + Tests)
df_final = st.session_state.historique_tests.copy()
df_final['ID_Segment'] = model.predict(df_final)
mapping = {1: "Économe", 0: "Standard", 2: "VIP"}
df_final['Nom_Segment'] = df_final['ID_Segment'].map(mapping)

# 5. Graphique 3D Dynamique
fig = px.scatter_3d(df_final, x='depenses', y='visites', z='anciennete', color='Nom_Segment',
                    color_discrete_map={"Économe": "#EF553B", "Standard": "#636EFA", "VIP": "#00CC96"},
                    template="plotly_dark")
st.plotly_chart(fig, width="stretch")

# 6. Export Excel (Contiendra maintenant tes tests !)
st.subheader("📥 Exporter la base (Données initiales + Vos tests)")

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Base_Complete')
    return output.getvalue()

st.download_button(label="📥 Télécharger Excel", data=to_excel(df_final), file_name="collecte_ia.xlsx")

st.write("Aperçu de la base de données actuelle :")
st.dataframe(df_final)
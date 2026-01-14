import pandas as pd
import plotly.express as px
import joblib

# 1. Charger le modèle et les données
model = joblib.load("modele_clusters.pkl")
data = {
    'depenses': [120, 150, 110, 140, 3000, 3500, 3200, 3100, 600, 550, 700, 650, 5000, 4800, 5200],
    'visites': [1, 2, 1, 2, 10, 12, 11, 10, 5, 4, 6, 5, 20, 18, 22]
}
df = pd.DataFrame(data)

# 2. Prédire les groupes pour l'affichage
df['segment'] = model.predict(df).astype(str)

# 3. Générer le graphique en nuage de points
fig = px.scatter(df, x='depenses', y='visites', color='segment', 
                 title="Visualisation des segments découverts par l'IA",
                 labels={'depenses': 'Dépenses (€)', 'visites': 'Visites / mois'},
                 template="plotly_dark")

fig.show()
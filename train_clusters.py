
import pandas as pd
from sklearn.cluster import KMeans
import joblib

# 1. Données d'apprentissage 3D
data = {
    'depenses': [120, 150, 110, 140, 3000, 3500, 3200, 3100, 600, 550, 700, 650, 5000, 4800, 5200],
    'visites': [1, 2, 1, 2, 10, 12, 11, 10, 5, 4, 6, 5, 20, 18, 22],
    'anciennete': [2, 3, 1, 2, 24, 30, 28, 26, 12, 10, 14, 11, 48, 50, 45]
}

df = pd.DataFrame(data)

# 2. Entraînement
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df)

# 3. Sauvegarde
joblib.dump(kmeans, "modele_clusters_3d.pkl")
print("✅ Modèle 3D prêt pour l'exportation !")
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import estimate_bandwidth, MeanShift
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# 1. chargement dataset
df = pd.read_csv("Mall_Customers.csv", sep=";")  

# 2. Sélection des colonnes importantes
features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# 3. Standardisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# 4. Estimation du bandwidth pour Mean Shift
bandwidth = estimate_bandwidth(X_scaled, quantile=0.1, n_samples=len(X_scaled))
print(f"➡️ Bandwidth estimé : {bandwidth:.3f}")

# 5. Application de Mean Shift
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
labels = ms.fit_predict(X_scaled)

# 6. Ajout des labels au dataset
df['Cluster'] = labels

# 7. Affichage de la répartition par cluster
print("\n📊 Répartition par cluster :")
print(df['Cluster'].value_counts())

# 8. Moyennes par cluster
cluster_means = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
print("\n📈 Profil moyen des clusters :")
print(cluster_means)

# 9. Visualisation PCA (réduction à 2D)
pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)
df['PC1'] = coords[:, 0]
df['PC2'] = coords[:, 1]

# 10. Scatter plot (PCA 2D)
plt.figure(figsize=(5, 5))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='tab10', s=80)
plt.title("Visualisation des clusters avec PCA (2D)")
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.legend(title="Cluster")
plt.grid(True)
plt.tight_layout()
plt.show()

# 11. Graphe : Profil par cluster (barres groupées)
cluster_means.plot(kind='bar', figsize=(6, 6))
plt.title("Profil moyen des clients par cluster")
plt.ylabel("Valeur moyenne")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title="Variables")
plt.tight_layout()
plt.show()

# 12. Export du fichier final
df.to_csv("clients_avec_clusters.csv", index=False)

print("\n✅ Export terminé : fichier 'clients_avec_clusters.csv' généré.")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

# Load dataset
df = pd.read_csv('/content/sample_data/healthcare-dataset-stroke-data.csv')
df = df.drop(columns=['id'])
df = df.dropna()
df_encoded = pd.get_dummies(df.drop(columns=['stroke']), drop_first=True)
X = df_encoded.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

# KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df_pca['Cluster'] = kmeans.fit_predict(X_pca)

# Plot KMeans
plt.figure(figsize=(8,6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='Set1')
plt.title('Hasil Clustering dengan K-Means setelah PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.savefig('/content/kmeans_pca.png')
plt.close()

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
df_pca['DBSCAN_Cluster'] = dbscan.fit_predict(X_pca)

# Plot DBSCAN
plt.figure(figsize=(8,6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='DBSCAN_Cluster', palette='tab10')
plt.title('Hasil Clustering dengan DBSCAN setelah PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.savefig('/content/dbscan_pca.png')
plt.close()

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
df_tsne = pd.DataFrame(X_tsne, columns=['Dim1', 'Dim2'])
df_tsne['Cluster'] = kmeans.labels_

# Plot t-SNE
plt.figure(figsize=(8,6))
sns.scatterplot(data=df_tsne, x='Dim1', y='Dim2', hue='Cluster', palette='Set1')
plt.title('Visualisasi t-SNE dengan label dari K-Means')
plt.xlabel('Dimensi 1')
plt.ylabel('Dimensi 2')
plt.grid(True)
plt.savefig('/content/tsne_kmeans.png')
plt.close()

# Silhouette Score
print("=== Silhouette Scores ===")
print(f"KMeans (PCA): {silhouette_score(X_pca, df_pca['Cluster']):.3f}")
if len(set(dbscan.labels_)) > 1:
    print(f"DBSCAN (PCA): {silhouette_score(X_pca, dbscan.labels_):.3f}")
else:
    print("DBSCAN: Tidak bisa dihitung karena hanya 1 cluster atau semua dianggap noise.")

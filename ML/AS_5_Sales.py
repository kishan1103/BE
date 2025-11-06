# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Load dataset
data = pd.read_csv("sales_data_sample.csv", encoding='latin1')
print("Initial shape:", data.shape)
print(data.head())

# Data preprocessing — selecting numeric columns
numeric_data = data.select_dtypes(include=['int64', 'float64']).copy()
numeric_data = numeric_data.dropna()

print("\nNumeric columns used for clustering:")
print(numeric_data.columns.tolist())

# Normalize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# -----------------------------
# K-MEANS CLUSTERING
# -----------------------------

# Finding optimal number of clusters using Elbow Method
inertia = []
K = range(2, 10)

for k in K:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(scaled_data)
    inertia.append(model.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()

# Choose k=3
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)
numeric_data['KMeans_Cluster'] = clusters

# Visualize first two features
plt.figure(figsize=(6,4))
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=clusters, cmap='viridis')
plt.title('K-Means Clustering Visualization')
plt.xlabel(numeric_data.columns[0])
plt.ylabel(numeric_data.columns[1])
plt.show()

# Cluster summary
kmeans_summary = numeric_data.groupby('KMeans_Cluster').mean()
print("\nK-Means Cluster summary:")
print(kmeans_summary)

# -----------------------------
# HIERARCHICAL CLUSTERING
# -----------------------------

# Perform hierarchical clustering using 'ward' linkage
linked = linkage(scaled_data, method='ward')

# Dendrogram visualization
plt.figure(figsize=(10, 6))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=False)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data points')
plt.ylabel('Euclidean distance')
plt.show()

# Choose 3 clusters (cutting the dendrogram)
hier_clusters = fcluster(linked, t=3, criterion='maxclust')
numeric_data['Hier_Cluster'] = hier_clusters

# Visualize hierarchical clustering results
plt.figure(figsize=(6,4))
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=hier_clusters, cmap='plasma')
plt.title('Hierarchical Clustering Visualization')
plt.xlabel(numeric_data.columns[0])
plt.ylabel(numeric_data.columns[1])
plt.show()

# Cluster summary
hier_summary = numeric_data.groupby('Hier_Cluster').mean()
print("\nHierarchical Cluster summary:")
print(hier_summary)

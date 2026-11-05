import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering
)

from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv("Iris.csv")

print(df.head())
print(df.columns)

x = df[
    [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
]

print(x.head())

scaler = StandardScaler()

x_scaled = scaler.fit_transform(x)

wcss = []

for k in range(1, 7):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(x_scaled)

    wcss.append(model.inertia_)

plt.plot(range(1, 7), wcss, marker="o")

plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")

plt.show()

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans_clusters = kmeans.fit_predict(x_scaled)

print("\nK-Means Clusters:")
print(kmeans_clusters)

print(pd.Series(kmeans_clusters).value_counts())

linked = linkage(x_scaled, method="ward")

plt.figure(figsize=(10, 5))

dendrogram(linked)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")

plt.show()

hc = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

hc_clusters = hc.fit_predict(x_scaled)

print("\nHierarchical Clusters:")
print(hc_clusters)

plt.figure(figsize=(8, 6))

plt.scatter(
    x_scaled[:, 2],
    x_scaled[:, 3],
    c=hc_clusters
)

plt.xlabel("Petal Length")
plt.ylabel("Petal Width")

plt.title("Hierarchical Clustering on Iris Dataset")

plt.show()

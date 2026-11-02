
# K-Means Clustering on Iris Dataset

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("Iris.csv")
                                                                                                    
# Remove unnecessary columns
df = df.drop(["Id", "Species"], axis=1, errors="ignore")

# Feature scaling
sc = StandardScaler()
X = sc.fit_transform(df)

# Elbow Method
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Apply K-Means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

# Display results
print("Cluster Labels:")
print(y_pred)

print("\nNumber of Clusters:", len(set(y_pred)))

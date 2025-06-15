# ╔════════════════════════════════════════════════════════════════════╗
# ║                                                                    ║
# ║      Credit Card Clustering Analysis                                ║
# ║      Coded by Pakistani Ethical Hacker: Mr Sabaz Ali Khan           ║
# ║                                                                    ║
# ╚════════════════════════════════════════════════════════════════════╝

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate synthetic credit card data
np.random.seed(42)
n_samples = 1000
X, _ = make_blobs(n_samples=n_samples, centers=4, cluster_std=1.5, random_state=42)

# Create DataFrame with realistic feature names
data = pd.DataFrame(X, columns=['Balance', 'Purchases'])
data['Credit_Limit'] = np.random.uniform(1000, 10000, n_samples)
data['Payments'] = np.random.uniform(0, 5000, n_samples)

# Preprocess the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualize the results
plt.figure(figsize=(10, 6))
scatter = plt.scatter(data['Balance'], data['Purchases'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Balance ($)')
plt.ylabel('Purchases ($)')
plt.title('Credit Card Customer Clusters\nCoded by Mr Sabaz Ali Khan')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)
plt.show()

# Print cluster statistics
print("\nCluster Statistics:")
print(data.groupby('Cluster').mean().round(2))
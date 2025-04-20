import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

# Set page config
st.set_page_config(page_title="k-Means Clustering App", layout="centered")

# Title
st.markdown("<h1 style='text-align: center;'>🔍 K-Means Clustering App with Iris Dataset by Ei Thazin Tun</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Configure Clustering")
k = st.sidebar.slider("Select number of clusters (k)", 2, 10, 4)

# Load dataset
iris = load_iris()
X = iris.data

# Reduce to 2D with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Apply KMeans
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X)

# Define color list (make sure enough colors for up to 10 clusters)
color_list = ['orange', 'green', 'blue', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# Plot
fig, ax = plt.subplots()
for i in range(k):
    cluster_points = X_pca[labels == i]
    if len(cluster_points) > 0:
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   color=color_list[i], label=f"Cluster {i}", s=50)

# Titles and legend
ax.set_title("Clusters (2D PCA Projection)")
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.legend(title="Clusters")

# Show in Streamlit
st.pyplot(fig)

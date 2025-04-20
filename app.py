import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

# Set up page
st.set_page_config(page_title="k-Means Clustering App", layout="centered")

# Page title
st.markdown("<h1 style='text-align: center;'>🔍 K-Means Clustering App with Iris Dataset</h1>", unsafe_allow_html=True)

# Sidebar for cluster count
st.sidebar.header("Configure Clustering")
k = st.sidebar.slider("Select number of clusters (k)", 2, 10, 3)

# Load data
iris = load_iris()
X = iris.data

# PCA for 2D projection
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# KMeans clustering
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X)

# Fixed custom colors
color_list = ['orange', 'green', 'blue', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
cluster_colors = [color_list[i] for i in labels]

# Plotting
fig, ax = plt.subplots()
for i in range(k):
    cluster_points = X_pca[labels == i]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
               color=color_list[i], label=f"Cluster {i}", s=50)

ax.set_title("Clusters (2D PCA Projection)")
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.legend()

# Show plot
st.pyplot(fig)

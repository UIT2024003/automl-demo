import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.title("AutoML Feature Engineering Demo")

st.write("### Step 1: Raw Data")

# Load dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)

st.dataframe(X.head())

st.write(f"Original Shape: {X.shape}")

# Slider for number of components
n_comp = st.slider("Select number of features after reduction (PCA)", 1, 4, 2)

# Apply PCA button
if st.button("Apply Feature Extraction (PCA)"):
    st.write("### Step 2: Applying Feature Engineering...")

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)

    st.write(f"Reduced Shape: {X_pca.shape}")
    st.write("Explained Variance:", pca.explained_variance_ratio_)

    # Plot (only if 2D)
    if n_comp == 2:
        st.write("### Visualization")
        fig, ax = plt.subplots()
        ax.scatter(X_pca[:, 0], X_pca[:, 1])
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        st.pyplot(fig)

st.write("---")
st.write("AutoML automatically performs these steps (scaling + PCA + feature selection)")
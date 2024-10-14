import streamlit as st
import numpy as np
import umap
import plotly.graph_objects as go
import time

# Set page title
st.set_page_config(page_title="UMAP Visualization of Random Data")

# Title
st.title("UMAP Visualization of Random Data")

# Sliders for number of points and dimension
n_points = st.slider("Number of points", min_value=100, max_value=5000, value=1000, step=100)
n_dimensions = st.slider("Number of dimensions", min_value=2, max_value=4000, value=100, step=1)

# Generate random data
@st.cache_data
def generate_data(n_points, n_dimensions):
    return np.random.rand(n_points, n_dimensions)

data = generate_data(n_points, n_dimensions)

# UMAP parameters
n_neighbors = st.slider("Number of neighbors", min_value=2, max_value=100, value=15, step=1)
min_dist = st.slider("Minimum distance", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

# Perform UMAP dimensionality reduction
@st.cache_data
def perform_umap(data, n_neighbors, min_dist):
    t1 = time.time()
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embedding = reducer.fit_transform(data)
    t2 = time.time()
    print(t2-t1, "s for computing umap.")
    st.info(str(t2-t1) + "s for computing umap.")
    return embedding

embedding = perform_umap(data, n_neighbors, min_dist)

# Create scatter plot using Plotly
fig = go.Figure(data=go.Scatter(
    x=embedding[:, 0],
    y=embedding[:, 1],
    mode='markers',
    marker=dict(
        size=5,
        color=np.arange(n_points),
        colorscale='Viridis',
        showscale=True
    )
))

fig.update_layout(
    title='UMAP Visualization',
    xaxis_title='UMAP 1',
    yaxis_title='UMAP 2',
    width=700,
    height=500
)

# Display the plot
st.plotly_chart(fig)

# Add explanation
st.markdown("""
This app visualizes random high-dimensional data using UMAP (Uniform Manifold Approximation and Projection).

- Use the sliders to adjust the number of points and dimensions in the input data.
- Adjust the UMAP parameters (number of neighbors and minimum distance) to see how they affect the visualization.
- The color of each point represents its index in the original dataset.

UMAP is a dimensionality reduction technique that can help visualize high-dimensional data in a 2D space while preserving some of the data's structure.
""")
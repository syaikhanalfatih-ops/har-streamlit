import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from utils import hero, pca_pipeline, siapkan_xy, tata_letak_plotly

hero("Reduksi Dimensi (PCA)", "Proyeksi komponen utama fitur sensor UCI HAR.")

X, y = siapkan_xy()
n_max = min(20, X.shape[1])
n_comp = st.slider("Jumlah komponen PCA:", 2, n_max, min(3, n_max), key="pca_n")

pipe = pca_pipeline(n_comp)
Z = pipe.fit_transform(X)
evr = pipe.named_steps["pca"].explained_variance_ratio_
cum_evr = np.cumsum(evr)

st.metric("Kumulatif Variansi Dijelaskan", f"{cum_evr[-1] * 100:.2f}%")

df_pca = pd.DataFrame({
    "PC1": Z[:, 0],
    "PC2": Z[:, 1],
    "Kelas": np.where(y.values == 1, "Dinamis", "Statis"),
})
fig = px.scatter(
    df_pca.sample(n=min(2000, len(df_pca)), random_state=42),
    x="PC1",
    y="PC2",
    color="Kelas",
    title="Proyeksi PCA (PC1 vs PC2)",
)
st.plotly_chart(tata_letak_plotly(fig), width="stretch")

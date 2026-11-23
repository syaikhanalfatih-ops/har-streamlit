import streamlit as st
from utils import KONFIGURASI, fmt, hasil_cv, hero, latih_model_dasar

hero("Modelling & Evaluasi", "Perbandingan SVC RBF, Random Forest, K-NN, dan Decision Tree.")

st.subheader("Konfigurasi Model")
st.dataframe(KONFIGURASI, width="stretch")

_, df_uji, _ = latih_model_dasar()
df_cv = hasil_cv()

st.subheader("Evaluasi Data Uji")
st.dataframe(df_uji.set_index("Model"), width="stretch")

st.subheader("Evaluasi 5-Fold Cross Validation (Data Latih)")
st.dataframe(df_cv.set_index("Model"), width="stretch")

terbaik_row = df_cv.iloc[0]
st.success(f"Skor F1 CV tertinggi pada data latih: **{terbaik_row['Model']}** ({fmt(terbaik_row['F1'])})")

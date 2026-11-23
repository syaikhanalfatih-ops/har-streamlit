import streamlit as st
from utils import fmt, hero, kotak, muat_data, ringkas_kualitas_data

hero("Tentang Dataset UCI HAR", "Struktur data, subjek penelitian, dan ringkasan kualitas data.")

df = muat_data()
rk = ringkas_kualitas_data()

kotak(
    "Dataset UCI HAR merekam sinyal akselerometer dan giroskop 3-sumbu (50 Hz) dari 30 sukarelawan berusia 19-48 tahun "
    "yang memakai smartphone Samsung Galaxy S II di pinggang.",
    jenis="info",
    tag="SUMBER & PENGUMPULAN DATA",
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Missing Value", rk["missing"])
m2.metric("Baris Duplikat", rk["duplikat"])
m3.metric("Baris Outlier (IQR)", rk["outlier_baris"])
m4.metric("Fitur Skewness |sk| > 1", rk["miring"])

st.caption(f"Proporsi nilai nol pada fitur: {rk['nol_sel']:,} sel ({fmt(rk['nol_persen'], 2)}%).".replace(",", "."))

st.subheader("Distribusi Label Aktivitas")
st.bar_chart(df["Activity"].value_counts())

st.subheader("Cuplikan Kolom Awal")
st.dataframe(df.iloc[:15, :12], width="stretch")

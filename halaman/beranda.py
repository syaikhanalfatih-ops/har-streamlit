import streamlit as st
from utils import AKTIVITAS_DINAMIS, AKTIVITAS_STATIS, fmt, hero, kotak, muat_data, ringkas_kualitas_data

hero(
    "Klasifikasi Aktivitas Manusia (UCI HAR)",
    "Aplikasi Streamlit berbasis dataset Human Activity Recognition Using Smartphones (akselerometer & giroskop).",
)

df = muat_data()
rk = ringkas_kualitas_data()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Jumlah Baris", f"{rk['baris']:,}".replace(",", "."))
c2.metric("Jumlah Kolom", rk["kolom"])
c3.metric("Jumlah Subjek", rk["subjek"])
c4.metric("Rasio Dinamis/Statis", fmt(rk["rasio"], 2))

st.latex(r"X = [x_1, x_2, x_3, \dots, x_{561}], \quad x_i \in [-1, 1]")

k1, k2 = st.columns(2)
with k1:
    kotak(
        f"<b>{rk['dinamis']:,} sampel</b> — {', '.join(AKTIVITAS_DINAMIS)}".replace(",", "."),
        jenis="sesudah",
        tag="AKTIVITAS DINAMIS (KELAS 1)",
    )
with k2:
    kotak(
        f"<b>{rk['statis']:,} sampel</b> — {', '.join(AKTIVITAS_STATIS)}".replace(",", "."),
        jenis="sebelum",
        tag="AKTIVITAS STATIS (KELAS 0)",
    )

st.subheader("Sampel Dataset UCI HAR")
st.dataframe(df.head(50), width="stretch")

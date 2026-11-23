import streamlit as st
from utils import bagi_data, hero, kotak

hero("Preprocessing Data", "Pembagian data latih/uji, imputasi median, dan standardisasi fitur.")

Xtr, Xte, ytr, yte = bagi_data()

c1, c2, c3 = st.columns(3)
c1.metric("Jumlah Fitur", Xtr.shape[1])
c2.metric("Data Latih (80%)", Xtr.shape[0])
c3.metric("Data Uji (20%)", Xte.shape[0])

kotak(
    "Pembagian data menggunakan <b>stratified train-test split</b> (80% latih, 20% uji, <code>random_state=42</code>) "
    "agar proporsi aktivitas dinamis dan statis seimbang.",
    jenis="sesudah",
    tag="STRATIFIED SPLIT",
)

st.write("Statistik ringkas 6 fitur pertama pada data latih:")
st.dataframe(Xtr.iloc[:, :6].describe().T[["mean", "std", "min", "max"]], width="stretch")

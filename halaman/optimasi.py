import pandas as pd
import streamlit as st
from utils import fmt, hero, jalankan_optimasi, kotak

hero("Teknik Optimasi (SVM)", "Seleksi fitur SelectKBest + GridSearchCV untuk model SVC RBF.")

res = jalankan_optimasi()
sebelum = res["hasil"]["Sebelum"]
sesudah = res["hasil"]["Sesudah"]

c1, c2, c3 = st.columns(3)
c1.metric("Akurasi Sebelum", fmt(sebelum["Accuracy"]))
c2.metric("Akurasi Sesudah", fmt(sesudah["Accuracy"]))
c3.metric("Skor CV Terbaik", fmt(res["cv_terbaik"]))

st.write("Parameter terbaik hasil Grid Search:", res["params"])
kotak(
    f"Jumlah fitur terpilih: <b>{len(res['fitur_terpilih'])}</b> fitur — "
    f"{', '.join(res['fitur_terpilih'][:10])}{'...' if len(res['fitur_terpilih']) > 10 else ''}",
    jenis="info",
    tag="FITUR TERPILIH (SELECTKBEST)",
)

st.dataframe(pd.DataFrame({"Sebelum": sebelum, "Sesudah": sesudah}).T, width="stretch")

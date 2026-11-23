import streamlit as st
from utils import bagi_data, daftar_fitur, hero, model_prediksi

hero("Prediksi Satu Sampel", "Uji prediksi aktivitas dinamis/statis berdasarkan sampel dataset atau input fitur.")

_, Xte, _, yte = bagi_data()
daftar_m = model_prediksi()
pilihan_model = st.selectbox("Pilih model:", list(daftar_m.keys()), key="pred_model")
idx_sampel = st.number_input("Indeks baris data uji:", min_value=0, max_value=len(Xte) - 1, value=0, key="pred_idx")

baris_x = Xte.iloc[[int(idx_sampel)]]
label_asli = int(yte.iloc[int(idx_sampel)])

if st.button("Jalankan Prediksi", key="btn_pred_satu"):
    m = daftar_m[pilihan_model]
    pred = int(m.predict(baris_x)[0])
    teks_pred = "Aktivitas Dinamis (1)" if pred == 1 else "Aktivitas Statis (0)"
    teks_asli = "Aktivitas Dinamis (1)" if label_asli == 1 else "Aktivitas Statis (0)"
    st.info(f"Hasil Prediksi: **{teks_pred}** | Label Asli: **{teks_asli}**")
    if hasattr(m, "predict_proba"):
        prob = m.predict_proba(baris_x)[0]
        st.write(f"Probabilitas — Statis: `{prob[0]:.4f}`, Dinamis: `{prob[1]:.4f}`")

st.caption(f"Total fitur yang diharapkan model: {len(daftar_fitur())} kolom.")

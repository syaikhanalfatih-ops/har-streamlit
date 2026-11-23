import pandas as pd
import streamlit as st
from utils import daftar_fitur, hero, model_prediksi

hero("Prediksi Massal (CSV)", "Unggah file CSV dengan fitur sensor UCI HAR untuk prediksi banyak baris.")

daftar_m = model_prediksi()
pilihan_model = st.selectbox("Pilih model:", list(daftar_m.keys()), key="massal_model")
berkas = st.file_uploader("Unggah file CSV:", type=["csv"], key="massal_csv")

if berkas is not None:
    df_up = pd.read_csv(berkas)
    fitur_wajib = daftar_fitur()
    hilang = [c for c in fitur_wajib if c not in df_up.columns]
    if hilang:
        st.error(f"Kolom fitur kurang ({len(hilang)} kolom), contoh: {hilang[:5]}")
    else:
        m = daftar_m[pilihan_model]
        pred = m.predict(df_up[fitur_wajib])
        hasil_df = df_up.copy()
        hasil_df["Prediksi_Kelas"] = pred
        hasil_df["Prediksi_Kategori"] = ["Dinamis" if p == 1 else "Statis" for p in pred]
        st.success(f"Prediksi selesai untuk {len(hasil_df)} baris.")
        st.dataframe(hasil_df.head(50), width="stretch")
        st.download_button(
            "Unduh Hasil Prediksi (CSV)",
            data=hasil_df.to_csv(index=False).encode("utf-8"),
            file_name="hasil_prediksi_har.csv",
            mime="text/csv",
            key="dl_hasil_har",
        )

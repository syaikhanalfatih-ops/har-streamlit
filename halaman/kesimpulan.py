import streamlit as st
from utils import fmt, hasil_cv, hero, jalankan_optimasi, kotak

hero("Kesimpulan & Saran", "Ringkasan hasil eksperimen klasifikasi aktivitas UCI HAR.")

df_cv = hasil_cv()
opt = jalankan_optimasi()
terbaik = df_cv.iloc[0]

kotak(
    f"Pada 5-fold cross-validation data latih, model teratas adalah <b>{terbaik['Model']}</b> "
    f"dengan rata-rata F1 <b>{fmt(terbaik['F1'])}</b>. Pipeline optimasi SVC RBF "
    f"mencapai skor CV terbaik <b>{fmt(opt['cv_terbaik'])}</b>.",
    jenis="sesudah",
    tag="RINGKASAN EVALUASI",
)

kotak(
    "Untuk pengembangan lanjutan, klasifikasi dapat diperluas dari biner (dinamis vs statis) "
    "menjadi <b>multikelas 6 aktivitas</b> (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) "
    "serta diuji dengan validasi berbasis subjek (<i>GroupKFold</i> per <code>subject</code>).",
    jenis="info",
    tag="SARAN PENGEMBANGAN",
)

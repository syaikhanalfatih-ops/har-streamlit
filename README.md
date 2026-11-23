# Klasifikasi Aktivitas Manusia UCI HAR (Streamlit)

Aplikasi Streamlit multi-halaman untuk klasifikasi **biner** aktivitas manusia
(dinamis vs statis) dari dataset **UCI Human Activity Recognition Using Smartphones**.

## Demo

- Streamlit Community Cloud: `https://har-streamlit.streamlit.app` *(isi setelah deploy)*

## Menjalankan lokal

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app.py
```

Buka `http://localhost:8501`.

## Struktur

```
HAR/
├── app.py                 # entry point + navigasi (st.navigation)
├── utils.py               # pemuatan data, model, evaluasi, cache & artefak
├── buat_artefak.py        # membuat ulang artefak pra-perhitungan
├── har_dataset.csv        # 10.299 baris x 563 kolom (561 fitur + subject + Activity)
├── requirements.txt       # versi dependensi yang di-pin
├── artefak/               # hasil latih yang sudah di-precompute (dipakai di Cloud)
└── halaman/               # 10 halaman aplikasi
```

## Artefak pra-perhitungan

Halaman Optimasi, Modelling, Prediksi, dan Prediksi Massal memakai hasil latih
yang disimpan di `artefak/` agar tidak melatih ulang model saat aplikasi dibuka
(terutama penting di Streamlit Community Cloud yang resource-nya terbatas).

Bila data, fitur, atau konfigurasi model berubah, buat ulang artefak:

```bash
python buat_artefak.py
```

`utils.py` otomatis memakai artefak bila tersedia; jika tidak ada, model akan
dilatih ulang saat halaman dibuka.

> Penting: artefak dihasilkan dengan versi library di `requirements.txt`
> (scikit-learn 1.9.1). Ganti versi library = buat ulang artefak.

## Halaman

| Grup | Halaman |
|---|---|
| Pengantar | Beranda, Tentang Dataset, Eksplorasi Data (EDA) |
| Pengolahan & Model | Preprocessing, Reduksi Dimensi (PCA), Teknik Optimasi (SVM), Modelling & Evaluasi |
| Penerapan | Prediksi Satu Sampel, Prediksi Massal (CSV), Kesimpulan & Saran |

## Deploy ke Streamlit Community Cloud

1. Push folder ini ke GitHub (repository publik).
2. Buka [share.streamlit.io](https://share.streamlit.io) → **Create app**.
3. Pilih repository, branch `main`, main file `app.py`, Python **3.12**.
4. Deploy.

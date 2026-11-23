"""
Membuat artefak pra-perhitungan untuk aplikasi Streamlit HAR.

Tujuan: halaman Optimasi, Modelling, Prediksi, dan Prediksi Massal tidak perlu
melatih ulang model berat saat aplikasi pertama kali dibuka (mis. di Streamlit
Community Cloud). Artefak disimpan di folder ./artefak dan dimuat otomatis
oleh utils.py.

Jalankan:
    python buat_artefak.py

Ulangi perintah ini setiap kali data, fitur, atau konfigurasi model berubah.
"""
import time
from pathlib import Path

import utils as u


def ukuran(path):
    return f"{Path(path).stat().st_size / 1024 / 1024:.2f} MB"


def main():
    total = time.perf_counter()
    print("Membuat artefak HAR (butuh beberapa menit pada mesin ini)...\n", flush=True)

    print("[1/3] Metrik 4 model pada data uji...", flush=True)
    t = time.perf_counter()
    model, df_uji, prediksi = u.hitung_model_dasar()
    p = u.simpan_artefak("model_dasar", (model, df_uji, prediksi))
    print(f"      selesai {time.perf_counter() - t:.1f}s -> {p.name} ({ukuran(p)})", flush=True)

    print("[2/3] 5-fold cross validation 4 model...", flush=True)
    t = time.perf_counter()
    df_cv = u.hitung_cv()
    p = u.simpan_artefak("cv", df_cv)
    print(f"      selesai {time.perf_counter() - t:.1f}s -> {p.name} ({ukuran(p)})", flush=True)

    print("[3/3] GridSearchCV SVC (400 kombinasi x fold)...", flush=True)
    t = time.perf_counter()
    opt = u.hitung_optimasi()
    p = u.simpan_artefak("optimasi", opt)
    print(f"      selesai {time.perf_counter() - t:.1f}s -> {p.name} ({ukuran(p)})", flush=True)

    total_detik = time.perf_counter() - total
    print(f"\nSelesai dalam {total_detik / 60:.1f} menit.", flush=True)
    print("\nEvaluasi data uji:")
    print(df_uji.set_index("Model").round(4).to_string(), flush=True)
    print("\nCross validation:")
    print(df_cv.set_index("Model").round(4).to_string(), flush=True)
    print("\nParameter terbaik:", opt["params"], flush=True)
    print("Skor CV terbaik:", round(opt["cv_terbaik"], 4), flush=True)


if __name__ == "__main__":
    main()

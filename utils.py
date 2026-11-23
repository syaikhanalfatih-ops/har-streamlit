"""
Fungsi bersama untuk website Klasifikasi Aktivitas Manusia UCI HAR (Streamlit).
Berisi: tema, pemuatan data HAR, definisi model, evaluasi, dan cache hasil pelatihan.
"""
import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
DATA_PATH = BASE / "har_dataset.csv"
ALT_DATA_PATH = BASE / "data" / "har_dataset.csv"
ARTEFAK_DIR = BASE / "artefak"
RANDOM_STATE = 42

AKTIVITAS_LIST = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]
AKTIVITAS_DINAMIS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"]
AKTIVITAS_STATIS = ["SITTING", "STANDING", "LAYING"]

WARNA_DINAMIS = "#2a9d8f"
WARNA_STATIS = "#e63946"
WARNA_AKSEN = "#2563eb"
WARNA_NETRAL = "#6c757d"


def pasang_tema():
    st.markdown(
        f"""
        <style>
        .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }}
        h1 {{ letter-spacing: -0.5px; }}
        [data-testid="stMetric"] {{
            background: rgba(128,128,128,0.08);
            border: 1px solid rgba(128,128,128,0.20);
            border-left: 5px solid {WARNA_AKSEN};
            padding: 0.8rem 1rem; border-radius: 10px;
        }}
        .kotak {{
            border-radius: 10px; padding: 0.9rem 1.1rem; margin: 0.5rem 0 1rem 0;
            border: 1px solid rgba(128,128,128,0.25); background: rgba(128,128,128,0.07);
        }}
        .kotak.sebelum {{ border-left: 5px solid {WARNA_STATIS}; }}
        .kotak.sesudah {{ border-left: 5px solid {WARNA_DINAMIS}; }}
        .kotak.info    {{ border-left: 5px solid {WARNA_AKSEN}; }}
        .kotak b.tag   {{ font-size: 0.75rem; letter-spacing: 1px; }}
        .hero {{
            padding: 2rem 2rem; border-radius: 16px; margin-bottom: 1.2rem;
            background: linear-gradient(135deg, {WARNA_AKSEN} 0%, #0ea5e9 100%); color: white;
        }}
        .hero h1 {{ color: white; margin: 0 0 .3rem 0; }}
        .hero p  {{ color: rgba(255,255,255,.94); margin: 0; font-size: 1.05rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(judul, subjudul):
    st.markdown(f'<div class="hero"><h1>{judul}</h1><p>{subjudul}</p></div>', unsafe_allow_html=True)


def kotak(isi, jenis="info", tag=None):
    label = f'<b class="tag">{tag}</b><br>' if tag else ""
    st.markdown(f'<div class="kotak {jenis}">{label}{isi}</div>', unsafe_allow_html=True)


def tata_letak_plotly(fig, tinggi=380):
    fig.update_layout(
        height=tinggi,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def fmt(x, d=4):
    return f"{x:.{d}f}".replace(".", ",")


def path_artefak(nama):
    return ARTEFAK_DIR / f"{nama}.pkl"


def muat_artefak(nama):
    """Muat hasil pra-perhitungan bila tersedia (dipakai di Streamlit Cloud)."""
    p = path_artefak(nama)
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except Exception:
        return None


def simpan_artefak(nama, obj):
    ARTEFAK_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path_artefak(nama), compress=3)
    return path_artefak(nama)


@st.cache_data
def muat_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    if ALT_DATA_PATH.exists():
        return pd.read_csv(ALT_DATA_PATH, encoding="utf-8-sig")

    base_dir = BASE / "UCI HAR Dataset"
    features = pd.read_csv(base_dir / "features.txt", sep=r"\s+", header=None, names=["id", "name"])
    act_labels = pd.read_csv(base_dir / "activity_labels.txt", sep=r"\s+", header=None, names=["id", "activity"])

    counts = {}
    col_names = []
    for idx, name in zip(features["id"], features["name"]):
        if name in counts:
            counts[name] += 1
            col_names.append(f"{name}_{idx}")
        else:
            counts[name] = 1
            col_names.append(name)

    X_train = pd.read_csv(base_dir / "train" / "X_train.txt", sep=r"\s+", header=None)
    y_train = pd.read_csv(base_dir / "train" / "y_train.txt", sep=r"\s+", header=None, names=["activity_id"])
    sub_train = pd.read_csv(base_dir / "train" / "subject_train.txt", sep=r"\s+", header=None, names=["subject"])

    X_test = pd.read_csv(base_dir / "test" / "X_test.txt", sep=r"\s+", header=None)
    y_test = pd.read_csv(base_dir / "test" / "y_test.txt", sep=r"\s+", header=None, names=["activity_id"])
    sub_test = pd.read_csv(base_dir / "test" / "subject_test.txt", sep=r"\s+", header=None, names=["subject"])

    X_all = pd.concat([X_train, X_test], ignore_index=True)
    X_all.columns = col_names
    y_all = pd.concat([y_train, y_test], ignore_index=True)
    sub_all = pd.concat([sub_train, sub_test], ignore_index=True)

    act_map = dict(zip(act_labels["id"], act_labels["activity"]))
    return pd.concat([sub_all, y_all["activity_id"].map(act_map).rename("Activity"), X_all], axis=1)


@st.cache_data
def siapkan_xy():
    df = muat_data()
    # Target biner: 1 = Aktivitas Dinamis (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS), 0 = Statis
    y = df["Activity"].isin(AKTIVITAS_DINAMIS).astype(int)
    drop_cols = [c for c in ["subject", "Activity"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    return X, y


@st.cache_data
def bagi_data():
    X, y = siapkan_xy()
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def daftar_fitur():
    X, _ = siapkan_xy()
    return list(X.columns)


def _pipe(model, scale=True):
    langkah = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        langkah.append(("scaler", StandardScaler()))
    langkah.append(("model", model))
    return Pipeline(langkah)


NAMA_MODEL = ["SVC RBF", "Random Forest", "K-NN", "Decision Tree"]


def model_empat():
    return {
        "SVC RBF": _pipe(SVC(C=10, gamma="scale", probability=True, random_state=RANDOM_STATE)),
        "Random Forest": _pipe(
            RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE),
            scale=False,
        ),
        "K-NN": _pipe(KNeighborsClassifier(n_neighbors=9, weights="distance")),
        "Decision Tree": _pipe(DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE), scale=False),
    }


KONFIGURASI = pd.DataFrame(
    [
        ("Non-parametrik", "SVC kernel RBF", "C = 10, gamma = scale", "Ya (StandardScaler)"),
        ("Pohon dan ensemble", "Random Forest", "100 pohon, max_depth = 10", "Tidak perlu"),
        ("Non-parametrik", "K-NN Classifier", "k = 9, weights = distance", "Ya (StandardScaler)"),
        ("Pohon dan ensemble", "Decision Tree", "max_depth = 5", "Tidak perlu"),
    ],
    columns=["Kelompok", "Model", "Konfigurasi", "Perlu scaling"],
)


def model_svc_optimasi(k=25, C=10, gamma=0.01, probabilitas=True):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("seleksi", SelectKBest(f_classif, k=k)),
        ("model", SVC(C=C, gamma=gamma, probability=probabilitas, random_state=RANDOM_STATE)),
    ])


def skor_model(m, X):
    if hasattr(m, "predict_proba"):
        return m.predict_proba(X)[:, 1]
    return m.decision_function(X)


def hitung_metrik(y_true, y_pred, skor=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (Dinamis)": precision_score(y_true, y_pred, zero_division=0),
        "Recall (Dinamis)": recall_score(y_true, y_pred, zero_division=0),
        "F1 (Dinamis)": f1_score(y_true, y_pred, zero_division=0),
        "F1 makro": f1_score(y_true, y_pred, average="macro"),
        "ROC-AUC": roc_auc_score(y_true, skor) if skor is not None else np.nan,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def hitung_model_dasar():
    Xtr, Xte, ytr, yte = bagi_data()
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    model, baris, prediksi = {}, [], {}
    for nama, m in model_empat().items():
        t0 = time.perf_counter()
        m.fit(Xtr, ytr)
        t_latih = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred = m.predict(Xte)
        t_pred = time.perf_counter() - t0

        skor = skor_model(m, Xte)
        met = hitung_metrik(yte, pred, skor)
        cvs = cross_validate(m, Xtr, ytr, cv=cv, scoring="f1_macro")["test_score"]
        met.update({
            "Model": nama,
            "F1 makro CV": cvs.mean(),
            "F1 makro CV std": cvs.std(),
            "Waktu latih (dtk)": t_latih,
            "Waktu prediksi (dtk)": t_pred,
        })
        baris.append(met)
        model[nama] = m
        prediksi[nama] = (pred, skor)
    return model, pd.DataFrame(baris), prediksi


@st.cache_resource(show_spinner="Memuat model dasar HAR...")
def latih_model_dasar():
    tersimpan = muat_artefak("model_dasar")
    if tersimpan is not None:
        return tersimpan
    return hitung_model_dasar()


@st.cache_resource(show_spinner="Menjalankan Grid Search HAR...")
def jalankan_optimasi():
    tersimpan = muat_artefak("optimasi")
    if tersimpan is not None:
        return tersimpan
    return hitung_optimasi()


def hitung_optimasi():
    Xtr, Xte, ytr, yte = bagi_data()
    max_k = min(30, Xtr.shape[1])
    k_opsi = sorted(set([x for x in [10, 15, 20, 25, 30] if x <= max_k] + [max_k]))
    pipa = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("seleksi", SelectKBest(f_classif)),
        ("model", SVC(random_state=RANDOM_STATE)),
    ])
    grid = {
        "seleksi__k": k_opsi,
        "model__C": [0.1, 1, 10, 100],
        "model__gamma": ["scale", 0.001, 0.01, 0.1],
    }
    gs = GridSearchCV(pipa, grid, cv=5, scoring="accuracy", n_jobs=-1).fit(Xtr, ytr)
    bp = gs.best_params_
    terbaik = model_svc_optimasi(bp["seleksi__k"], bp["model__C"], bp["model__gamma"]).fit(Xtr, ytr)
    sebelum = model_svc_optimasi(max_k, 1, "scale").fit(Xtr, ytr)
    hasil = {}
    for nama, m in [("Sebelum", sebelum), ("Sesudah", terbaik)]:
        pred, skor = m.predict(Xte), skor_model(m, Xte)
        hasil[nama] = hitung_metrik(yte, pred, skor)
    fitur_terpilih = list(Xtr.columns[terbaik.named_steps["seleksi"].get_support()])
    return {
        "params": bp,
        "cv_terbaik": gs.best_score_,
        "model": terbaik,
        "hasil": hasil,
        "fitur_terpilih": fitur_terpilih,
        "tabel": pd.DataFrame(gs.cv_results_),
    }


def hitung_cv():
    Xtr, _, ytr, _ = bagi_data()
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    skor = {
        "Accuracy": "accuracy",
        "Precision": "precision",
        "Recall": "recall",
        "F1": "f1",
        "ROC-AUC": "roc_auc",
    }
    baris = []
    for nama, m in model_empat().items():
        r = cross_validate(m, Xtr, ytr, cv=cv, scoring=skor)
        baris.append({
            "Model": nama,
            **{k: r[f"test_{k}"].mean() for k in skor},
            "F1 (std)": r["test_F1"].std(),
        })
    return pd.DataFrame(baris).sort_values("F1", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner="Memuat validasi silang 4 model HAR...")
def hasil_cv():
    tersimpan = muat_artefak("cv")
    if tersimpan is not None:
        return tersimpan
    return hitung_cv()


@st.cache_resource(show_spinner="Menyiapkan model prediksi HAR...")
def model_prediksi():
    daftar = {"SVC RBF + optimasi (terbaik)": jalankan_optimasi()["model"]}
    dasar, _, _ = latih_model_dasar()
    daftar.update(dasar)
    return daftar


def pca_pipeline(n):
    return Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=n, random_state=RANDOM_STATE))])


def ringkas_kualitas_data():
    df = muat_data()
    X, y = siapkan_xy()
    q1, q3 = X.quantile(0.25), X.quantile(0.75)
    iqr = q3 - q1
    mask = (X < q1 - 1.5 * iqr) | (X > q3 + 1.5 * iqr)
    return {
        "baris": len(df),
        "kolom": df.shape[1],
        "subjek": int(df["subject"].nunique()) if "subject" in df.columns else 0,
        "missing": int(df.isna().sum().sum()),
        "duplikat": int(df.duplicated().sum()),
        "nol_sel": int((X == 0).sum().sum()),
        "nol_persen": float((X == 0).sum().sum() / X.size * 100),
        "outlier_baris": int(mask.any(axis=1).sum()),
        "outlier_sel": int(mask.sum().sum()),
        "miring": int((X.skew().abs() > 1).sum()),
        "dinamis": int((y == 1).sum()),
        "statis": int((y == 0).sum()),
        "rasio": float((y == 1).sum() / max(1, (y == 0).sum())),
    }

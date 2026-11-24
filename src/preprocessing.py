"""Pembersihan data, pemilihan fitur, dan pembagian data (train/test).

Aturan utama: semua transformasi yang belajar dari data (misalnya scaler)
hanya boleh di-fit pada data train. Di proyek ini scaler dipasang di dalam
Pipeline model, jadi otomatis hanya belajar dari data train.
"""
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

SEED = 42

# Arti kolom label di Kaggle: 1 = legitimate, 0 = phishing.
# Di proyek ini dibuat kolom y_phishing: 1 = phishing (kelas positif), 0 = legitimate.
LABEL_COL = "label"
Y_COL = "y_phishing"
GROUP_COL = "group_domain"

# Kolom teks / identitas, tidak dipakai sebagai fitur.
TEXT_ID_COLS = ["FILENAME", "URL", "Domain", "TLD", "Title"]

# Fitur turunan statistik dari seluruh dataset (bukan murni dari string URL).
# URLSimilarityIndex bernilai 100 untuk semua baris legitimate -> kandidat kebocoran label.
DERIVED_COLS = ["URLSimilarityIndex", "CharContinuationRate", "TLDLegitimateProb", "URLCharProb"]

# Fitur leksikal/struktural murni dari string URL (18 fitur), dipakai di Skenario B dan C.
URL_ONLY = [
    "URLLength", "DomainLength", "IsDomainIP", "TLDLength", "NoOfSubDomain",
    "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio",
    "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL",
    "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL", "IsHTTPS",
]


def load_data(path):
    """Baca CSV, buang baris dengan URL duplikat (simpan yang pertama), buat kolom bantu.

    Mengembalikan (df, info). info berisi jumlah baris sebelum dan sesudah pembersihan.
    """
    raw = pd.read_csv(path)
    info = {"baris_awal": len(raw), "url_duplikat_dibuang": int(raw["URL"].duplicated().sum())}
    df = raw.drop_duplicates(subset="URL", keep="first").reset_index(drop=True)
    info["baris_akhir"] = len(df)

    df[Y_COL] = (df[LABEL_COL] == 0).astype(int)  # 1 = phishing
    # Kelompok domain: huruf kecil, awalan "www." dibuang supaya www.x.com dan x.com satu kelompok
    df[GROUP_COL] = df["Domain"].str.lower().str.replace(r"^www\.", "", regex=True)
    return df, info


def feature_sets(df):
    """Daftar fitur untuk tiap skenario."""
    skip = set(TEXT_ID_COLS + [LABEL_COL, Y_COL, GROUP_COL])
    semua = [c for c in df.select_dtypes("number").columns if c not in skip]
    return {
        "A": semua,                                                    # semua fitur numerik
        "A_tanpa_USI": [c for c in semua if c != "URLSimilarityIndex"],  # uji kebocoran USI
        "B": URL_ONLY,
        "C": URL_ONLY,
    }


def split_random(df, test_size=0.2, seed=SEED):
    """Stratified random split. Mengembalikan indeks train dan test."""
    train_idx, test_idx = train_test_split(
        df.index, test_size=test_size, random_state=seed, stratify=df[Y_COL]
    )
    return train_idx, test_idx


def split_group(df, test_size=0.2, seed=SEED):
    """Split berdasarkan kelompok domain: satu domain hanya ada di train ATAU test."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_pos, test_pos = next(gss.split(df, df[Y_COL], groups=df[GROUP_COL]))
    return df.index[train_pos], df.index[test_pos]

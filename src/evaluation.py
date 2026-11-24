"""Model, evaluasi, dan pencatatan hasil. Kelas positif = phishing (y_phishing = 1)."""
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .preprocessing import GROUP_COL, SEED, Y_COL

METRIC_COLUMNS = [
    "scenario", "model", "split", "features", "n_features", "seed",
    "val_f1", "val_roc_auc", "test_f1", "test_precision", "test_recall",
    "test_roc_auc", "inference_ms_per_sample", "n_train", "n_test", "notes",
]


def make_models(seed=SEED):
    """Dua model baseline dengan hyperparameter bawaan. Scaler ada di dalam Pipeline,
    jadi hanya belajar dari data train (dan dari lipatan train saat cross-validation)."""
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]),
        "RandomForest": RandomForestClassifier(random_state=seed, n_jobs=-1),
    }


def inference_ms_per_sample(model, X_test, repeats=3):
    """Rata-rata waktu prediksi per sampel (ms), satu thread, nilai terbaik dari beberapa ulangan.
    Hanya mengukur waktu model memprediksi, BUKAN waktu mengambil/mengekstrak fitur."""
    if hasattr(model, "n_jobs"):
        model.set_params(n_jobs=1)
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict(X_test)
        best = min(best, time.perf_counter() - t0)
    return best / len(X_test) * 1000


def plot_confusion(cm, path, title):
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    labels = ["Legitimate", "Phishing"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual"); ax.set_title(title, fontsize=9)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_scenario(df, scenario, features, train_idx, test_idx, split_name,
                 seed=SEED, grouped_cv=False, do_cv=True, fig_dir=None, notes=""):
    """Latih LR dan RF pada train, ukur di test. Mengembalikan DataFrame (satu baris per model).

    do_cv=True  : hitung metrik validasi dengan 5-fold cross-validation DI DALAM data train.
    grouped_cv  : True -> GroupKFold per domain (untuk Skenario C), False -> StratifiedKFold.
    Data test hanya dipakai sekali, untuk evaluasi akhir.
    """
    X_tr, y_tr = df.loc[train_idx, features], df.loc[train_idx, Y_COL]
    X_te, y_te = df.loc[test_idx, features], df.loc[test_idx, Y_COL]
    rows = []
    for name, model in make_models(seed).items():
        val_f1 = val_auc = np.nan
        if do_cv:
            if grouped_cv:
                cv = GroupKFold(n_splits=5)
                groups = df.loc[train_idx, GROUP_COL]
            else:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                groups = None
            cvres = cross_validate(model, X_tr, y_tr, cv=cv, groups=groups,
                                   scoring=["f1", "roc_auc"])
            val_f1, val_auc = cvres["test_f1"].mean(), cvres["test_roc_auc"].mean()

        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        score = model.predict_proba(X_te)[:, 1]
        cm = confusion_matrix(y_te, pred, labels=[0, 1])
        if fig_dir is not None:
            plot_confusion(cm, Path(fig_dir) / f"cm_{scenario}_{name}_seed{seed}.png",
                           f"Skenario {scenario} - {name}")
        rows.append({
            "scenario": scenario, "model": name, "split": split_name,
            "features": "semua" if scenario.startswith("A") else "URL-only",
            "n_features": len(features), "seed": seed,
            "val_f1": val_f1, "val_roc_auc": val_auc,
            "test_f1": f1_score(y_te, pred), "test_precision": precision_score(y_te, pred),
            "test_recall": recall_score(y_te, pred), "test_roc_auc": roc_auc_score(y_te, score),
            "inference_ms_per_sample": inference_ms_per_sample(model, X_te),
            "n_train": len(X_tr), "n_test": len(X_te), "notes": notes,
        })
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def save_results(new_rows, path):
    """Tambahkan hasil ke CSV. Baris lama dengan (scenario, model, seed) yang sama diganti."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        old = pd.read_csv(path)
        key = ["scenario", "model", "seed"]
        merged = old.merge(new_rows[key], on=key, how="left", indicator=True)
        old = old[merged["_merge"].eq("left_only").values]
        out = pd.concat([old, new_rows], ignore_index=True)
    else:
        out = new_rows
    out.to_csv(path, index=False)
    return out

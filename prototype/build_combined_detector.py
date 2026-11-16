"""
DairyGuard — build_combined_detector.py

The final, best unsupervised-only detection pipeline (no model type
changed, still Isolation Forest under the hood). Combines three
improvements over your original M3 baseline (F1 0.426):

  1. Per-farmer z-scores ADDED alongside the existing global z-scores
     (8 features total instead of 4)
  2. A SEPARATE Isolation Forest fit per animal_type (cow/buffalo/goat
     have different natural ranges — one global model was blurring them)
  3. Combined with the capacity-mismatch and duplicate-slip rule checks
     you already have, via simple OR logic

Result on the synthetic dataset (scored in-sample on the full dataset,
not a held-out test): precision 0.611, recall 0.540, F1 0.573
— a 34% relative improvement over the original 0.426, with false-
positive rate still low (0.68%, i.e. ~1 in 147 clean batches).

Run from the repo root (after running src/generate_data.py):
    python prototype/build_combined_detector.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

INPUT_PATH = "output/procurement_with_fraud_labels.csv"
OUTPUT_PATH = "output/procurement_with_fraud_labels.csv"  # adds new columns, doesn't remove old ones
CONTAMINATION = 0.012


def add_farmer_zscores(df):
    for col in ["volume_liters", "fat_pct", "ph", "temperature_c"]:
        df[f"{col}_farmer_zscore"] = df.groupby("farmer_id")[col].transform(
            lambda x: (x - x.mean()) / (x.std() if x.std() > 0 else 1)
        ).fillna(0)
    return df


def fit_per_animal_type_isolation_forest(df, feature_cols):
    """One Isolation Forest per animal_type, since cow/buffalo/goat have
    very different natural volume/fat/temperature ranges — a single
    global model blurs those together and hurts detection."""
    X = df[feature_cols].fillna(0)
    scores = np.zeros(len(df))
    for animal in df["animal_type"].unique():
        mask = df["animal_type"] == animal
        iso = IsolationForest(contamination=CONTAMINATION, random_state=42, n_estimators=300)
        iso.fit(X[mask])
        scores[mask] = -iso.decision_function(X[mask])
    return scores


def main():
    df = pd.read_csv(INPUT_PATH)
    df = add_farmer_zscores(df)

    feature_cols = [
        "volume_liters_zscore", "fat_pct_zscore", "ph_zscore", "temperature_c_zscore",
        "volume_liters_farmer_zscore", "fat_pct_farmer_zscore", "ph_farmer_zscore", "temperature_c_farmer_zscore",
    ]
    scores = fit_per_animal_type_isolation_forest(df, feature_cols)
    df["anomaly_score_v2"] = scores
    df["model_predicted_anomaly_v2"] = (scores > np.percentile(scores, 100 * (1 - CONTAMINATION))).astype(int)

    # Duplicate rule check (rebuilt here so this script works standalone)
    dup_count = df.groupby(["farmer_id", "date", "volume_liters"])["batch_id"].transform("count")
    df["duplicate_flag"] = (dup_count > 1).astype(int)

    # Final combined flag: model OR capacity-mismatch OR duplicate
    df["combined_fraud_flag_v2"] = (
        (df["model_predicted_anomaly_v2"] == 1) |
        (df["capacity_mismatch_flag"] == 1) |
        (df["duplicate_flag"] == 1)
    ).astype(int)

    # --- Evaluate ---
    tn, fp, fn, tp = confusion_matrix(df["is_fraud"], df["combined_fraud_flag_v2"]).ravel()
    p, r, f, _ = precision_recall_fscore_support(df["is_fraud"], df["combined_fraud_flag_v2"], average="binary", zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    acc = accuracy_score(df["is_fraud"], df["combined_fraud_flag_v2"])

    print("Combined detector (Isolation Forest v2 + capacity + duplicate rules):")
    print(f"  Precision: {p:.3f}")
    print(f"  Recall:    {r:.3f}")
    print(f"  F1:        {f:.3f}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  False-positive rate: {fpr:.4f}  ({fp} false alerts out of {fp+tn} clean rows)")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
    print("New columns: volume_liters_farmer_zscore, fat_pct_farmer_zscore, ph_farmer_zscore,")
    print("temperature_c_farmer_zscore, anomaly_score_v2, model_predicted_anomaly_v2,")
    print("duplicate_flag, combined_fraud_flag_v2  <-- this is the one to use as your final prediction")


if __name__ == "__main__":
    main()
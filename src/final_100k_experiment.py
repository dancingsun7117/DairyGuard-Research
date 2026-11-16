"""
DairyGuard — Final 100K Reproducible Experiment

Protocol
--------
1. Full dataset: 100,000 records.
2. Stratified 80:20 split:
      - 80,000 development/training partition
      - 20,000 untouched blind test partition
3. Internal 80:20 validation split inside the 80,000 development partition:
      - 64,000 model-development records
      - 16,000 validation records
4. Hyperparameters are fixed from the development/validation stage.
5. Final Isolation Forest is retrained on the complete 80,000-record
   training partition, using only normal training records.
6. Final XGBoost is retrained on the complete 80,000-record training
   partition, using the final Isolation Forest outputs as features.
7. The 20,000-record test partition is evaluated exactly once.
8. The test set is never used for tuning.

Important leakage protection
----------------------------
- is_fraud is the target and is excluded from model features.
- fraud_type is excluded.
- adulteration_flag is excluded. In this generated dataset it exactly
  matches is_fraud, so including it would be direct label leakage.
- transaction/farmer/centre/buyer identifiers are excluded.

This script reports the final 80K-trained model results.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE_WITHIN_TRAIN = 0.20

IF_CONTAMINATION = 0.05
IF_ESTIMATORS = 300

XGB_ESTIMATORS = 300
XGB_MAX_DEPTH = 5
XGB_LEARNING_RATE = 0.05
XGB_SCALE_POS_WEIGHT = 50.0
PROBABILITY_THRESHOLD = 0.50
N_JOBS = 4

NUMERIC_FEATURES = [
    "quantity_litres",
    "fat_percent",
    "snf_percent",
    "ph",
    "temperature_c",
]
CATEGORICAL_FEATURES = ["animal_type"]

TARGET = "is_fraud"


def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def get_if_features(if_model, df):
    base = df[NUMERIC_FEATURES].copy()
    score = -if_model.decision_function(base)
    flag = (if_model.predict(base) == -1).astype(int)
    return score, flag


def metric_dict(y_true, y_prob, threshold=0.50):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to the 100K DairyGuard CSV")
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the final 80K-trained model bundle",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Leakage check.
    leakage_message = None
    if "adulteration_flag" in df.columns:
        same = df["adulteration_flag"].astype(int).equals(df[TARGET].astype(int))
        if same:
            leakage_message = (
                "adulteration_flag exactly equals is_fraud in this dataset; "
                "it is excluded from XGBoost to prevent direct label leakage."
            )
        else:
            leakage_message = (
                "adulteration_flag exists but does not exactly equal is_fraud; "
                "it is still excluded from model features."
            )

    y = df[TARGET].astype(int)

    # ------------------------------------------------------------
    # STEP 1: 80K development / 20K blind test
    # ------------------------------------------------------------
    dev_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # ------------------------------------------------------------
    # STEP 2: 64K development-fit / 16K internal validation
    # Used only to represent the validation stage. Hyperparameters
    # are already fixed in this final script.
    # ------------------------------------------------------------
    dev_y = dev_df[TARGET].astype(int)

    fit_df, val_df = train_test_split(
        dev_df,
        test_size=VALIDATION_SIZE_WITHIN_TRAIN,
        stratify=dev_y,
        random_state=RANDOM_STATE,
    )

    # ------------------------------------------------------------
    # STEP 3: FINAL Isolation Forest on ALL 80K training records
    # only normal records are used for fitting.
    # ------------------------------------------------------------
    final_if = IsolationForest(
        n_estimators=IF_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    final_if.fit(dev_df.loc[dev_df[TARGET] == 0, NUMERIC_FEATURES])

    # IF outputs for complete 80K training and 20K blind test.
    train_if_score, train_if_flag = get_if_features(final_if, dev_df)
    test_if_score, test_if_flag = get_if_features(final_if, test_df)

    train_model_df = dev_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    test_model_df = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()

    train_model_df["if_score"] = train_if_score
    train_model_df["if_flag"] = train_if_flag

    test_model_df["if_score"] = test_if_score
    test_model_df["if_flag"] = test_if_flag

    # ------------------------------------------------------------
    # STEP 4: FINAL preprocessing fit on ALL 80K training records
    # ------------------------------------------------------------
    preprocessor = make_preprocessor()
    X_train_base = preprocessor.fit_transform(train_model_df)
    X_test_base = preprocessor.transform(test_model_df)

    y_train = dev_df[TARGET].astype(int).to_numpy()
    y_test = test_df[TARGET].astype(int).to_numpy()

    # ------------------------------------------------------------
    # STEP 5: FINAL XGBoost trained on ALL 80K training records
    # ------------------------------------------------------------
    xgb = XGBClassifier(
        n_estimators=XGB_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        scale_pos_weight=XGB_SCALE_POS_WEIGHT,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=N_JOBS,
    )

    xgb.fit(X_train_base, y_train)

    # ------------------------------------------------------------
    # STEP 6: ONE FINAL evaluation on untouched 20K test
    # ------------------------------------------------------------
    test_prob = xgb.predict_proba(X_test_base)[:, 1]
    metrics = metric_dict(
        y_test,
        test_prob,
        threshold=PROBABILITY_THRESHOLD,
    )

    # ------------------------------------------------------------
    # Save predictions / split / metrics
    # ------------------------------------------------------------
    predictions = test_df.copy()
    predictions["predicted_probability"] = test_prob
    predictions["predicted_label"] = (
        test_prob >= PROBABILITY_THRESHOLD
    ).astype(int)
    predictions["if_score"] = test_if_score
    predictions["if_flag"] = test_if_flag

    predictions_path = Path("dairyguard_100k_final_80k_test_predictions.csv")
    split_path = Path("dairyguard_100k_final_80k_split.csv")
    metrics_path = Path("dairyguard_100k_final_80k_metrics.json")

    predictions.to_csv(predictions_path, index=False)

    split_out = pd.DataFrame(
        {
            "row_index": test_df.index,
            "partition": "blind_test_20k",
            "is_fraud": y_test,
        }
    )
    split_out.to_csv(split_path, index=False)

    output = {
        "experiment": "DairyGuard final 100K / final models trained on complete 80K training partition",
        "dataset_records": int(len(df)),
        "dataset_anomalies": int(y.sum()),
        "development_records": int(len(dev_df)),
        "development_anomalies": int(dev_df[TARGET].sum()),
        "internal_validation_records": int(len(val_df)),
        "internal_validation_anomalies": int(val_df[TARGET].sum()),
        "development_fit_records_during_validation_stage": int(len(fit_df)),
        "blind_test_records": int(len(test_df)),
        "blind_test_anomalies": int(y_test.sum()),
        "blind_test_normal": int((y_test == 0).sum()),
        "final_if_training_records": int((dev_df[TARGET] == 0).sum()),
        "final_xgboost_training_records": int(len(dev_df)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "probability_threshold": PROBABILITY_THRESHOLD,
        "if_estimators": IF_ESTIMATORS,
        "if_contamination": IF_CONTAMINATION,
        "xgb_estimators": XGB_ESTIMATORS,
        "xgb_max_depth": XGB_MAX_DEPTH,
        "xgb_learning_rate": XGB_LEARNING_RATE,
        "xgb_scale_pos_weight": XGB_SCALE_POS_WEIGHT,
        "leakage_check": leakage_message,
        "metrics": metrics,
    }

    metrics_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    if args.save_model:
        model_bundle = {
            "preprocessor": preprocessor,
            "isolation_forest": final_if,
            "xgboost": xgb,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "threshold": PROBABILITY_THRESHOLD,
            "random_state": RANDOM_STATE,
        }
        joblib.dump(model_bundle, "dairyguard_100k_final_80k_model.joblib")

    print("\nDairyGuard — FINAL 100K EXPERIMENT")
    print("================================")
    print(f"Full dataset      : {len(df):,} records")
    print(f"Full anomaly      : {int(y.sum()):,} ({y.mean()*100:.2f}%)")
    print(f"Training          : {len(dev_df):,} ({int(dev_df[TARGET].sum()):,} anomalies)")
    print(f"Validation        : {len(val_df):,} ({int(val_df[TARGET].sum()):,} anomalies)")
    print(f"Validation fit    : {len(fit_df):,}")
    print(f"Blind test        : {len(test_df):,} ({int(y_test.sum()):,} anomalies)")
    print("Split             : 80:20 stratified, random_state=42")
    print("FINAL RETRAIN     : Isolation Forest + XGBoost trained on complete 80K")
    if leakage_message:
        print("LEAKAGE CHECK     :", leakage_message)

    print("\nFINAL BLIND TEST RESULTS")
    print("=========================")
    for key in [
        "precision",
        "recall",
        "f1",
        "accuracy",
        "false_positive_rate",
        "false_negative_rate",
        "roc_auc",
        "pr_auc",
    ]:
        print(f"{key:22s}: {metrics[key]:.6f}")

    print(
        f"TP={metrics['TP']}  FP={metrics['FP']}  "
        f"FN={metrics['FN']}  TN={metrics['TN']}"
    )

    print("\nSaved:")
    print(f"  {predictions_path}")
    print(f"  {split_path}")
    print(f"  {metrics_path}")
    if args.save_model:
        print("  dairyguard_100k_final_80k_model.joblib")


if __name__ == "__main__":
    main()

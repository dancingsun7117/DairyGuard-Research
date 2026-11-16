# DairyGuard Research

Fraud and anomaly detection for dairy milk procurement, combining Isolation Forest anomaly scores, an XGBoost classifier and deterministic rule checks into a risk score for ranking records for inspection, evaluated on a synthetic 100,000-record dataset with injected fraud labels.

> A manuscript describing this work is under revision for resubmission.

## My contribution
I trained the Isolation Forest component and implemented the rule-based checks. (Team members: add names and roles.)

## Data
All records are **synthetic**. Fraud is injected with known labels, so every result measures detection of *injected* anomalies, not real fraud. The 100,000-record dataset has columns `quantity_litres`, `fat_percent`, `snf_percent`, `ph`, `temperature_c`, `animal_type` and the label `is_fraud`. **[Describe how it was generated, or link the generator.]**

## Final experiment: `src/final_100k_experiment.py`
```
pip install -r requirements.txt
python src/final_100k_experiment.py path/to/dairyguard_100k.csv
```
Protocol:
1. Stratified 80:20 split (random state 42): 80,000 training records and a 20,000-record blind test set.
2. Isolation Forest (300 trees) fitted on the **normal training records only**, so it is semi-supervised, not purely unsupervised. Its score and flag become two features.
3. XGBoost (300 trees, depth 5, learning rate 0.05, `scale_pos_weight` 50) trained on the 80,000 training records using the numeric features, the animal type and the two Isolation Forest features.
4. The test set is evaluated once with a fixed 0.50 probability threshold. No tuning on the test set.
5. `is_fraud`, `fraud_type`, `adulteration_flag` (identical to `is_fraud` in this dataset) and all identifiers are excluded from the features.

The script prints precision, recall, F1, accuracy, false-positive rate, ROC-AUC, PR-AUC and the confusion matrix, and writes `dairyguard_100k_final_80k_metrics.json`.

## Results
The script was smoke-tested end to end on a small toy CSV with the expected columns (numbers from that test are meaningless and not reported). The real 100,000-record numbers must come from your own run.

**[Run the script and paste your verified numbers here, with the exact precision, recall and confusion counts.]**

## Hybrid risk score (described in the manuscript)
The manuscript fuses the three signals into one risk score `R = w1*s_IF + w2*p_XGB + w3*r_rule` (Isolation Forest score, XGBoost probability, binary rule flag; weights 0.40 / 0.45 / 0.15; threshold R = 0.6399066144; scores normalised to [0, 1]). `src/final_100k_experiment.py` covers the XGBoost + Isolation Forest stage only. **[Add the script that computes R, the rule flags on the 100K dataset, and the ablation table (Rules only, Isolation Forest only, XGBoost alone, XGBoost + Isolation Forest, final hybrid), and link it here.]**

## Prototype pipeline: `prototype/`
An earlier, smaller pipeline (about 15,000 rows) with the rule checks (capacity mismatch, duplicate slips, OCR slip-ID matching) and per-animal-type Isolation Forests. `generate_data.py` needs `data/adfJuly2025final_1.csv` and `data/census.csv` (Maharashtra district procurement and livestock-census tables; check redistribution terms or link the sources).

## Limitations
- Fraud labels are synthetic, so scores show detection of our injected anomalies, not real-world fraud.
- Isolation Forest scores for training rows are computed on data the forest was fitted on (for normal rows), while test rows are unseen; out-of-fold scores would be cleaner.
- The validation split in the script is only counted, not used for tuning; hyperparameters are fixed in the code.
- `scale_pos_weight` 50 is well above the class ratio, which favours recall over precision.
- In the prototype, some rules mirror how fraud was injected, and the Louvain step uses fraud labels to build its graph, so it is an illustration only.
- Validation on real procurement data is future work.

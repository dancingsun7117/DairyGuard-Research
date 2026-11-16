# DairyGuard Research

Fraud and anomaly detection for dairy milk procurement, combining Isolation Forest anomaly scores, an XGBoost classifier and deterministic rule checks into a risk score for ranking records for inspection, evaluated on a synthetic 100,000-record dataset with injected fraud labels.
> A manuscript describing this work is under revision for resubmission.

## My contribution
I trained the Isolation Forest component ,implemented the rule-based checks and calculated final risk score fusing Isolation Forest scores, an XGBoost classifier and deterministic
rule checks .

### Dataset & Data Generation

- **100,000 synthetic records** generated using real-world procurement distributions, publicly available Government of Maharashtra procurement data and patterns, and simulated ERP-style records.
- Features: `quantity_litres`, `fat_percent`, `snf_percent`, `ph`, `temperature_c`, `animal_type`, and `is_fraud`.
- Six classes of integrity anomalies injected with known labels.
- Evaluated using **precision, recall, and false-positive rates**.

**Note:** Fraud labels represent injected anomalies, not verified real-world fraud.
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

Blind test set: 20,000 transactions (922 anomalies, 19,078 normal). All models trained on the 80,000-record training partition; test set evaluated once.

| Method | Precision (%) | Recall (%) | F1 (%) | Accuracy (%) | FPR (%) | FNR (%) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|
| XGBoost | 74.67 | 99.46 | 85.30 | 98.42 | 1.63 | 0.54 | 0.9944 | 0.8691 |
| Isolation Forest | 48.12 | 98.37 | 64.62 | 95.04 | 5.13 | 1.63 | 0.9913 | 0.8162 |
| XGBoost + Isolation Forest | 74.13 | 99.46 | 84.95 | 98.38 | 1.65 | 0.54 | 0.9947 | 0.8593 |
| Rules only* | 75.14 | 59.98 | 66.71 | 97.26 | 0.96 | 40.02 | 0.7951 | 0.4507 |
| Final hybrid system (R >= 0.6399066144) | 76.07 | 96.53 | 85.09 | 98.49 | 1.44 | 3.47 | 0.9939 | 0.8514 |

*Rules-only output is a single binary flag, so ROC-AUC = (TPR + TNR)/2 and PR-AUC = precision x recall at that operating point.

**Reading the table.** XGBoost alone has the best F1 and recall. The final hybrid trades about 3 points of recall for about 1.4 points of precision and a slightly lower false-positive rate (1.44% vs 1.63%), at essentially the same F1 (85.09 vs 85.30). Its added value is an explainable, rule-backed risk level per transaction, not higher raw accuracy. Rules alone catch about 60% of anomalies, which is why the supervised stage is needed.

## Hybrid risk score

`R = 0.40*s_IF + 0.45*p_XGB + 0.15*r_rule` (Isolation Forest score normalised to [0, 1], XGBoost probability, binary rule flag). `src/final_100k_experiment.py` covers XGBoost + Isolation Forest only; `src/hybrid_ablation.py` computes R, the label-free rule flags (duplicate slip, capacity mismatch) and the ablation table. `adulteration_flag` is not used as a rule flag because it equals `is_fraud` in this dataset.

## Prototype pipeline: `prototype/`
An earlier, smaller pipeline (about 15,000 rows) with the rule checks (capacity mismatch, duplicate slips, OCR slip-ID matching) and per-animal-type Isolation Forests. `generate_data.py` needs `data/adfJuly2025final_1.csv` and `data/census.csv` (Maharashtra district procurement and livestock-census tables; check redistribution terms or link the sources).

## Limitations
- Fraud labels are synthetic, so scores show detection of our injected anomalies, not real-world fraud.
- Isolation Forest scores for training rows are computed on data the forest was fitted on (for normal rows), while test rows are unseen; out-of-fold scores would be cleaner.
- The validation split in the script is only counted, not used for tuning; hyperparameters are fixed in the code.
- `scale_pos_weight` 50 is well above the class ratio, which favours recall over precision.
- In the prototype, some rules mirror how fraud was injected, and the Louvain step uses fraud labels to build its graph, so it is an illustration only.
- Validation on real procurement data is future work.

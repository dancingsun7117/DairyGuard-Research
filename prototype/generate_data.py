import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

# ---- STEP A: Load real procurement data ----
govt_df = pd.read_csv("data/adfJuly2025final_1.csv")
govt_df = govt_df[~govt_df["DIST"].isin(["Total", "DIST TTL", "Grand Total", "MHND"])]
govt_df["DIST"] = govt_df["DIST"].str.strip()

# ---- STEP B: Load real livestock census ----
census_df = pd.read_csv("data/census.csv")
census_df.columns = [c.strip() for c in census_df.columns]
census_df["District"] = census_df["District"].str.strip().str.title()

# ---- STEP C: Map abbreviated procurement names -> full census names ----
NAME_MAP = {
    "A.Nagar": "Ahmednagar",
    "A\"bad": "Aurangabad",
    "Amarvti": "Amravati",
    "Bhandra": "Bhandara",
    "Buldhna": "Buldhana",
    "Chandrpur": "Chandrapur",
    "Dharashiv": "Osmanabad",
    "Gdchrli": "Gadchiroli",
    "Gondia": "Gondiya",
    "Kolhapr": "Kolhapur",
    "Nagpr": "Nagpur",
    "Nandrbr": "Nandurbar",
    "Sindrg.": "Sindhudurg",
    "Yvtmal": "Yavatmal",
}

govt_df["District_Full"] = govt_df["DIST"].apply(lambda d: NAME_MAP.get(d, d))

# ---- STEP D: Merge into one clean district reference table ----
district_table = govt_df[["District_Full", "Total Procurement"]].merge(
    census_df[["District", "Cattle-Total", "Buffaloes-Total", "Goat -Total"]],
    left_on="District_Full", right_on="District", how="left"
)
district_table = district_table.drop(columns=["District"]).rename(columns={
    "District_Full": "district",
    "Total Procurement": "real_procurement_volume",
    "Cattle-Total": "cattle_pop",
    "Buffaloes-Total": "buffalo_pop",
    "Goat -Total": "goat_pop",
})

unmatched = district_table[district_table["cattle_pop"].isna()]
if len(unmatched) > 0:
    print("WARNING - unmatched districts:")
    print(unmatched["district"].tolist())
else:
    print(f"All {len(district_table)} districts matched successfully.")

print("\nMerged district table (first 5 rows):")
print(district_table.head())

district_table.to_csv("output/district_reference.csv", index=False)
print("\nSaved to output/district_reference.csv")
# ============================================================
# STEP E: Generate farmer master data (Pashu-Aadhar style IDs)
# ============================================================

FARMERS_PER_DISTRICT = 15   # keep small for now - beginner scale, easy to inspect
DAYS = 30                   # 1 month of daily records
START_DATE = datetime(2025, 6, 1)

farmers = []
farmer_counter = 1

for _, row in district_table.iterrows():
    district = row["district"]
    total_animals = row["cattle_pop"] + row["buffalo_pop"] + row["goat_pop"]

    p_cattle = row["cattle_pop"] / total_animals
    p_buffalo = row["buffalo_pop"] / total_animals
    p_goat = row["goat_pop"] / total_animals

    for _ in range(FARMERS_PER_DISTRICT):
        species = np.random.choice(
            ["cow", "buffalo", "goat"],
            p=[p_cattle, p_buffalo, p_goat]
        )
        farmer_id = f"PA{farmer_counter:06d}"
        collector_id = f"DCS-{district[:3].upper()}-{np.random.randint(1,6)}"

        farmers.append({
            "farmer_id": farmer_id,
            "district": district,
            "collector_id": collector_id,
            "animal_type": species,
        })
        farmer_counter += 1

farmers_df = pd.DataFrame(farmers)
print(f"\nStep E done: Generated {len(farmers_df)} farmers across {district_table.shape[0]} districts")
print(farmers_df.head(10))
print("\nAnimal type distribution:")
print(farmers_df["animal_type"].value_counts())

farmers_df.to_csv("output/farmers_master.csv", index=False)
print("\nSaved to output/farmers_master.csv")
# ============================================================
# STEP F: Generate clean daily procurement transactions
# ============================================================

# Realistic daily yield per animal (liters) - vet-standard ranges
YIELD_RANGES = {
    "cow": (8, 15),
    "buffalo": (5, 10),
    "goat": (1, 3),
}
FAT_RANGES = {
    "cow": (3.5, 4.5),
    "buffalo": (6.0, 8.0),
    "goat": (3.5, 4.5),
}

records = []
batch_counter = 1

for _, farmer in farmers_df.iterrows():
    species = farmer["animal_type"]
    base_yield = np.random.uniform(*YIELD_RANGES[species])  # this farmer's "normal" daily volume

    for day_offset in range(DAYS):
        date = START_DATE + timedelta(days=day_offset)

        # small natural day-to-day variation (+/- 10%)
        volume = round(base_yield * np.random.uniform(0.9, 1.1), 2)
        fat_pct = round(np.random.uniform(*FAT_RANGES[species]), 2)
        ph = round(np.random.uniform(6.4, 6.8), 2)

        records.append({
            "batch_id": f"B{batch_counter:07d}",
            "farmer_id": farmer["farmer_id"],
            "district": farmer["district"],
            "collector_id": farmer["collector_id"],
            "animal_type": species,
            "date": date.strftime("%Y-%m-%d"),
            "volume_liters": volume,
            "fat_pct": fat_pct,
            "ph": ph,
            "is_fraud": 0,          # ground truth label - clean for now
            "fraud_type": "none",
        })
        batch_counter += 1

transactions_df = pd.DataFrame(records)
print(f"\nStep F done: Generated {len(transactions_df)} clean transactions")
print(transactions_df.head(10))

transactions_df.to_csv("output/clean_transactions.csv", index=False)
print("\nSaved to output/clean_transactions.csv")
# ============================================================
# STEP G: Inject labeled fraud into a copy of the clean data
# ============================================================

fraud_df = transactions_df.copy()

N_DUPLICATES = 40
N_VOLUME_SPIKES = 40
N_QUALITY_MISMATCH = 40
N_OCR_MISMATCH = 30

rng = np.random.default_rng(42)

# ---- 1. Duplicates: same batch logged twice with a new batch_id ----
dup_rows = fraud_df.sample(N_DUPLICATES, random_state=1).copy()
dup_rows["batch_id"] = [f"B{batch_counter + i:07d}" for i in range(len(dup_rows))]
batch_counter += len(dup_rows)
dup_rows["is_fraud"] = 1
dup_rows["fraud_type"] = "duplicate_slip"

# ---- 2. Volume spikes: inflate volume +100-200% ----
spike_idx = fraud_df.sample(N_VOLUME_SPIKES, random_state=2).index
fraud_df.loc[spike_idx, "volume_liters"] = (
    fraud_df.loc[spike_idx, "volume_liters"] * rng.uniform(2.0, 3.0, size=N_VOLUME_SPIKES)
).round(2)
fraud_df.loc[spike_idx, "is_fraud"] = 1
fraud_df.loc[spike_idx, "fraud_type"] = "volume_spike"

# ---- 3. Quality mismatch: diluted fat% (below species floor) ----
quality_idx = fraud_df.sample(N_QUALITY_MISMATCH, random_state=3).index
fraud_df.loc[quality_idx, "fat_pct"] = (
    fraud_df.loc[quality_idx, "fat_pct"] * rng.uniform(0.5, 0.7, size=N_QUALITY_MISMATCH)
).round(2)
fraud_df.loc[quality_idx, "is_fraud"] = 1
fraud_df.loc[quality_idx, "fraud_type"] = "quality_mismatch_dilution"

# ---- 4. OCR/manual-entry mismatch: slip value disagrees with system value ----
fraud_df["slip_volume_liters"] = fraud_df["volume_liters"]
ocr_idx = fraud_df.sample(N_OCR_MISMATCH, random_state=4).index
fraud_df.loc[ocr_idx, "slip_volume_liters"] = (
    fraud_df.loc[ocr_idx, "volume_liters"] * rng.uniform(1.15, 1.4, size=N_OCR_MISMATCH)
).round(2)
fraud_df.loc[ocr_idx, "is_fraud"] = 1
fraud_df.loc[ocr_idx, "fraud_type"] = "ocr_manual_mismatch"

# ---- Combine duplicates back in ----
fraud_df = pd.concat([fraud_df, dup_rows], ignore_index=True)

print(f"\nStep G done: {len(fraud_df)} total rows (including {len(dup_rows)} injected duplicates)")
print("Fraud label breakdown:")
print(fraud_df["fraud_type"].value_counts())
print(f"\nTotal fraud rate: {(fraud_df['is_fraud'].sum() / len(fraud_df) * 100):.2f}%")

fraud_df.to_csv("output/procurement_with_fraud_labels.csv", index=False)
print("\nSaved to output/procurement_with_fraud_labels.csv  <-- this is your Core 1 dataset")
# ============================================================
# STEP H: Historical-baseline deviation (sustained drift, not spike)
# ============================================================

N_BASELINE_DRIFT_FARMERS = 8
drift_farmers = farmers_df.sample(N_BASELINE_DRIFT_FARMERS, random_state=5)["farmer_id"].tolist()

for fid in drift_farmers:
    farmer_rows = fraud_df[fraud_df["farmer_id"] == fid].sort_values("date")
    # affect the last 10 days of this farmer's record - gradual/sustained drop
    drift_dates = farmer_rows["date"].tolist()[-10:]
    idx = fraud_df[(fraud_df["farmer_id"] == fid) & (fraud_df["date"].isin(drift_dates))].index

    fraud_df.loc[idx, "volume_liters"] = (
        fraud_df.loc[idx, "volume_liters"] * rng.uniform(0.35, 0.55, size=len(idx))
    ).round(2)
    fraud_df.loc[idx, "is_fraud"] = 1
    fraud_df.loc[idx, "fraud_type"] = "baseline_deviation"

print(f"\nStep H done: injected sustained baseline drift into {N_BASELINE_DRIFT_FARMERS} farmers (last 10 days each)")

# ============================================================
# STEP I: Mass-balance reconciliation (route total vs plant intake)
# ============================================================

# Sum what farmers actually reported, grouped by collector + date
route_daily = fraud_df.groupby(["collector_id", "date"])["volume_liters"].sum().reset_index()
route_daily = route_daily.rename(columns={"volume_liters": "collector_logged_total"})

# Simulate "plant intake" - normally should match collector total closely
route_daily["plant_intake_total"] = (
    route_daily["collector_logged_total"] * rng.uniform(0.98, 1.02, size=len(route_daily))
).round(2)

# Inject mismatch on a handful of collector-days (someone's skimming between route and plant)
N_MASS_BALANCE_ANOMALIES = 15
mb_idx = route_daily.sample(N_MASS_BALANCE_ANOMALIES, random_state=6).index
route_daily.loc[mb_idx, "plant_intake_total"] = (
    route_daily.loc[mb_idx, "collector_logged_total"] * rng.uniform(0.6, 0.8, size=N_MASS_BALANCE_ANOMALIES)
).round(2)

route_daily["variance_pct"] = (
    (route_daily["collector_logged_total"] - route_daily["plant_intake_total"])
    / route_daily["collector_logged_total"] * 100
).round(2)
route_daily["is_mass_balance_anomaly"] = 0
route_daily.loc[mb_idx, "is_mass_balance_anomaly"] = 1

print(f"\nStep I done: {len(route_daily)} collector-day records, {N_MASS_BALANCE_ANOMALIES} flagged mass-balance anomalies")
print(route_daily[route_daily["is_mass_balance_anomaly"] == 1].head())

route_daily.to_csv("output/mass_balance_reconciliation.csv", index=False)
fraud_df.to_csv("output/procurement_with_fraud_labels.csv", index=False)

print("\nAll done. Final fraud breakdown across full dataset:")
print(fraud_df["fraud_type"].value_counts())
print(f"Total fraud rate: {(fraud_df['is_fraud'].sum() / len(fraud_df) * 100):.2f}%")
# ============================================================
# STEP J: Network anomaly clusters (real Louvain community detection)
# ============================================================

import networkx as nx

farmer_fraud_dates = fraud_df[fraud_df["is_fraud"] == 1].groupby("farmer_id")["date"].apply(set).to_dict()
farmer_fraud_rate = fraud_df.groupby("farmer_id")["is_fraud"].mean().to_dict()

# Build farmer-farmer graph: edges between farmers sharing a collector,
# heavily upweighted if they were also flagged fraud on the SAME dates (collusion signal)
G = nx.Graph()
G.add_nodes_from(farmers_df["farmer_id"])
for collector_id, group in farmers_df.groupby("collector_id"):
    farmer_ids = group["farmer_id"].tolist()
    for i in range(len(farmer_ids)):
        for j in range(i+1, len(farmer_ids)):
            f1, f2 = farmer_ids[i], farmer_ids[j]
            weight = 1.0
            shared_fraud_dates = len(farmer_fraud_dates.get(f1, set()) & farmer_fraud_dates.get(f2, set()))
            weight += shared_fraud_dates * 3.0
            G.add_edge(f1, f2, weight=weight)

# Real Louvain community detection (modularity-based graph clustering)
communities = nx.community.louvain_communities(G, weight="weight", seed=42, resolution=1.0)

overall_avg = fraud_df["is_fraud"].mean()
rows = []
for i, comm in enumerate(communities):
    if len(comm) < 2:
        continue
    rates = [farmer_fraud_rate.get(f, 0) for f in comm]
    avg_rate = np.mean(rates)
    rows.append({
        "community_id": i, "size": len(comm),
        "avg_fraud_rate": round(avg_rate, 4),
        "fraud_rate_vs_baseline": round(avg_rate / overall_avg, 2) if overall_avg > 0 else 0,
        "farmer_ids": ", ".join(list(comm))
    })

comm_df = pd.DataFrame(rows).sort_values("avg_fraud_rate", ascending=False)
comm_df["suspicious_community"] = (comm_df["fraud_rate_vs_baseline"] >= 3).astype(int)

print(f"\nStep J done: Graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Louvain found {len(communities)} communities ({len(comm_df)} with size>=2)")
print(f"Flagged {comm_df['suspicious_community'].sum()} communities as suspicious (>=3x baseline fraud rate)")
print(comm_df.head(10)[["community_id","size","avg_fraud_rate","fraud_rate_vs_baseline","suspicious_community"]].to_string(index=False))

comm_df.to_csv("output/louvain_communities.csv", index=False)
print("\nSaved to output/louvain_communities.csv")
# ============================================================
# STEP K: Generate OCR-scanned slip text (with realistic corruption)
# ============================================================

def corrupt_ocr(text, corrupt=False):
    if not corrupt:
        return text
    confusions = {"0": "O", "1": "l", "5": "S", "8": "B", "2": "Z"}
    chars = list(text)
    n_corrupt = rng.integers(1, 3)
    positions = rng.choice(len(chars), size=min(n_corrupt, len(chars)), replace=False)
    for pos in positions:
        c = chars[pos]
        if c in confusions:
            chars[pos] = confusions[c]
    return "".join(chars)

N_OCR_TEXT_MISMATCH = 35
fraud_df["slip_id_ocr_text"] = fraud_df["farmer_id"]

ocr_text_idx = fraud_df.sample(N_OCR_TEXT_MISMATCH, random_state=7).index
fraud_df.loc[ocr_text_idx, "slip_id_ocr_text"] = fraud_df.loc[ocr_text_idx, "farmer_id"].apply(
    lambda x: corrupt_ocr(x, corrupt=True)
)
fraud_df.loc[ocr_text_idx, "needs_ocr_review"] = 1
fraud_df["needs_ocr_review"] = fraud_df["needs_ocr_review"].fillna(0).astype(int)

print(f"\nStep K done: Generated OCR text field, {N_OCR_TEXT_MISMATCH} rows corrupted for testing")

fraud_df.to_csv("output/procurement_with_fraud_labels.csv", index=False)
# ============================================================
# STEP L: Add temperature field (+ inject temperature anomalies)
# ============================================================

fraud_df["temperature_c"] = np.round(rng.uniform(4.0, 10.0, size=len(fraud_df)), 1)

N_TEMP_ANOMALIES = 35
temp_idx = fraud_df.sample(N_TEMP_ANOMALIES, random_state=8).index
fraud_df.loc[temp_idx, "temperature_c"] = np.round(rng.uniform(15.0, 25.0, size=N_TEMP_ANOMALIES), 1)
fraud_df.loc[temp_idx, "is_fraud"] = 1
still_clean = fraud_df.loc[temp_idx][fraud_df.loc[temp_idx, "fraud_type"] == "none"].index
fraud_df.loc[still_clean, "fraud_type"] = "temperature_anomaly"

print(f"\nStep L done: Added temperature field, {N_TEMP_ANOMALIES} temperature anomalies injected")
print(fraud_df[["farmer_id", "temperature_c", "fraud_type"]].sample(5, random_state=9))

fraud_df.to_csv("output/procurement_with_fraud_labels.csv", index=False)
print("\nUpdated output/procurement_with_fraud_labels.csv with temperature field")
# ============================================================
# STEP M: Train Isolation Forest on species-normalized features
# ============================================================

from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

for col in ["volume_liters", "fat_pct", "ph", "temperature_c"]:
    fraud_df[f"{col}_zscore"] = fraud_df.groupby("animal_type")[col].transform(
        lambda x: (x - x.mean()) / x.std()
    )

FEATURES = ["volume_liters_zscore", "fat_pct_zscore", "ph_zscore", "temperature_c_zscore"]
X = fraud_df[FEATURES].fillna(0)

DETECTABLE_TYPES = ["volume_spike", "quality_mismatch_dilution", "baseline_deviation", "temperature_anomaly"]
detectable_mask = fraud_df["fraud_type"].isin(DETECTABLE_TYPES)
true_contamination = detectable_mask.sum() / len(fraud_df)

model = IsolationForest(n_estimators=200, contamination=true_contamination, random_state=42)
model.fit(X)
fraud_df["model_predicted_anomaly"] = (model.predict(X) == -1).astype(int)
fraud_df["anomaly_score"] = model.decision_function(X)

print(f"\nStep M done: Isolation Forest trained (contamination={true_contamination:.4f})")
print(f"Flagged {fraud_df['model_predicted_anomaly'].sum()} rows as anomalies")

eval_df = fraud_df[detectable_mask | (fraud_df["fraud_type"] == "none")]
print(f"\nEvaluation against the {detectable_mask.sum()} numerically-detectable fraud rows:")
print(classification_report(eval_df["is_fraud"], eval_df["model_predicted_anomaly"],
                             target_names=["normal","fraud"], zero_division=0))
print("(Note: duplicates + OCR mismatches are intentionally excluded here - Isolation")
print(" Forest can't see them since they don't change volume/fat/ph/temperature values.")
print(" Those are caught by separate rule-based + semantic matching layers.)")

fraud_df.to_csv("output/procurement_with_fraud_labels.csv", index=False)
print("\nUpdated output/procurement_with_fraud_labels.csv with model predictions")

# ============================================================
# STEP N: Hugging Face semantic text matcher (OCR vs system ID)
# ============================================================

from sentence_transformers import SentenceTransformer, util

print("\nLoading Hugging Face model (first run downloads ~90MB, please wait)...")
text_model = SentenceTransformer("all-MiniLM-L6-v2")

# Embed both the true system farmer_id and the OCR-scanned text
system_ids = fraud_df["farmer_id"].tolist()
ocr_texts = fraud_df["slip_id_ocr_text"].tolist()

system_embeddings = text_model.encode(system_ids, show_progress_bar=True, batch_size=256)
ocr_embeddings = text_model.encode(ocr_texts, show_progress_bar=True, batch_size=256)

# Cosine similarity between each pair (row-by-row, not all-vs-all)
similarities = util.pairwise_cos_sim(system_embeddings, ocr_embeddings)
fraud_df["ocr_semantic_similarity"] = similarities.numpy()

# Flag low-similarity pairs for manual review
SIMILARITY_THRESHOLD = 0.90
fraud_df["hf_flagged_ocr_mismatch"] = (fraud_df["ocr_semantic_similarity"] < SIMILARITY_THRESHOLD).astype(int)

print(f"\nStep N done: HF semantic matcher flagged {fraud_df['hf_flagged_ocr_mismatch'].sum()} rows for OCR review")

# Check how well it catches our known corrupted rows
from sklearn.metrics import classification_report
print("\nPerformance vs known OCR corruption labels:")
print(classification_report(fraud_df["needs_ocr_review"], fraud_df["hf_flagged_ocr_mismatch"],
                             target_names=["clean_ocr", "corrupted_ocr"], zero_division=0))

fraud_df.to_csv("output/procurement_with_fraud_labels.csv", index=False)
print("\nFinal dataset saved to output/procurement_with_fraud_labels.csv")
print(f"Total columns: {list(fraud_df.columns)}")
# ============================================================
# STEP N2: Add edit-distance check, combine with semantic matcher
# ============================================================

import difflib

def edit_distance_similarity(a, b):
    """Character-level similarity, 1.0 = identical, 0.0 = completely different"""
    return difflib.SequenceMatcher(None, str(a), str(b)).ratio()

fraud_df["ocr_edit_similarity"] = fraud_df.apply(
    lambda r: edit_distance_similarity(r["farmer_id"], r["slip_id_ocr_text"]), axis=1
)

EDIT_THRESHOLD = 0.95
fraud_df["edit_flagged_ocr_mismatch"] = (fraud_df["ocr_edit_similarity"] < EDIT_THRESHOLD).astype(int)

# Combine: flag if EITHER semantic OR edit-distance catches it
fraud_df["combined_flagged_ocr_mismatch"] = (
    (fraud_df["edit_flagged_ocr_mismatch"] == 1) | (fraud_df["hf_flagged_ocr_mismatch"] == 1)
).astype(int)

print(f"\nStep N2 done: Edit-distance flagged {fraud_df['edit_flagged_ocr_mismatch'].sum()} rows")
print(f"Combined (semantic OR edit-distance) flagged {fraud_df['combined_flagged_ocr_mismatch'].sum()} rows")

print("\n=== Edit-distance alone ===")
print(classification_report(fraud_df["needs_ocr_review"], fraud_df["edit_flagged_ocr_mismatch"],
                             target_names=["clean","corrupted"], zero_division=0))

print("\n=== Combined (semantic OR edit-distance) ===")
print(classification_report(fraud_df["needs_ocr_review"], fraud_df["combined_flagged_ocr_mismatch"],
                             target_names=["clean","corrupted"], zero_division=0))

fraud_df.to_csv("output/procurement_with_fraud_labels.csv", index=False)
print("Updated output/procurement_with_fraud_labels.csv with combined OCR detection")
# ============================================================
# STEP P: Capacity-mismatch fraud (declared animals vs physical max yield)
# ============================================================

typical_volume = transactions_df.groupby("farmer_id")["volume_liters"].median().reset_index()
typical_volume.columns = ["farmer_id", "typical_volume"]
farmers_df = farmers_df.merge(typical_volume, on="farmer_id", how="left")

farmers_df["mid_yield_per_animal"] = farmers_df["animal_type"].map(lambda s: sum(YIELD_RANGES[s]) / 2)
farmers_df["declared_animals"] = np.ceil(
    farmers_df["typical_volume"] / farmers_df["mid_yield_per_animal"]
).astype(int).clip(lower=1)

farmers_df["max_yield_per_animal"] = farmers_df["animal_type"].map(lambda s: YIELD_RANGES[s][1])
farmers_df["max_possible_volume"] = farmers_df["declared_animals"] * farmers_df["max_yield_per_animal"]

fraud_df = fraud_df.merge(
    farmers_df[["farmer_id", "declared_animals", "max_possible_volume"]],
    on="farmer_id", how="left"
)

N_CAPACITY_MISMATCH = 35
capacity_idx = fraud_df[fraud_df["fraud_type"] == "none"].sample(N_CAPACITY_MISMATCH, random_state=10).index
fraud_df.loc[capacity_idx, "volume_liters"] = (
    fraud_df.loc[capacity_idx, "max_possible_volume"] * rng.uniform(1.3, 1.8, size=N_CAPACITY_MISMATCH)
).round(2)
fraud_df.loc[capacity_idx, "is_fraud"] = 1
fraud_df.loc[capacity_idx, "fraud_type"] = "capacity_mismatch"

fraud_df["capacity_mismatch_flag"] = (fraud_df["volume_liters"] > fraud_df["max_possible_volume"]).astype(int)

print(f"\nStep P done: Injected {N_CAPACITY_MISMATCH} capacity-mismatch fraud rows")
print(f"Total rows flagged by capacity check: {fraud_df['capacity_mismatch_flag'].sum()}")
print("(Note: some flags overlap with volume_spike rows - both checks correctly agree those are suspicious)")

fraud_df.to_csv("output/procurement_with_fraud_labels.csv", index=False)
print("Updated output/procurement_with_fraud_labels.csv with capacity-mismatch check")
# ============================================================
# STEP M2: Retune Isolation Forest threshold for higher precision
# ============================================================

DETECTABLE_TYPES_V2 = ["volume_spike", "quality_mismatch_dilution", "baseline_deviation",
                       "temperature_anomaly", "capacity_mismatch"]
y_true_v2 = fraud_df["fraud_type"].isin(DETECTABLE_TYPES_V2).astype(int)

STRICT_PERCENTILE = 0.80
strict_threshold = np.percentile(fraud_df["anomaly_score"], STRICT_PERCENTILE)
fraud_df["model_predicted_anomaly_strict"] = (fraud_df["anomaly_score"] < strict_threshold).astype(int)

print(f"\nStep M2 done: Retuned to stricter threshold (top {STRICT_PERCENTILE}% most anomalous)")
print(classification_report(y_true_v2, fraud_df["model_predicted_anomaly_strict"],
                             target_names=["normal","fraud"], zero_division=0))

fraud_df.to_csv("output/procurement_with_fraud_labels.csv", index=False)
print("Updated output/procurement_with_fraud_labels.csv with strict-threshold predictions")
# ============================================================
# STEP M3: Rescore with Isolation Forest AFTER all fraud injected (fixes ordering bug)
# ============================================================

for col in ["volume_liters", "fat_pct", "ph", "temperature_c"]:
    fraud_df[f"{col}_zscore"] = fraud_df.groupby("animal_type")[col].transform(
        lambda x: (x - x.mean()) / x.std()
    )

FEATURES_FINAL = ["volume_liters_zscore", "fat_pct_zscore", "ph_zscore", "temperature_c_zscore"]
X_final = fraud_df[FEATURES_FINAL].fillna(0)

DETECTABLE_TYPES_FINAL = ["volume_spike", "quality_mismatch_dilution", "baseline_deviation",
                          "temperature_anomaly", "capacity_mismatch"]
y_true_final = fraud_df["fraud_type"].isin(DETECTABLE_TYPES_FINAL).astype(int)
contamination_final = y_true_final.sum() / len(fraud_df)

model_final = IsolationForest(n_estimators=200, contamination=contamination_final, random_state=42)
model_final.fit(X_final)
fraud_df["anomaly_score_final"] = model_final.decision_function(X_final)

STRICT_PERCENTILE = 0.80
strict_threshold_final = np.percentile(fraud_df["anomaly_score_final"], STRICT_PERCENTILE)
fraud_df["model_predicted_anomaly_final"] = (fraud_df["anomaly_score_final"] < strict_threshold_final).astype(int)

print(f"\nStep M3 done: Rescored AFTER all fraud injection (contamination={contamination_final:.4f})")
print(classification_report(y_true_final, fraud_df["model_predicted_anomaly_final"],
                             target_names=["normal","fraud"], zero_division=0))

fraud_df.to_csv("output/procurement_with_fraud_labels.csv", index=False)
print("Updated output/procurement_with_fraud_labels.csv with corrected final predictions")
# ============================================================
# STEP O: Save trained model for Hugging Face upload
# ============================================================

import joblib

joblib.dump(model_final, "output/isolation_forest_model.joblib")
print(f"\nStep O done: Model saved to output/isolation_forest_model.joblib")
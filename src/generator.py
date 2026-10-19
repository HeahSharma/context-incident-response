"""
generator.py - creates simulated Sentinel-style alerts, labels them using
policy/rubric.yaml, adds label noise, and saves data/alerts.json + data/dataset.csv.
 
Run from the project folder:   python src/generator.py
"""
import json
from pathlib import Path
 
import numpy as np
import pandas as pd
import yaml
 
ROOT = Path(__file__).resolve().parent.parent
SEED = 42
N_ALERTS = 700
 
# ---------------------------------------------------------------------------
# 1. Load the rubric
# ---------------------------------------------------------------------------
with open(ROOT / "policy" / "rubric.yaml") as f:
    RUBRIC = yaml.safe_load(f)
 
TIERS = RUBRIC["tiers"]                       # ordered low -> high
CUTOFFS = RUBRIC["cutoffs"]
FEATURES = RUBRIC["features"]
OVERRIDES = RUBRIC["overrides"]
NOISE_RATE = RUBRIC["noise"]["rate"]
TIER_IDX = {t: i for i, t in enumerate(TIERS)}
 
 
# ---------------------------------------------------------------------------
# 2. Scoring and labelling (this is the "policy")
# ---------------------------------------------------------------------------
def score_alert(row):
    """Add up the points for one alert."""
    total = 0
    for name, spec in FEATURES.items():
        value = row[name]
        if spec["type"] == "ordinal":
            total += value * spec["points_per_level"]
        else:
            total += value * spec["points"]
    return total
 
 
def tier_from_score(score):
    """Highest cutoff the score reaches."""
    tier = TIERS[0]
    for t in TIERS:
        if score >= CUTOFFS[t]:
            tier = t
    return TIER_IDX[tier]
 
 
def apply_overrides(row, tier_idx):
    """Override rules beat the score. Returns (tier_idx, list of rule names fired)."""
    fired = []
    for rule in OVERRIDES:
        if all(row[k] == v for k, v in rule["if"].items()):
            if "min_tier" in rule:
                tier_idx = max(tier_idx, TIER_IDX[rule["min_tier"]])
            if "max_tier" in rule:
                tier_idx = min(tier_idx, TIER_IDX[rule["max_tier"]])
            if "force_tier" in rule:
                tier_idx = TIER_IDX[rule["force_tier"]]
            fired.append(rule["name"])
    return tier_idx, fired
 
 
def add_noise(tier_idx, rng):
    """With probability NOISE_RATE, shift the label by one tier (mimics analyst disagreement)."""
    if rng.random() >= NOISE_RATE:
        return tier_idx
    if tier_idx == 0:
        return 1
    if tier_idx == len(TIERS) - 1:
        return tier_idx - 1
    return tier_idx + rng.choice([-1, 1])
 
 
# ---------------------------------------------------------------------------
# 3. Alert sampling. Each attack type has its own realistic profile.
#    Ordinal features: list of probabilities for values 0..max.
#    Binary features: probability of being 1.
# ---------------------------------------------------------------------------
PROFILES = {
    "powershell_execution": {
        "weight": 0.22,
        "severity": [0.10, 0.35, 0.40, 0.15],
        "detection_confidence": [0.15, 0.45, 0.40],
        "ti_reputation": [0.50, 0.30, 0.20],
        "mitre_stage": [0.05, 0.30, 0.45, 0.20],
        "behavior_anomaly": [0.35, 0.40, 0.25],
        "p_priv": 0.20, "p_mfa": 0.80, "p_vpn": 0.10, "p_offhours": 0.25,
        "p_maint": 0.12, "p_internet": 0.20,
    },
    "impossible_travel": {
        "weight": 0.20,
        "severity": [0.25, 0.45, 0.25, 0.05],
        "detection_confidence": [0.20, 0.50, 0.30],
        "ti_reputation": [0.65, 0.25, 0.10],
        "mitre_stage": [0.30, 0.55, 0.15, 0.00],
        "behavior_anomaly": [0.30, 0.40, 0.30],
        "p_priv": 0.10, "p_mfa": 0.80, "p_vpn": 0.40, "p_offhours": 0.30,
        "p_maint": 0.03, "p_internet": 0.50,
    },
    "ransomware": {
        "weight": 0.12,
        "severity": [0.00, 0.10, 0.35, 0.55],
        "detection_confidence": [0.05, 0.30, 0.65],
        "ti_reputation": [0.15, 0.30, 0.55],
        "mitre_stage": [0.00, 0.05, 0.30, 0.65],
        "behavior_anomaly": [0.10, 0.30, 0.60],
        "p_priv": 0.20, "p_mfa": 0.65, "p_vpn": 0.02, "p_offhours": 0.45,
        "p_maint": 0.02, "p_internet": 0.25,
    },
    "brute_force": {
        "weight": 0.22,
        "severity": [0.30, 0.40, 0.25, 0.05],
        "detection_confidence": [0.25, 0.45, 0.30],
        "ti_reputation": [0.30, 0.40, 0.30],
        "mitre_stage": [0.20, 0.55, 0.25, 0.00],
        "behavior_anomaly": [0.40, 0.40, 0.20],
        "p_priv": 0.10, "p_mfa": 0.70, "p_vpn": 0.05, "p_offhours": 0.40,
        "p_maint": 0.02, "p_internet": 0.65,
    },
    "phishing_click": {
        "weight": 0.24,
        "severity": [0.30, 0.40, 0.25, 0.05],
        "detection_confidence": [0.25, 0.45, 0.30],
        "ti_reputation": [0.40, 0.35, 0.25],
        "mitre_stage": [0.15, 0.65, 0.20, 0.00],
        "behavior_anomaly": [0.45, 0.40, 0.15],
        "p_priv": 0.05, "p_mfa": 0.85, "p_vpn": 0.08, "p_offhours": 0.20,
        "p_maint": 0.02, "p_internet": 0.15,
    },
}
 
# How likely each asset criticality level (0..4) is, in a typical hospital estate.
ASSET_CRIT_PROBS = [0.45, 0.28, 0.15, 0.08, 0.04]
ROLE_PROBS_REGULAR = [0.45, 0.35, 0.20, 0.00]
ROLE_PROBS_PRIV = [0.00, 0.10, 0.40, 0.50]
 
 
def sample_alert(attack, rng):
    p = PROFILES[attack]
    pick = lambda probs: int(rng.choice(len(probs), p=probs))
    flag = lambda prob: int(rng.random() < prob)
 
    row = {"attack_type": attack}
    row["severity"] = pick(p["severity"])
    row["detection_confidence"] = pick(p["detection_confidence"])
    row["ti_reputation"] = pick(p["ti_reputation"])
    row["mitre_stage"] = pick(p["mitre_stage"])
    row["behavior_anomaly"] = pick(p["behavior_anomaly"])
 
    row["privileged_account"] = flag(p["p_priv"])
    row["role_level"] = pick(ROLE_PROBS_PRIV if row["privileged_account"] else ROLE_PROBS_REGULAR)
    row["mfa_enabled"] = flag(p["p_mfa"])
 
    # Asset and business facts are linked: critical assets are usually
    # production, sensitive and high business impact.
    crit = pick(ASSET_CRIT_PROBS)
    row["asset_criticality"] = crit
    row["internet_facing"] = flag(p["p_internet"])
    row["sensitive_data"] = flag(0.08 + 0.15 * crit)
    impact = int(np.clip(crit + rng.integers(-1, 2), 0, 4))
    row["business_impact"] = impact
    row["production_system"] = flag(0.15 + 0.12 * crit)
    row["regulated_system"] = flag(0.08 + 0.12 * crit)
 
    row["off_hours"] = flag(p["p_offhours"])
    row["maintenance_window"] = flag(p["p_maint"])
    row["known_travel_or_vpn"] = flag(p["p_vpn"])
    return row
 
 
# ---------------------------------------------------------------------------
# 4. Build the dataset
# ---------------------------------------------------------------------------
def build_dataset(n=N_ALERTS, seed=SEED):
    rng = np.random.default_rng(seed)
    attacks = list(PROFILES)
    weights = [PROFILES[a]["weight"] for a in attacks]
 
    rows = []
    for i in range(n):
        attack = attacks[rng.choice(len(attacks), p=weights)]
        row = sample_alert(attack, rng)
        row["score"] = score_alert(row)
        base_tier = tier_from_score(row["score"])
        clean_tier, fired = apply_overrides(row, base_tier)
        noisy_tier = add_noise(clean_tier, rng)
        row["score_tier"] = TIERS[base_tier]
        row["overrides_fired"] = ";".join(fired)
        row["label_clean"] = TIERS[clean_tier]
        row["label"] = TIERS[noisy_tier]
        row["alert_id"] = f"ALT-{i + 1:04d}"
        rows.append(row)
    return pd.DataFrame(rows)
 
 
def to_sentinel_json(df):
    """Wrap each row in a simplified Sentinel-style alert record (only detection fields at this stage)."""
    alerts = []
    for r in df.to_dict("records"):
        alerts.append({
            "AlertId": r["alert_id"],
            "AlertType": r["attack_type"],
            "Severity": ["Informational", "Low", "Medium", "High"][r["severity"]],
            "ConfidenceLevel": ["Low", "Medium", "High"][r["detection_confidence"]],
            "ThreatIntelReputation": ["Clean", "Suspicious", "Malicious"][r["ti_reputation"]],
            "MitreStage": r["mitre_stage"],
        })
    return alerts
 
 
def summarize(df):
    print("\nTier counts (final noisy label):")
    print(df["label"].value_counts().reindex(TIERS).fillna(0).astype(int).to_string())
    print("\nTier share (%):")
    print((df["label"].value_counts(normalize=True).reindex(TIERS).fillna(0) * 100).round(1).to_string())
    print("\nTier share by attack type (%), clean labels:")
    print((pd.crosstab(df["attack_type"], df["label_clean"], normalize="index")
           .reindex(columns=TIERS).fillna(0) * 100).round(1).to_string())
    print("\nOverride rules fired (count):")
    print(df["overrides_fired"].replace("", "none").str.split(";").explode().value_counts().to_string())
    print(f"\nScore range: {df['score'].min()} to {df['score'].max()}, median {df['score'].median()}")
    print(f"Labels changed by noise: {(df['label'] != df['label_clean']).mean() * 100:.1f}%")
 
 
if __name__ == "__main__":
    df = build_dataset()
    (ROOT / "data").mkdir(exist_ok=True)
    df.to_csv(ROOT / "data" / "dataset.csv", index=False)
    with open(ROOT / "data" / "alerts.json", "w") as f:
        json.dump(to_sentinel_json(df), f, indent=2)
    print(f"Saved {len(df)} alerts to data/dataset.csv and data/alerts.json")
    summarize(df)

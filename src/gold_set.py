"""
gold_set.py - independent gold set to test whether the model generalises beyond the rubric.
 
STEP 1 (make the sheets):   python src/gold_set.py make
    Picks 40 alerts from the held-out TEST set (8 per attack type), writes plain-English
    descriptions for two labellers, and a hidden answer key.
        data/gold/labeller_1.csv, labeller_2.csv   -> give one to each person
        data/gold/instructions.txt                 -> give to both people
        data/gold/key.csv                          -> DO NOT show to labellers
 
STEP 2 (after both have filled in the 'your_tier' column and saved as
        labeller_1_done.csv / labeller_2_done.csv in data/gold/):
                            python src/gold_set.py score
    Prints agreement between the two people, then how the model, the rubric and the
    severity baseline compare against what the humans decided.
"""
import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
 
import model as M
 
ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "gold"
TIERS, TIER_IDX = M.TIERS, M.TIER_IDX
N_PER_TYPE = 8
 
SEV = ["Informational", "Low", "Medium", "High"]
CONF = ["Low", "Medium", "High"]
TI = ["Clean", "Suspicious", "Known malicious"]
MITRE = ["None", "Early (recon/initial access)", "Mid (execution/privilege escalation)", "Late (impact/exfiltration)"]
ROLE = ["Basic user", "Standard staff", "Senior staff / manager", "Executive / admin role"]
ANOM = ["None", "Some", "Strong"]
CRIT = ["Very low", "Low", "Medium", "High", "Life-critical"]
IMPACT = ["None", "Low", "Moderate", "High", "Severe"]
YN = lambda v: "Yes" if v else "No"
 
INSTRUCTIONS = """GOLD SET LABELLING INSTRUCTIONS
 
You are acting as a security analyst at a mid-size hospital-and-clinics group.
For each alert, read the facts and choose ONE response tier for your 'your_tier' column.
 
Tiers (type them exactly as written):
  Monitor      - No action now. Keep an eye on it.
  Investigate  - An analyst should look into it soon. No containment yet.
  Contain      - Take containment action now (isolate the host, disable the account,
                 block the address).
  Escalate     - Needs senior or management approval before acting, for example because
                 of critical systems or major patient/business impact.
 
Rules:
  1. Label on your own judgement. Do NOT discuss the alerts with the other labeller.
  2. Do not look anything up and do not use any scoring formula. Just decide what a sensible
     team would do.
  3. Fill in every row. If torn between two tiers, pick the one you would act on.
  4. Save the file as labeller_N_done.csv (keep it as .csv) and send it back.
"""
 
 
def describe(r):
    return {
        "alert_id": r["alert_id"],
        "attack_type": r["attack_type"].replace("_", " "),
        "severity": SEV[r["severity"]],
        "detection_confidence": CONF[r["detection_confidence"]],
        "threat_intel": TI[r["ti_reputation"]],
        "attack_stage": MITRE[r["mitre_stage"]],
        "privileged_account": YN(r["privileged_account"]),
        "user_role": ROLE[r["role_level"]],
        "mfa_enabled": YN(r["mfa_enabled"]),
        "unusual_user_behaviour": ANOM[r["behavior_anomaly"]],
        "asset_criticality": CRIT[r["asset_criticality"]],
        "internet_facing": YN(r["internet_facing"]),
        "holds_sensitive_data": YN(r["sensitive_data"]),
        "business_impact_if_down": IMPACT[r["business_impact"]],
        "production_system": YN(r["production_system"]),
        "regulated_system": YN(r["regulated_system"]),
        "outside_working_hours": YN(r["off_hours"]),
        "in_maintenance_window": YN(r["maintenance_window"]),
        "known_travel_or_vpn": YN(r["known_travel_or_vpn"]),
        "your_tier": "",
    }
 
 
def split_test():
    df = pd.read_csv(ROOT / "data" / "dataset.csv")
    train, test = train_test_split(df, test_size=0.3, random_state=M.SEED, stratify=df["label"])
    return train, test
 
 
def make():
    train, test = split_test()
    rng = np.random.default_rng(7)
    picks = []
    for _, grp in test.groupby("attack_type"):
        idx = rng.choice(len(grp), size=min(N_PER_TYPE, len(grp)), replace=False)
        picks.append(grp.iloc[idx])
    gold = pd.concat(picks)
 
    model = M.fit_model(train)
    pred = model.predict_proba(gold[M.FEATURES].to_numpy(dtype=float)).argmax(axis=1)
 
    key = gold[["alert_id", "attack_type", "label_clean", "severity"]].copy()
    key["model_pred"] = [TIERS[i] for i in pred]
    key["baseline_pred"] = [TIERS[i] for i in gold["severity"]]
    key = key.rename(columns={"label_clean": "rubric_label"}).drop(columns="severity")
 
    GOLD.mkdir(parents=True, exist_ok=True)
    key.to_csv(GOLD / "key.csv", index=False)
    (GOLD / "instructions.txt").write_text(INSTRUCTIONS)
 
    sheet = pd.DataFrame([describe(r) for r in gold.to_dict("records")])
    for n, seed in [(1, 11), (2, 22)]:
        sheet.sample(frac=1, random_state=seed).to_csv(GOLD / f"labeller_{n}.csv", index=False)
    print(f"Made {len(sheet)} gold alerts in data/gold/:")
    print("  labeller_1.csv, labeller_2.csv, instructions.txt (give to labellers)")
    print("  key.csv (keep hidden until both are done)")
 
 
def qwk(a, b):
    return round(float(cohen_kappa_score(a, b, weights="quadratic", labels=range(4))), 3)
 
 
def mae(a, b):
    return round(float(np.abs(np.asarray(a) - np.asarray(b)).mean()), 3)
 
 
def score():
    key = pd.read_csv(GOLD / "key.csv")
    labs = []
    for n in (1, 2):
        d = pd.read_csv(GOLD / f"labeller_{n}_done.csv")[["alert_id", "your_tier"]]
        d["your_tier"] = d["your_tier"].astype(str).str.strip().str.capitalize()
        bad = d[~d["your_tier"].isin(TIERS)]
        if len(bad):
            sys.exit(f"labeller_{n}_done.csv has blank or misspelt tiers in rows: {bad['alert_id'].tolist()[:10]}")
        labs.append(d.rename(columns={"your_tier": f"l{n}"}))
    df = key.merge(labs[0], on="alert_id").merge(labs[1], on="alert_id")
    for c in ["l1", "l2", "rubric_label", "model_pred", "baseline_pred"]:
        df[c] = df[c].map(TIER_IDX)
 
    print(f"Gold alerts: {len(df)}")
    print("\nAgreement between the two human labellers:")
    print(f"  exact agreement: {(df['l1'] == df['l2']).mean() * 100:.1f}%")
    print(f"  quadratic weighted kappa: {qwk(df['l1'], df['l2'])}")
 
    agree = df[df["l1"] == df["l2"]]
    print(f"\nAlerts where both humans agree: {len(agree)} (used as the gold label below)")
    rows = []
    for name, col in [("Severity baseline", "baseline_pred"), ("Ordinal model", "model_pred"), ("Rubric itself", "rubric_label")]:
        rows.append({
            "system": name,
            "qwk_vs_human_agreed": qwk(agree["l1"], agree[col]),
            "mae_vs_human_agreed": mae(agree["l1"], agree[col]),
            "accuracy_vs_human_agreed": round(float((agree["l1"] == agree[col]).mean()), 3),
            "qwk_vs_labeller_1": qwk(df["l1"], df[col]),
            "qwk_vs_labeller_2": qwk(df["l2"], df[col]),
        })
    out = pd.DataFrame(rows)
    print("\n" + out.to_string(index=False))
    out.to_csv(ROOT / "outputs" / "gold_results.csv", index=False)
    print("\nSaved outputs/gold_results.csv")
 
 
if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("make", "score"):
        sys.exit("Usage: python src/gold_set.py make   |   python src/gold_set.py score")
    make() if sys.argv[1] == "make" else score()

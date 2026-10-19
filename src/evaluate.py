"""
evaluate.py - ablation and counterfactual experiments.
 
1. Ablation: remove one context dimension at a time, refit, and see how much the
   score drops. A big drop means that dimension matters.
2. Counterfactual: change one fact on test alerts (e.g. turn MFA on) and see how
   the predicted tier moves.
 
Run from the project folder:   python src/evaluate.py
Outputs: outputs/ablation.csv, outputs/counterfactuals.csv, outputs/ablation.png
"""
from pathlib import Path
 
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
 
import model as M
 
ROOT = Path(__file__).resolve().parent.parent
TIERS, TIER_IDX = M.TIERS, M.TIER_IDX
 
with open(ROOT / "policy" / "rubric.yaml") as f:
    SPEC = yaml.safe_load(f)["features"]
DIMS = sorted({s["dim"] for s in SPEC.values()})
 
 
def fit_and_score(train, test, feature_list):
    """Fit on a subset of features, return (qwk vs clean, mae vs clean, model)."""
    X = train[feature_list].to_numpy(dtype=float)
    y = train["label"].map(TIER_IDX).to_numpy()
    mdl = M.OrdinalLogit().fit(X, y)
    pred = mdl.predict_proba(test[feature_list].to_numpy(dtype=float)).argmax(axis=1)
    y_clean = test["label_clean"].map(TIER_IDX).to_numpy()
    qwk = cohen_kappa_score(y_clean, pred, weights="quadratic")
    mae = np.abs(y_clean - pred).mean()
    return round(qwk, 3), round(mae, 3), mdl
 
 
def expected_tier(mdl, df, feature_list):
    p = mdl.predict_proba(df[feature_list].to_numpy(dtype=float))
    return p @ np.arange(4)
 
 
if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "dataset.csv")
    train, test = train_test_split(df, test_size=0.3, random_state=M.SEED, stratify=df["label"])
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
 
    # ---- 1. Ablation ----
    full_qwk, full_mae, full_model = fit_and_score(train, test, M.FEATURES)
    rows = [{"setting": "all 5 dimensions", "features_used": len(M.FEATURES),
             "qwk_vs_clean": full_qwk, "mae_vs_clean": full_mae, "qwk_drop": 0.0}]
    for dim in DIMS:
        keep = [f for f in M.FEATURES if SPEC[f]["dim"] != dim]
        q, m, _ = fit_and_score(train, test, keep)
        rows.append({"setting": f"without {dim}", "features_used": len(keep),
                     "qwk_vs_clean": q, "mae_vs_clean": m, "qwk_drop": round(full_qwk - q, 3)})
    q, m, _ = fit_and_score(train, test, ["severity"])
    rows.append({"setting": "severity only", "features_used": 1,
                 "qwk_vs_clean": q, "mae_vs_clean": m, "qwk_drop": round(full_qwk - q, 3)})
    abl = pd.DataFrame(rows)
    abl.to_csv(out / "ablation.csv", index=False)
    print("Ablation (scored against clean labels, test set):")
    print(abl.to_string(index=False))
 
    # ---- 2. Counterfactuals ----
    base_exp = expected_tier(full_model, test, M.FEATURES)
    cf_defs = {
        "turn MFA on":              {"mfa_enabled": 1},
        "turn MFA off":             {"mfa_enabled": 0},
        "make account privileged":  {"privileged_account": 1},
        "put in maintenance window": {"maintenance_window": 1},
        "mark as known VPN/travel": {"known_travel_or_vpn": 1},
        "move to off-hours":        {"off_hours": 1},
        "asset criticality +1":     "crit_up",
        "threat intel = malicious": {"ti_reputation": 2},
    }
    cf_rows = []
    for name, change in cf_defs.items():
        cf = test.copy()
        if change == "crit_up":
            cf["asset_criticality"] = (cf["asset_criticality"] + 1).clip(upper=SPEC["asset_criticality"]["max"])
        else:
            for k, v in change.items():
                cf[k] = v
        new_pred = full_model.predict_proba(cf[M.FEATURES].to_numpy(dtype=float)).argmax(axis=1)
        old_pred = full_model.predict_proba(test[M.FEATURES].to_numpy(dtype=float)).argmax(axis=1)
        shift = expected_tier(full_model, cf, M.FEATURES) - base_exp
        cf_rows.append({
            "change": name,
            "avg_shift_in_expected_tier": round(float(shift.mean()), 3),
            "pct_alerts_changing_tier": round(float((new_pred != old_pred).mean() * 100), 1),
            "pct_moved_up": round(float((new_pred > old_pred).mean() * 100), 1),
            "pct_moved_down": round(float((new_pred < old_pred).mean() * 100), 1),
        })
    cf_df = pd.DataFrame(cf_rows)
    cf_df.to_csv(out / "counterfactuals.csv", index=False)
    print("\nCounterfactuals on test alerts:")
    print(cf_df.to_string(index=False))
 
    # ---- Chart ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
 
        plot = abl.copy()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(plot["setting"][::-1], plot["qwk_vs_clean"][::-1])
        ax.set_xlabel("Quadratic weighted kappa vs clean labels (test set)")
        ax.set_title("Ablation: effect of removing each context dimension")
        ax.set_xlim(0, 1)
        plt.tight_layout()
        fig.savefig(out / "ablation.png", dpi=150)
        print("\nSaved outputs/ablation.csv, counterfactuals.csv, ablation.png")
    except ImportError:
        print("\nSaved outputs/ablation.csv, counterfactuals.csv (matplotlib not found, no chart)")
"""
explain.py - shows WHY the model picked a tier for an alert.
 
For each alert, contribution of a feature = coefficient * (how far the alert's value
is from the average alert). Positive pushes toward a higher tier, negative toward lower.
 
Run from the project folder:   python src/explain.py
Output: prints 5 example explanations and saves outputs/explanations.csv
"""
from pathlib import Path
 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
 
from model import FEATURES, SEED, TIERS, fit_model, predict_probs
 
ROOT = Path(__file__).resolve().parent.parent
 
 
def explain_alert(model, row, top_n=3):
    x = row[FEATURES].to_numpy(dtype=float)
    contrib = model.beta * (x - model.mu) / model.sd
    order = np.argsort(-np.abs(contrib))[:top_n]
    return [(FEATURES[i], float(contrib[i]), row[FEATURES[i]]) for i in order]
 
 
if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "dataset.csv")
    train, test = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df["label"])
    model = fit_model(train)
    probs = predict_probs(model, test)
 
    records = []
    for i, (_, row) in enumerate(test.iterrows()):
        top = explain_alert(model, row)
        p = probs[i]
        records.append({
            "alert_id": row["alert_id"],
            "attack_type": row["attack_type"],
            "predicted_tier": TIERS[int(p.argmax())],
            "true_tier_clean": row["label_clean"],
            **{f"p_{t}": round(float(p[k]), 3) for k, t in enumerate(TIERS)},
            "top_factors": "; ".join(
                f"{name}={val} ({'+' if c > 0 else '-'}{abs(c):.2f})" for name, c, val in top
            ),
        })
    out = pd.DataFrame(records)
    (ROOT / "outputs").mkdir(exist_ok=True)
    out.to_csv(ROOT / "outputs" / "explanations.csv", index=False)
 
    for _, r in out.head(5).iterrows():
        print(f"\n{r['alert_id']} ({r['attack_type']}): predicted {r['predicted_tier']} "
              f"(rubric says {r['true_tier_clean']})")
        print("  probabilities:", {t: r[f'p_{t}'] for t in TIERS})
        print("  main factors: ", r["top_factors"])
    print("\nSaved outputs/explanations.csv")
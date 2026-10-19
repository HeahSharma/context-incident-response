"""
baseline.py - the severity-only baseline.
 
It ignores all context and maps the alert's severity straight to a response tier:
    severity 0 (Informational) -> Monitor
    severity 1 (Low)           -> Investigate
    severity 2 (Medium)        -> Contain
    severity 3 (High)          -> Escalate
Then it scores that against the true labels. Your model must beat these numbers.
 
Run from the project folder:   python src/baseline.py
"""
from pathlib import Path
 
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support
 
ROOT = Path(__file__).resolve().parent.parent
TIERS = ["Monitor", "Investigate", "Contain", "Escalate"]
TIER_IDX = {t: i for i, t in enumerate(TIERS)}
 
# The mapping a traditional severity-driven setup would use.
SEVERITY_TO_TIER = {0: "Monitor", 1: "Investigate", 2: "Contain", 3: "Escalate"}
 
 
def predict(df):
    return df["severity"].map(SEVERITY_TO_TIER)
 
 
def evaluate(true_labels, predicted, name):
    y_true = true_labels.map(TIER_IDX).to_numpy()
    y_pred = predicted.map(TIER_IDX).to_numpy()
 
    accuracy = (y_true == y_pred).mean()
    mae = np.abs(y_true - y_pred).mean()
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    prec, rec, _, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(4), zero_division=0
    )
 
    print(f"\n=== Baseline vs {name} ===")
    print(f"Accuracy:                     {accuracy:.3f}")
    print(f"Mean absolute error (tiers):  {mae:.3f}")
    print(f"Quadratic weighted kappa:     {qwk:.3f}")
    print("\nPer-tier results:")
    print(pd.DataFrame(
        {"precision": prec.round(3), "recall": rec.round(3), "n_true": support},
        index=TIERS,
    ).to_string())
    print("\nConfusion matrix (rows = true tier, columns = predicted):")
    print(pd.crosstab(
        pd.Categorical(true_labels, categories=TIERS),
        pd.Categorical(predicted, categories=TIERS),
        rownames=["true"], colnames=["predicted"], dropna=False,
    ).to_string())
 
 
if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "dataset.csv")
    pred = predict(df)
    evaluate(df["label"], pred, "noisy labels (what the model trains on)")
    evaluate(df["label_clean"], pred, "clean labels (the rubric's true answer)")

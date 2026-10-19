"""
model.py - ordinal logistic regression on the 17 context features.
 
Trains on the NOISY labels (like real analyst labels), then compares against the
severity-only baseline on a held-out test set, scored against both noisy and clean labels.
 
Run from the project folder:   python src/model.py
Outputs: outputs/metrics.csv, outputs/model_coefficients.csv, outputs/test_predictions.csv
"""
from pathlib import Path
 
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from scipy.special import expit
 
ROOT = Path(__file__).resolve().parent.parent
TIERS = ["Monitor", "Investigate", "Contain", "Escalate"]
TIER_IDX = {t: i for i, t in enumerate(TIERS)}
SEED = 42
 
with open(ROOT / "policy" / "rubric.yaml") as f:
    FEATURES = list(yaml.safe_load(f)["features"].keys())
 
SEVERITY_TO_TIER = {0: 0, 1: 1, 2: 2, 3: 3}   # same mapping as baseline.py
 
 
def metrics(y_true, y_pred):
    return {
        "accuracy": round(float((y_true == y_pred).mean()), 3),
        "mae_tiers": round(float(np.abs(y_true - y_pred).mean()), 3),
        "qwk": round(float(cohen_kappa_score(y_true, y_pred, weights="quadratic")), 3),
    }
 
 
class OrdinalLogit:
    """
    Proportional-odds (cumulative logit) ordinal regression.
        P(tier <= k) = sigmoid(threshold_k - x . beta)
    One coefficient per feature, plus 3 ordered thresholds between the 4 tiers.
    Features are standardised so coefficients are comparable (used for explanations).
    """
 
    def fit(self, X, y, n_classes=4, l2=1e-3):
        self.K = n_classes
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        Z = (X - self.mu) / self.sd
        n, p = Z.shape
 
        def unpack(w):
            beta = w[:p]
            t = np.cumsum(np.concatenate([[w[p]], np.exp(w[p + 1:])]))
            return beta, t
 
        def nll(w):
            beta, t = unpack(w)
            eta = Z @ beta
            cdf = expit(t[None, :] - eta[:, None])                       # n x (K-1)
            cdf = np.hstack([np.zeros((n, 1)), cdf, np.ones((n, 1))])
            prob = np.clip(cdf[np.arange(n), y + 1] - cdf[np.arange(n), y], 1e-12, None)
            return -np.log(prob).sum() + l2 * (beta ** 2).sum()
 
        w0 = np.concatenate([np.zeros(p), [-1.0], np.zeros(self.K - 2)])
        res = minimize(nll, w0, method="L-BFGS-B", options={"maxiter": 2000})
        self.beta, self.thresholds = unpack(res.x)
        self.converged = res.success
        return self
 
    def predict_proba(self, X):
        Z = (X - self.mu) / self.sd
        eta = Z @ self.beta
        cdf = expit(self.thresholds[None, :] - eta[:, None])
        cdf = np.hstack([np.zeros((len(Z), 1)), cdf, np.ones((len(Z), 1))])
        return np.diff(cdf, axis=1)
 
 
def fit_model(train_df):
    X = train_df[FEATURES].to_numpy(dtype=float)
    y = train_df["label"].map(TIER_IDX).to_numpy()
    return OrdinalLogit().fit(X, y)
 
 
def predict_probs(model, df):
    """Returns an (n, 4) array of tier probabilities."""
    return model.predict_proba(df[FEATURES].to_numpy(dtype=float))
 
 
if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "dataset.csv")
    train, test = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df["label"])
    print(f"Train: {len(train)} alerts, test: {len(test)} alerts")
 
    result = fit_model(train)
    print("Model converged:", result.converged)
    probs = predict_probs(result, test)
    pred = probs.argmax(axis=1)
    base_pred = test["severity"].map(SEVERITY_TO_TIER).to_numpy()
 
    y_noisy = test["label"].map(TIER_IDX).to_numpy()
    y_clean = test["label_clean"].map(TIER_IDX).to_numpy()
 
    rows = []
    for model_name, p in [("severity_baseline", base_pred), ("ordinal_logit", pred)]:
        for target, y in [("noisy_labels", y_noisy), ("clean_labels", y_clean)]:
            rows.append({"model": model_name, "scored_against": target, **metrics(y, p)})
    table = pd.DataFrame(rows)
    print("\nResults on held-out test set:")
    print(table.to_string(index=False))
 
    # Where does the model go wrong? Alerts where an override rule fired
    # (non-linear rules a linear model cannot fully learn).
    fired = test["overrides_fired"].fillna("") != ""
    print("\nQWK vs clean labels, by whether an override rule fired:")
    for name, mask in [("no override", ~fired), ("override fired", fired)]:
        if mask.sum() > 5:
            m = metrics(y_clean[mask.to_numpy()], pred[mask.to_numpy()])
            print(f"  {name} (n={int(mask.sum())}): {m}")
 
    # Save outputs
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "metrics.csv", index=False)
 
    coef_df = pd.DataFrame({"feature": FEATURES, "coefficient": result.beta})
    coef_df["abs"] = coef_df["coefficient"].abs()
    coef_df.sort_values("abs", ascending=False).drop(columns="abs").to_csv(
        out / "model_coefficients.csv", index=False
    )
 
    pred_df = test[["alert_id", "attack_type", "label", "label_clean"]].copy()
    pred_df["baseline_pred"] = [TIERS[i] for i in base_pred]
    pred_df["model_pred"] = [TIERS[i] for i in pred]
    for i, t in enumerate(TIERS):
        pred_df[f"p_{t}"] = probs[:, i].round(3)
    pred_df.to_csv(out / "test_predictions.csv", index=False)
    print("\nSaved outputs/metrics.csv, model_coefficients.csv, test_predictions.csv")
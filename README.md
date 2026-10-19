# Ordinal Risk-Based Incident Response Decisioning


## Overview
Modern SIEM tools such as Microsoft Sentinel are good at detecting threats, but deciding the right response is still left to static SOAR playbooks or manual analyst judgment. This project builds a Python prototype that sits between detection and automation. It takes a simulated Sentinel-style alert, enriches it with five context dimensions, and recommends one of four response tiers with an explanation.

## Research Question
Can contextual information improve incident response decisions compared with a traditional severity-only approach?

## Approach
- **Context dimensions:** security, user, asset, business, environmental (17 features in total)
- **Response tiers:** Monitor, Investigate, Contain, Escalate
- **Model:** ordinal logistic regression, so the output is explainable (top contributing factors and tier probabilities)
- **Data:** simulated alerts for 5 attack types (PowerShell execution, impossible travel, ransomware, brute force, phishing click), labelled using a written policy rubric
- **Evaluation:** compared against a severity-only baseline using quadratic weighted kappa, mean absolute error, and per-tier precision and recall, plus ablation and counterfactual experiments

## How to Run
python src/generator.py    # creates data/dataset.csv and data/alerts.json
python src/baseline.py     # severity-only baseline
python src/model.py        # ordinal model vs baseline
python src/explain.py      # per-alert explanations
python src/evaluate.py     # ablation and counterfactuals

## Repository Contents
- `policy/rubric.yaml` - frozen response policy and labelling rubric (fictional hospital group)
- `src/` - Python scripts: `generator.py`, `baseline.py`, `model.py`, `explain.py`, `evaluate.py`, `gold_set.py`
- `data/` - simulated alerts (`dataset.csv`, `alerts.json`)
- `outputs/` - results: metrics, model coefficients, explanations, ablation and counterfactual tables, ablation chart

## Status
Prototype pipeline on simulated data: rubric frozen, alert generator, severity baseline, ordinal logistic model, explanations, ablation and counterfactuals. In progress: independently labelled gold set (40 alerts, two labellers) and the final report.

## Tech Stack
Python, pandas, numpy, scipy, scikit-learn, matplotlib, PyYAML. The ordinal (proportional-odds) logistic regression is implemented directly with scipy. All data is simulated and local.

## Limitations
- All alerts are simulated and labelled by a fictional policy, so results show the model recovers that policy, not real-world accuracy.
- The labelling rubric is a points sum and the model is also linear, so scores against rubric labels are inflated. A small independently labelled gold set is used to check this.
- Feature enrichment is simulated inside the generator.

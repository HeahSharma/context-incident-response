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

## Repository Contents
- `docs/Prototype_Plan.pdf` - full prototype plan
- `policy/rubric.yaml` - draft response policy and labelling rubric
- `src/` - Python modules (in progress)

## Status
Work in progress. Prototype plan and rubric drafted.

## Tech Stack
Python, pandas, numpy, statsmodels, scikit-learn, matplotlib, PyYAML. All data is simulated and local.

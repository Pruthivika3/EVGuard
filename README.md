# EVGuard AI — Advanced Machine Learning Project

## Hybrid XGBoost + BiLSTM Framework for Early Detection of EV Battery Thermal Risk

This project is a complete starter framework for an Advanced Machine Learning assignment.

### Models
1. Logistic Regression — baseline
2. XGBoost — advanced tabular model
3. BiLSTM + Attention — deep temporal model

### Application
Streamlit dashboard for CSV upload and risk prediction.

## Dataset
The code expects a CSV containing battery telemetry. A synthetic demo dataset generator is included so the pipeline can be tested immediately.

Recommended columns:
- timestamp
- battery_id
- temperature
- voltage
- current
- soc
- soh
- internal_resistance
- risk_label

`risk_label` can be:
- 0 = Normal
- 1 = Warning
- 2 = Critical

For the final academic submission, replace the demo data with a properly sourced benchmark/real-world dataset and document its source.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Generate demo data:

```bash
python src/generate_demo_data.py
```

Train XGBoost:

```bash
python src/train_xgboost.py
```

Train BiLSTM:

```bash
python src/train_bilstm.py
```

Run the application:

```bash
streamlit run app/app.py
```

## Important
The included synthetic data is only for software testing. Do not report synthetic-data results as research findings.

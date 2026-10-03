import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import streamlit as st
from src.preprocessing import add_temporal_features, get_engineered_features
from config import MODEL_DIR

st.set_page_config(page_title="EVGuard AI", page_icon="🔋", layout="wide")

st.title("🔋 EVGuard AI")
st.subheader("Early EV Battery Thermal-Risk Detection")

st.info(
    "Educational AML demonstration. Predictions are not a safety certification "
    "and must not be used to operate or protect a real battery system."
)

model_path = MODEL_DIR / "xgboost_bundle.joblib"

if not model_path.exists():
    st.warning("Train the XGBoost model first: python src/train_xgboost.py")
    st.stop()

bundle = joblib.load(model_path)
model = bundle["model"]
features = bundle["features"]

uploaded = st.file_uploader("Upload battery telemetry CSV", type=["csv"])

if uploaded:
    raw = pd.read_csv(uploaded)
    df = add_temporal_features(raw)
else:
    demo_path = Path(__file__).resolve().parents[1] / "data" / "demo_battery_data.csv"
    if not demo_path.exists():
        st.warning("Run python src/generate_demo_data.py first.")
        st.stop()
    df = add_temporal_features(pd.read_csv(demo_path))

missing = [c for c in features if c not in df.columns]
if missing:
    st.error(f"Missing engineered features: {missing}")
    st.stop()

X = df[features].replace([float("inf"), float("-inf")], 0).fillna(0)
pred = model.predict(X)
prob = model.predict_proba(X).max(axis=1)

labels = {0: "NORMAL", 1: "WARNING", 2: "CRITICAL"}
df["prediction"] = [labels[int(x)] for x in pred]
df["confidence"] = prob

c1, c2, c3 = st.columns(3)
c1.metric("Samples", len(df))
c2.metric("Critical predictions", int((pred == 2).sum()))
c3.metric("Maximum confidence", f"{prob.max()*100:.1f}%")

st.dataframe(
    df[["timestamp", "battery_id", "temperature", "voltage", "current",
        "soc", "prediction", "confidence"]].tail(100),
    use_container_width=True
)

st.subheader("Telemetry")
st.line_chart(df.set_index("timestamp")[["temperature", "current", "voltage"]].tail(500))

st.subheader("Risk distribution")
st.bar_chart(df["prediction"].value_counts())

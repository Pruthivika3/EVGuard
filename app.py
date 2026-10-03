import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EVGuard AI",
    page_icon="🔋",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

FEATURES = [
    "cycle",
    "time",
    "voltage",
    "current",
    "temperature",
    "soc",
    "internal_resistance"
]

LABELS = [
    "CRITICAL",
    "NORMAL",
    "WARNING"
]


# ============================================================
# BI-LSTM + ATTENTION MODEL
# ============================================================

class Attention(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()

        self.attention = nn.Linear(
            hidden_dim,
            1
        )

    def forward(self, x):

        scores = self.attention(x).squeeze(-1)

        weights = torch.softmax(
            scores,
            dim=1
        )

        context = torch.sum(
            x * weights.unsqueeze(-1),
            dim=1
        )

        return context


class BiLSTMAttention(nn.Module):

    def __init__(
        self,
        input_size=7,
        hidden_size=64,
        num_layers=2,
        num_classes=3,
        dropout=0.2
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.attention = Attention(
            hidden_size * 2
        )

        self.fc = nn.Sequential(
            nn.Linear(
                hidden_size * 2,
                64
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                64,
                num_classes
            )
        )

    def forward(self, x):

        x, _ = self.lstm(x)

        x = self.attention(x)

        x = self.fc(x)

        return x


# ============================================================
# LOAD XGBOOST
# ============================================================

@st.cache_resource
def load_xgboost():

    model_path = MODEL_DIR / "xgboost_evguard.pkl"

    with open(
        model_path,
        "rb"
    ) as file:

        model = pickle.load(file)

    return model


# ============================================================
# LOAD BILSTM
# ============================================================
@st.cache_resource
def load_bilstm():

    model_path = (
        MODEL_DIR /
        "bilstm_attention_evguard.pt"
    )

    # Load complete checkpoint
    checkpoint = torch.load(
        model_path,
        map_location="cpu"
    )

    # Read architecture information saved during training
    input_size = checkpoint.get(
        "input_size",
        7
    )

    hidden_size = checkpoint.get(
        "hidden_size",
        64
    )

    num_layers = checkpoint.get(
        "num_layers",
        2
    )

    num_classes = checkpoint.get(
        "num_classes",
        3
    )

    # Create model with the same architecture
    model = BiLSTMAttention(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes
    )

    # Load ONLY the actual model weights
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # Load scaler
    scaler_path = (
        MODEL_DIR /
        "bilstm_scaler.pkl"
    )

    with open(
        scaler_path,
        "rb"
    ) as file:

        scaler = pickle.load(file)

    return model, scaler
# ============================================================
# XGBOOST PREDICTION
# ============================================================

def predict_xgboost(values):

    model = load_xgboost()

    data = pd.DataFrame(
        [values],
        columns=FEATURES
    )

    prediction = int(
        model.predict(data)[0]
    )

    probabilities = (
        model.predict_proba(data)[0]
    )

    label = LABELS[prediction]

    return label, probabilities


# ============================================================
# BILSTM PREDICTION
# ============================================================

def predict_bilstm(data):

    model, scaler = load_bilstm()

    X = data[
        FEATURES
    ].astype(float).values

    X_scaled = scaler.transform(X)

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32
    )

    X_tensor = X_tensor.unsqueeze(0)

    with torch.no_grad():

        output = model(
            X_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )[0].numpy()

    prediction = int(
        np.argmax(probabilities)
    )

    label = LABELS[prediction]

    return label, probabilities


# ============================================================
# HEADER
# ============================================================

st.title("🔋 EVGuard AI")

st.subheader(
    "Battery Health and Degradation Risk Prediction"
)

st.write(
    "An Advanced Machine Learning application "
    "for EV battery degradation risk prediction "
    "using XGBoost and BiLSTM with Attention."
)

st.info(
    "Risk definition used in this project: "
    "NORMAL = SOH ≥ 90%, "
    "WARNING = 80%–<90%, "
    "CRITICAL = SOH < 80%."
)


# ============================================================
# TABS
# ============================================================

tab1, tab2 = st.tabs(
    [
        "📊 Single Reading - XGBoost",
        "🧠 30-Point Sequence - BiLSTM"
    ]
)


# ============================================================
# TAB 1
# ============================================================

with tab1:

    st.header(
        "Single Battery Reading"
    )

    st.write(
        "Enter one telemetry reading "
        "to obtain an XGBoost prediction."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        cycle = st.number_input(
            "Cycle",
            min_value=1.0,
            value=100.0
        )

        time_value = st.number_input(
            "Time (seconds)",
            min_value=0.0,
            value=1500.0
        )

        voltage = st.number_input(
            "Voltage (V)",
            value=3.5
        )

    with col2:

        current = st.number_input(
            "Current (A)",
            value=-1.75
        )

        temperature = st.number_input(
            "Temperature (°C)",
            value=30.0
        )

        soc = st.number_input(
            "SOC",
            min_value=0.0,
            max_value=1.0,
            value=0.5
        )

    with col3:

        resistance = st.number_input(
            "Internal Resistance",
            min_value=0.0,
            value=1.75
        )

    if st.button(
        "Predict Battery Risk",
        type="primary"
    ):

        values = {
            "cycle": cycle,
            "time": time_value,
            "voltage": voltage,
            "current": current,
            "temperature": temperature,
            "soc": soc,
            "internal_resistance": resistance
        }

        try:

            label, probabilities = (
                predict_xgboost(values)
            )

            st.subheader(
                "Prediction Result"
            )

            if label == "CRITICAL":

                st.error(
                    f"🔴 {label}"
                )

            elif label == "WARNING":

                st.warning(
                    f"🟠 {label}"
                )

            else:

                st.success(
                    f"🟢 {label}"
                )

            confidence = (
                probabilities.max()
                * 100
            )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            probability_df = pd.DataFrame(
                {
                    "Risk Class": LABELS,
                    "Probability": probabilities
                }
            )

            st.subheader(
                "Class Probabilities"
            )

            st.bar_chart(
                probability_df.set_index(
                    "Risk Class"
                )
            )

        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )


# ============================================================
# TAB 2
# ============================================================

with tab2:

    st.header(
        "30-Point Battery Sequence"
    )

    st.write(
        "Upload a CSV containing exactly "
        "30 consecutive telemetry points."
    )

    st.caption(
        "Required columns: "
        + ", ".join(FEATURES)
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        data = pd.read_csv(
            uploaded_file
        )

        missing_columns = [
            column
            for column in FEATURES
            if column not in data.columns
        ]

        if missing_columns:

            st.error(
                "Missing columns: "
                + ", ".join(
                    missing_columns
                )
            )

        elif len(data) != 30:

            st.error(
                f"The file contains "
                f"{len(data)} rows. "
                f"Exactly 30 rows are required."
            )

        else:

            st.success(
                "Valid 30-point sequence."
            )

            st.dataframe(
                data[FEATURES],
                use_container_width=True
            )

            if st.button(
                "Predict Sequence Risk",
                type="primary"
            ):

                try:

                    label, probabilities = (
                        predict_bilstm(data)
                    )

                    st.subheader(
                        "BiLSTM Prediction"
                    )

                    if label == "CRITICAL":

                        st.error(
                            f"🔴 {label}"
                        )

                    elif label == "WARNING":

                        st.warning(
                            f"🟠 {label}"
                        )

                    else:

                        st.success(
                            f"🟢 {label}"
                        )

                    confidence = (
                        probabilities.max()
                        * 100
                    )

                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%"
                    )

                    probability_df = pd.DataFrame(
                        {
                            "Risk Class": LABELS,
                            "Probability": probabilities
                        }
                    )

                    st.subheader(
                        "Class Probabilities"
                    )

                    st.bar_chart(
                        probability_df.set_index(
                            "Risk Class"
                        )
                    )

                except Exception as e:

                    st.error(
                        f"Prediction error: {e}"
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "EVGuard AI | Advanced Machine Learning - 19MAM51 | "
    "Educational/research demonstration only."
)
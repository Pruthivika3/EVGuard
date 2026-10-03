import pandas as pd
import numpy as np

BASE_FEATURES = [
    "temperature", "voltage", "current",
    "soc", "soh", "internal_resistance"
]

def add_temporal_features(df, group_col="battery_id"):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values([group_col, "timestamp"])

    for col in BASE_FEATURES:
        df[f"{col}_diff"] = df.groupby(group_col)[col].diff().fillna(0)
        df[f"{col}_roll_mean_10"] = (
            df.groupby(group_col)[col]
            .transform(lambda x: x.rolling(10, min_periods=1).mean())
        )
        df[f"{col}_roll_std_10"] = (
            df.groupby(group_col)[col]
            .transform(lambda x: x.rolling(10, min_periods=1).std())
            .fillna(0)
        )

    return df

def get_engineered_features(df):
    engineered = add_temporal_features(df)
    exclude = {"timestamp", "battery_id", "risk_label"}
    return [c for c in engineered.columns if c not in exclude]

def load_data(path):
    df = pd.read_csv(path)
    required = BASE_FEATURES + ["timestamp", "battery_id", "risk_label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return add_temporal_features(df)

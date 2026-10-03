import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "ml_dataset.csv"
SCALER_FILE = BASE_DIR / "models" / "bilstm_scaler.pkl"

FEATURES = [
    "cycle",
    "time",
    "voltage",
    "current",
    "temperature",
    "soc",
    "internal_resistance"
]

# Same batteries used for model training
TRAIN_BATTERIES = [
    "B0005",
    "B0007",
    "B0018"
]

print("Loading dataset...")

df = pd.read_csv(DATA_FILE)

train_df = df[
    df["battery_id"].isin(TRAIN_BATTERIES)
].copy()

X_train = train_df[FEATURES].astype(float)

print("Training rows:", len(X_train))
print("Features:", FEATURES)

# Create scaler
scaler = StandardScaler()

scaler.fit(X_train)

# Save clean scaler
with open(
    SCALER_FILE,
    "wb"
) as f:
    pickle.dump(
        scaler,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )

print()
print("Scaler recreated successfully.")
print("Saved to:")
print(SCALER_FILE)
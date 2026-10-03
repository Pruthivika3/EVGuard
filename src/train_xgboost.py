import pandas as pd
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "ml_dataset.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "cycle",
    "time",
    "voltage",
    "current",
    "temperature",
    "soc",
    "internal_resistance"
]

TARGET = "risk"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)

X = df[FEATURES].copy()
y = df[TARGET].copy()


# ============================================================
# ENCODE TARGET
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print()
print("Classes:")
print(label_encoder.classes_)


# ============================================================
# BATTERY-LEVEL SPLIT
# ============================================================

train_batteries = [
    "B0005",
    "B0007",
    "B0018"
]

test_battery = "B0006"

train_df = df[
    df["battery_id"].isin(train_batteries)
].copy()

test_df = df[
    df["battery_id"] == test_battery
].copy()

X_train = train_df[FEATURES]
X_test = test_df[FEATURES]

y_train = label_encoder.transform(
    train_df[TARGET]
)

y_test = label_encoder.transform(
    test_df[TARGET]
)

print()
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# ============================================================
# XGBOOST
# ============================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42
)


# ============================================================
# TRAIN
# ============================================================

print()
print("Training XGBoost...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="macro"
)

recall = recall_score(
    y_test,
    y_pred,
    average="macro"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

print()
print("=" * 60)
print("XGBOOST RESULTS")
print("=" * 60)

print(
    f"Accuracy          : {accuracy:.4f}"
)

print(
    f"Macro Precision   : {precision:.4f}"
)

print(
    f"Macro Recall      : {recall:.4f}"
)

print(
    f"Macro F1          : {f1:.4f}"
)

print()
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

model_file = (
    MODEL_DIR /
    "xgboost_evguard.pkl"
)

with open(
    model_file,
    "wb"
) as f:

    pickle.dump(
        model,
        f
    )


# Save label encoder separately
encoder_file = (
    MODEL_DIR /
    "xgboost_label_encoder.pkl"
)

with open(
    encoder_file,
    "wb"
) as f:

    pickle.dump(
        label_encoder,
        f
    )


print()
print("Model saved:")
print(model_file)

print()
print("Label encoder saved:")
print(encoder_file)

print()
print("DONE!")
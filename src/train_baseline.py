import os
import joblib
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

DATA_PATH = r"D:\amlprj\data\ml_dataset.csv"
MODEL_PATH = r"D:\amlprj\models\logistic_baseline.pkl"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Deployment-realistic features
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

X = df[FEATURES]
y = df[TARGET]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Battery-level split
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y_encoded, groups=df["battery_id"])
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y_encoded[train_idx]
y_test = y_encoded[test_idx]

print("Training batteries:")
print(df.iloc[train_idx]["battery_id"].unique())

print("\nTesting batteries:")
print(df.iloc[test_idx]["battery_id"].unique())

# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression baseline
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)
recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)
f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

print("\n===== LOGISTIC REGRESSION BASELINE =====")
print(f"Accuracy          : {accuracy:.4f}")
print(f"Macro Precision   : {precision:.4f}")
print(f"Macro Recall      : {recall:.4f}")
print(f"Macro F1 Score    : {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model bundle
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

joblib.dump(
    {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "features": FEATURES
    },
    MODEL_PATH
)

print(f"\nModel saved to:")
print(MODEL_PATH)
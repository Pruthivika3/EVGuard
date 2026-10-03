import os
import random
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = r"D:\amlprj\data\ml_dataset.csv"

MODEL_PATH = (
    r"D:\amlprj\models\bilstm_attention_evguard.pt"
)

SCALER_PATH = (
    r"D:\amlprj\models\bilstm_scaler.pkl"
)

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

SEQ_LEN = 30

HIDDEN_SIZE = 64

NUM_LAYERS = 2

BATCH_SIZE = 64

EPOCHS = 15

LEARNING_RATE = 0.001

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# ATTENTION LAYER
# ============================================================

class Attention(nn.Module):

    def __init__(self, hidden_size):

        super().__init__()

        self.attention = nn.Linear(
            hidden_size * 2,
            1
        )

    def forward(self, x):

        # x shape:
        # batch_size x sequence_length x hidden_size*2

        scores = self.attention(x)

        weights = torch.softmax(
            scores,
            dim=1
        )

        context = torch.sum(
            x * weights,
            dim=1
        )

        return context


# ============================================================
# BiLSTM + ATTENTION MODEL
# ============================================================

class BiLSTMAttention(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        self.attention = Attention(
            hidden_size
        )

        self.fc = nn.Sequential(

            nn.Linear(
                hidden_size * 2,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                64,
                num_classes
            )
        )

    def forward(self, x):

        lstm_output, _ = self.lstm(x)

        context = self.attention(
            lstm_output
        )

        output = self.fc(
            context
        )

        return output


# ============================================================
# CREATE SEQUENCES
# ============================================================

def create_sequences(
    df,
    indices,
    scaler
):

    sequences = []

    labels = []

    selected = df.iloc[indices].copy()

    # Process each battery separately
    for battery_id, battery_df in selected.groupby(
        "battery_id"
    ):

        battery_df = battery_df.sort_values(
            ["cycle", "time"]
        )

        # Get feature dataframe
        feature_values = battery_df[
            FEATURES
        ]

        # Apply scaler
        scaled_values = scaler.transform(
            feature_values
        )

        # Labels are already integer encoded
        target_values = battery_df[
            TARGET
        ].values.astype(int)

        # Create non-overlapping sequences
        for start in range(
            0,
            len(scaled_values) - SEQ_LEN + 1,
            SEQ_LEN
        ):

            end = start + SEQ_LEN

            sequence = scaled_values[
                start:end
            ]

            # Label of final timestep
            label = target_values[
                end - 1
            ]

            sequences.append(
                sequence
            )

            labels.append(
                label
            )

    return (
        np.array(
            sequences,
            dtype=np.float32
        ),
        np.array(
            labels,
            dtype=np.int64
        )
    )


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading dataset...")

df = pd.read_csv(
    DATA_PATH
)

print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# ENCODE TARGET
# ============================================================

label_encoder = LabelEncoder()

df[TARGET] = label_encoder.fit_transform(
    df[TARGET]
)

print("\nRisk classes:")

for index, label in enumerate(
    label_encoder.classes_
):

    print(
        f"{index} -> {label}"
    )


# ============================================================
# BATTERY-LEVEL TRAIN / TEST SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=SEED
)

train_idx, test_idx = next(
    splitter.split(
        df[FEATURES],
        df[TARGET],
        groups=df["battery_id"]
    )
)


print("\nTraining batteries:")

print(
    df.iloc[
        train_idx
    ]["battery_id"].unique()
)


print("\nTesting batteries:")

print(
    df.iloc[
        test_idx
    ]["battery_id"].unique()
)


# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()

scaler.fit(
    df.iloc[
        train_idx
    ][FEATURES]
)


# ============================================================
# CREATE TRAINING SEQUENCES
# ============================================================

print(
    "\nCreating training sequences..."
)

X_train, y_train = create_sequences(
    df,
    train_idx,
    scaler
)

print(
    f"Training sequences: {X_train.shape}"
)


# ============================================================
# CREATE TESTING SEQUENCES
# ============================================================

print(
    "\nCreating testing sequences..."
)

X_test, y_test = create_sequences(
    df,
    test_idx,
    scaler
)

print(
    f"Testing sequences: {X_test.shape}"
)


# ============================================================
# CHECK CLASS DISTRIBUTION
# ============================================================

print(
    "\nTraining class distribution:"
)

print(
    np.bincount(y_train)
)


print(
    "\nTesting class distribution:"
)

print(
    np.bincount(y_test)
)


# ============================================================
# PYTORCH DATASETS
# ============================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.long
)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.long
)


train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"\nDevice: {device}"
)


# ============================================================
# CREATE MODEL
# ============================================================

model = BiLSTMAttention(
    input_size=len(FEATURES),
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=len(
        label_encoder.classes_
    )
)

model = model.to(
    device
)


print("\nModel architecture:")

print(model)


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# TRAINING
# ============================================================

print(
    "\n===== TRAINING BiLSTM + ATTENTION ====="
)


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(
            device
        )

        y_batch = y_batch.to(
            device
        )

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            X_batch
        )

        # Calculate loss
        loss = criterion(
            outputs,
            y_batch
        )

        # Backpropagation
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        # Update weights
        optimizer.step()

        total_loss += loss.item()

    average_loss = (
        total_loss /
        len(train_loader)
    )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"- Loss: {average_loss:.4f}"
    )


# ============================================================
# EVALUATION
# ============================================================

print(
    "\nEvaluating model..."
)

model.eval()

all_predictions = []

all_targets = []

all_probabilities = []


with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(
            device
        )

        outputs = model(
            X_batch
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_targets.extend(
            y_batch.numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )


y_pred = np.array(
    all_predictions
)

y_true = np.array(
    all_targets
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)


print(
    "\n===== BiLSTM + ATTENTION RESULTS ====="
)

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
    f"Macro F1 Score    : {f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_true,
        y_pred
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    os.path.dirname(
        MODEL_PATH
    ),
    exist_ok=True
)


torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "input_size":
            len(FEATURES),

        "hidden_size":
            HIDDEN_SIZE,

        "num_layers":
            NUM_LAYERS,

        "num_classes":
            len(
                label_encoder.classes_
            ),

        "features":
            FEATURES,

        "label_classes":
            label_encoder.classes_.tolist(),

        "seq_len":
            SEQ_LEN
    },
    MODEL_PATH
)


# ============================================================
# SAVE SCALER
# ============================================================

joblib.dump(
    scaler,
    SCALER_PATH
)


print(
    "\nModel saved to:"
)

print(
    MODEL_PATH
)


print(
    "\nScaler saved to:"
)

print(
    SCALER_PATH
)


print(
    "\n===== TRAINING COMPLETE ====="
)
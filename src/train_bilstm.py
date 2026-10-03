import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from config import DATA_DIR, MODEL_DIR, WINDOW_SIZE, RANDOM_STATE
from preprocessing import load_data, BASE_FEATURES

class Attention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.score = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)
        return (x * weights).sum(dim=1)

class BiLSTMAttention(nn.Module):
    def __init__(self, input_size, hidden=64, classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden, batch_first=True,
            bidirectional=True, num_layers=2, dropout=0.2
        )
        self.attention = Attention(hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, classes)
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.attention(x)
        return self.classifier(x)

def make_sequences(df, scaler):
    Xs, ys, groups = [], [], []
    for battery_id, g in df.groupby("battery_id"):
        g = g.sort_values("timestamp")
        values = scaler.transform(g[BASE_FEATURES])
        labels = g["risk_label"].values
        for i in range(WINDOW_SIZE, len(g)):
            Xs.append(values[i-WINDOW_SIZE:i])
            ys.append(labels[i])
            groups.append(battery_id)
    return np.array(Xs, dtype=np.float32), np.array(ys), np.array(groups)

def main():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    df = load_data(DATA_DIR / "demo_battery_data.csv")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, groups=df["battery_id"]))

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    scaler = StandardScaler()
    scaler.fit(train_df[BASE_FEATURES])

    X_train, y_train, _ = make_sequences(train_df, scaler)
    X_test, y_test, _ = make_sequences(test_df, scaler)

    model = BiLSTMAttention(len(BASE_FEATURES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    Xt = torch.tensor(X_train)
    yt = torch.tensor(y_train, dtype=torch.long)

    model.train()
    for epoch in range(15):
        optimizer.zero_grad()
        logits = model(Xt)
        loss = criterion(logits, yt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(f"Epoch {epoch+1:02d}/15 - loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_test)).argmax(dim=1).numpy()

    print(classification_report(y_test, pred, zero_division=0))
    print("Macro F1:", f1_score(y_test, pred, average="macro"))

    torch.save({
        "state_dict": model.state_dict(),
        "input_size": len(BASE_FEATURES),
        "window_size": WINDOW_SIZE,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_
    }, MODEL_DIR / "bilstm_attention.pt")

if __name__ == "__main__":
    main()

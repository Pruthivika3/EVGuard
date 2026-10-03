import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR

def generate(n_batteries=20, points_per_battery=500, seed=42):
    rng = np.random.default_rng(seed)
    rows = []

    for b in range(n_batteries):
        temp = 28 + rng.normal(0, 1)
        voltage = 3.75 + rng.normal(0, 0.02)
        current = rng.normal(15, 3)
        soc = rng.uniform(40, 95)
        soh = rng.uniform(0.85, 1.0)
        resistance = rng.uniform(0.015, 0.04)

        for t in range(points_per_battery):
            current += rng.normal(0, 2)
            current = np.clip(current, -20, 100)

            # Occasional stress episodes for demonstration only.
            stress = ((t > points_per_battery * 0.72) and (b % 4 == 0))
            if stress:
                temp += 0.08 + rng.normal(0, 0.03)
                resistance += 0.00015
                current += 1.0
            else:
                temp += rng.normal(0, 0.04)
                temp += (30 - temp) * 0.01

            voltage = 3.75 - 0.0015 * current + rng.normal(0, 0.015)
            soc = np.clip(soc - current * 0.0008 + rng.normal(0, 0.02), 0, 100)
            soh = np.clip(soh - max(temp - 45, 0) * 0.00001, 0.7, 1.0)

            if temp >= 48 or (temp > 43 and current > 55):
                label = 2
            elif temp >= 39 or current > 45 or resistance > 0.055:
                label = 1
            else:
                label = 0

            rows.append({
                "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(seconds=t * 10),
                "battery_id": f"B{b+1:03d}",
                "temperature": temp,
                "voltage": voltage,
                "current": current,
                "soc": soc,
                "soh": soh,
                "internal_resistance": resistance,
                "risk_label": label
            })

    df = pd.DataFrame(rows)
    out = DATA_DIR / "demo_battery_data.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} rows to {out}")

if __name__ == "__main__":
    generate()

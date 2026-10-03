import pandas as pd
import numpy as np

INPUT = r"D:\amlprj\data\nasa_standardized.csv"
OUTPUT = r"D:\amlprj\data\ml_dataset.csv"

df = pd.read_csv(INPUT)

# Replace invalid resistance values
df["internal_resistance"] = df["internal_resistance"].replace(
    [np.inf, -np.inf],
    np.nan
)

# Treat extremely large resistance values as invalid
df.loc[
    df["internal_resistance"] > 100,
    "internal_resistance"
] = np.nan

# Fill missing resistance using median
df["internal_resistance"] = df["internal_resistance"].fillna(
    df["internal_resistance"].median()
)

# Remove rows with any remaining missing values
df = df.dropna()

# Save
df.to_csv(
    OUTPUT,
    index=False
)

print("ML dataset created.")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print()
print("Risk distribution:")
print(df["risk"].value_counts())
print()
print("Resistance statistics:")
print(df["internal_resistance"].describe())
print()
print(f"Saved to: {OUTPUT}")
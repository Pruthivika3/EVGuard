import pandas as pd
from pathlib import Path

DATA_FILE = Path("data/nasa_standardized.csv")
OUTPUT_FILE = Path("data/sample_30_point_sequence.csv")

FEATURES = [
    "cycle",
    "time",
    "voltage",
    "current",
    "temperature",
    "soc",
    "internal_resistance"
]

# Load NASA standardized dataset
df = pd.read_csv(DATA_FILE)

# Select battery B0005
sample = (
    df[df["battery_id"] == "B0005"]
    .sort_values(["cycle", "time"])
)

# Find the first cycle containing 30 points
for cycle in sample["cycle"].unique():

    cycle_data = sample[
        sample["cycle"] == cycle
    ]

    if len(cycle_data) >= 30:

        result = cycle_data[
            FEATURES
        ].head(30)

        result.to_csv(
            OUTPUT_FILE,
            index=False
        )

        print()
        print("SUCCESS!")
        print("Created:")
        print(OUTPUT_FILE)
        print()
        print("Rows:", len(result))
        print()
        print(result)

        break

else:
    print("ERROR: Could not find a cycle with 30 points.")
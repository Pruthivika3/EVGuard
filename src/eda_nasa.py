import pandas as pd
import matplotlib.pyplot as plt

DATA = r"D:\amlprj\data\nasa_standardized.csv"
OUT = r"D:\amlprj\data\eda"

import os
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(DATA)

# 1. Risk distribution
df["risk"].value_counts().plot(kind="bar")
plt.title("NASA Battery Risk Distribution")
plt.xlabel("Risk")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig(f"{OUT}\\risk_distribution.png", dpi=300)
plt.close()

# 2. SOH distribution
df["soh"].plot(kind="hist", bins=40)
plt.title("State of Health Distribution")
plt.xlabel("SOH (%)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{OUT}\\soh_distribution.png", dpi=300)
plt.close()

# 3. SOH degradation by battery
for battery in df["battery_id"].unique():

    temp = df[df["battery_id"] == battery]

    cycle_soh = (
        temp.groupby("cycle")["soh"]
        .mean()
        .reset_index()
    )

    plt.plot(
        cycle_soh["cycle"],
        cycle_soh["soh"],
        label=battery
    )

plt.title("Battery SOH Degradation")
plt.xlabel("Cycle")
plt.ylabel("SOH (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUT}\\soh_degradation.png", dpi=300)
plt.close()

# 4. Temperature vs cycle
for battery in df["battery_id"].unique():

    temp = df[df["battery_id"] == battery]

    cycle_temp = (
        temp.groupby("cycle")["temperature"]
        .mean()
        .reset_index()
    )

    plt.plot(
        cycle_temp["cycle"],
        cycle_temp["temperature"],
        label=battery
    )

plt.title("Battery Temperature Across Cycles")
plt.xlabel("Cycle")
plt.ylabel("Temperature")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUT}\\temperature_cycles.png", dpi=300)
plt.close()

# 5. Voltage vs time sample
battery = df["battery_id"].iloc[0]
sample = df[df["battery_id"] == battery]

plt.plot(
    sample["time"],
    sample["voltage"]
)

plt.title(f"Voltage Profile - {battery}")
plt.xlabel("Time")
plt.ylabel("Voltage")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUT}\\voltage_profile.png", dpi=300)
plt.close()

# 6. Feature correlation
numeric = df.select_dtypes(
    include="number"
)

corr = numeric.corr()

plt.figure(figsize=(10, 8))
plt.imshow(corr, aspect="auto")
plt.colorbar()

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=90
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns
)

plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(
    f"{OUT}\\correlation_matrix.png",
    dpi=300
)

plt.close()

print("EDA completed successfully.")
print(f"Plots saved to: {OUT}")

print()
print("Generated files:")

for file in os.listdir(OUT):
    print(file)
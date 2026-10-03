import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

RAW_DIR = r"D:\amlprj\data\nasa_raw"
OUTPUT_FILE = r"D:\amlprj\data\nasa_standardized.csv"

BATTERIES = ["B0005", "B0006", "B0007", "B0018"]

RATED_CAPACITY = 2.0
N_POINTS = 30


def unwrap(value):
    """Unwrap MATLAB object arrays."""
    while isinstance(value, np.ndarray) and value.dtype == object:
        if value.size != 1:
            break
        value = value.flat[0]

    return value


def extract_cycles(file_path, battery_id):

    mat = loadmat(
        file_path,
        squeeze_me=False,
        struct_as_record=False
    )

    battery = mat[battery_id][0, 0]

    # Get cycle field
    cycles = battery.cycle

    # Unwrap MATLAB object
    cycles = unwrap(cycles)

    print(
        f"  Cycle object type: {type(cycles)}"
    )

    print(
        f"  Cycle shape: "
        f"{getattr(cycles, 'shape', 'N/A')}"
    )

    rows = []

    # Convert cycle collection into iterable
    if isinstance(cycles, np.ndarray):
        cycles = cycles.flatten()
    else:
        cycles = [cycles]

    print(
        f"  Total cycles found: {len(cycles)}"
    )

    for cycle_number, cycle in enumerate(
        cycles,
        start=1
    ):

        try:

            cycle = unwrap(cycle)

            cycle_type = cycle.type

            if isinstance(cycle_type, np.ndarray):
                cycle_type = cycle_type.item()

            cycle_type = str(
                cycle_type
            ).strip()

            if cycle_type.lower() != "discharge":
                continue

            data = unwrap(cycle.data)

            voltage = np.asarray(
                unwrap(
                    data.Voltage_measured
                ),
                dtype=float
            ).flatten()

            current = np.asarray(
                unwrap(
                    data.Current_measured
                ),
                dtype=float
            ).flatten()

            temperature = np.asarray(
                unwrap(
                    data.Temperature_measured
                ),
                dtype=float
            ).flatten()

            time = np.asarray(
                unwrap(
                    data.Time
                ),
                dtype=float
            ).flatten()

        except Exception as e:

            print(
                f"  Cycle {cycle_number} skipped: "
                f"{e}"
            )

            continue

        if len(voltage) < 5:
            continue

        min_len = min(
            len(voltage),
            len(current),
            len(temperature),
            len(time)
        )

        voltage = voltage[:min_len]
        current = current[:min_len]
        temperature = temperature[:min_len]
        time = time[:min_len]

        # Remove NaN / infinity
        valid = (
            np.isfinite(voltage)
            & np.isfinite(current)
            & np.isfinite(temperature)
            & np.isfinite(time)
        )

        voltage = voltage[valid]
        current = current[valid]
        temperature = temperature[valid]
        time = time[valid]

        if len(time) < 5:
            continue

        # Sort by time
        order = np.argsort(time)

        time = time[order]
        voltage = voltage[order]
        current = current[order]
        temperature = temperature[order]

        # Remove duplicate timestamps
        time, unique_indices = np.unique(
            time,
            return_index=True
        )

        voltage = voltage[unique_indices]
        current = current[unique_indices]
        temperature = temperature[unique_indices]

        if len(time) < 5:
            continue

        # Fixed-length representation
        new_time = np.linspace(
            time.min(),
            time.max(),
            N_POINTS
        )

        voltage_resampled = np.interp(
            new_time,
            time,
            voltage
        )

        current_resampled = np.interp(
            new_time,
            time,
            current
        )

        temperature_resampled = np.interp(
            new_time,
            time,
            temperature
        )

        # Discharge capacity in Ah
        capacity = np.trapezoid(
            np.abs(current),
            time / 3600.0
        )

        if (
            not np.isfinite(capacity)
            or capacity <= 0
        ):
            continue

        # SOH
        soh = (
            capacity / RATED_CAPACITY
        ) * 100.0

        # Project-defined risk classes
        if soh >= 90:
            risk = "NORMAL"

        elif soh >= 80:
            risk = "WARNING"

        else:
            risk = "CRITICAL"

        duration = (
            new_time.max()
            - new_time.min()
        )

        for point in range(N_POINTS):

            v = voltage_resampled[point]
            c = current_resampled[point]
            temp = temperature_resampled[point]

            # Resistance proxy
            if abs(c) > 1e-6:
                resistance = abs(v / c)
            else:
                resistance = 0.0

            # Approximate SOC
            if duration > 0:

                soc = 1.0 - (
                    (
                        new_time[point]
                        - new_time.min()
                    )
                    / duration
                )

            else:

                soc = 1.0

            rows.append({

                "battery_id":
                    battery_id,

                "cycle":
                    cycle_number,

                "time":
                    new_time[point],

                "voltage":
                    v,

                "current":
                    c,

                "temperature":
                    temp,

                "soc":
                    soc,

                "capacity":
                    capacity,

                "soh":
                    soh,

                "internal_resistance":
                    resistance,

                "risk":
                    risk
            })

    return rows


def main():

    all_rows = []

    for battery_id in BATTERIES:

        file_path = os.path.join(
            RAW_DIR,
            battery_id + ".mat"
        )

        if not os.path.exists(file_path):

            print(
                f"Missing: {file_path}"
            )

            continue

        print()
        print(
            f"Processing {battery_id}..."
        )

        rows = extract_cycles(
            file_path,
            battery_id
        )

        print(
            f"  Extracted records: "
            f"{len(rows)}"
        )

        all_rows.extend(rows)

    if not all_rows:

        raise RuntimeError(
            "No battery data could be extracted."
        )

    df = pd.DataFrame(
        all_rows
    )

    os.makedirs(
        os.path.dirname(
            OUTPUT_FILE
        ),
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 60)
    print(
        "NASA DATASET CREATED SUCCESSFULLY"
    )
    print("=" * 60)

    print(
        f"Rows     : {len(df)}"
    )

    print(
        f"Columns  : {len(df.columns)}"
    )

    print(
        f"File     : {OUTPUT_FILE}"
    )

    print()
    print(
        "Battery distribution:"
    )

    print(
        df["battery_id"].value_counts()
    )

    print()
    print(
        "Risk distribution:"
    )

    print(
        df["risk"].value_counts()
    )

    print()
    print(
        "First 5 rows:"
    )

    print(
        df.head()
    )


if __name__ == "__main__":
    main()
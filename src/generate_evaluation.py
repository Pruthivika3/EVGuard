import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ============================================================
# PATHS
# ============================================================

OUTPUT_DIR = r"D:\amlprj\data\evaluation"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# MODEL RESULTS
# ============================================================

results = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "XGBoost",
        "BiLSTM + Attention"
    ],

    "Accuracy": [
        0.8798,
        0.8619,
        0.9405
    ],

    "Macro Precision": [
        0.8525,
        0.8240,
        0.9426
    ],

    "Macro Recall": [
        0.8916,
        0.7781,
        0.8803
    ],

    "Macro F1": [
        0.8496,
        0.7844,
        0.9063
    ]
})


# ============================================================
# SAVE COMPARISON TABLE
# ============================================================

results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.csv"
    ),
    index=False
)


print(
    "\nModel comparison saved:"
)

print(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.csv"
    )
)


# ============================================================
# COMPARISON CHART
# ============================================================

metrics = [
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1"
]

x = np.arange(
    len(results["Model"])
)

width = 0.18

plt.figure(
    figsize=(12, 6)
)

for i, metric in enumerate(metrics):

    plt.bar(
        x + (i - 1.5) * width,
        results[metric],
        width,
        label=metric
    )

plt.xticks(
    x,
    results["Model"]
)

plt.ylabel(
    "Score"
)

plt.ylim(
    0,
    1
)

plt.title(
    "EVGuard AI Model Comparison"
)

plt.legend()

plt.tight_layout()

comparison_plot = os.path.join(
    OUTPUT_DIR,
    "model_comparison.png"
)

plt.savefig(
    comparison_plot,
    dpi=300
)

plt.close()


# ============================================================
# CONFUSION MATRICES
# ============================================================

classes = [
    "CRITICAL",
    "NORMAL",
    "WARNING"
]


confusion_matrices = {

    "Logistic Regression": np.array([
        [2712, 0, 378],
        [0, 961, 209],
        [19, 0, 761]
    ]),

    "XGBoost": np.array([
        [3039, 0, 51],
        [3, 754, 413],
        [229, 0, 551]
    ]),

    "BiLSTM + Attention": np.array([
        [103, 0, 0],
        [0, 37, 2],
        [8, 0, 18]
    ])
}


for model_name, matrix in confusion_matrices.items():

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("+", "plus")
    )

    # Raw confusion matrix
    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=classes
    )

    display.plot(
        ax=ax,
        values_format="d"
    )

    plt.title(
        f"{model_name} - Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{safe_name}_confusion_matrix.png"
        ),
        dpi=300
    )

    plt.close()

    # Normalized matrix
    row_totals = matrix.sum(
        axis=1,
        keepdims=True
    )

    normalized = (
        matrix /
        row_totals
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=normalized,
        display_labels=classes
    )

    display.plot(
        ax=ax,
        values_format=".2f"
    )

    plt.title(
        f"{model_name} - Normalized Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{safe_name}_normalized_confusion_matrix.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# ERROR ANALYSIS
# ============================================================

error_analysis = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "XGBoost",
        "BiLSTM + Attention"
    ],

    "Major_Error": [
        "CRITICAL classified as WARNING and NORMAL classified as WARNING",
        "NORMAL classified as WARNING; WARNING classified as CRITICAL",
        "WARNING classified as CRITICAL"
    ],

    "Key_Observation": [
        "Good overall recall, but WARNING precision is limited.",
        "Strong CRITICAL detection but weaker separation between NORMAL and WARNING.",
        "Best overall performance, with remaining errors concentrated in WARNING."
    ]
})


error_path = os.path.join(
    OUTPUT_DIR,
    "error_analysis.csv"
)

error_analysis.to_csv(
    error_path,
    index=False
)


# ============================================================
# REPORT SUMMARY
# ============================================================

summary = """
EVGuard AI - Model Evaluation Summary

Dataset:
NASA Li-ion battery aging dataset.

Evaluation protocol:
Battery-level split.

Training batteries:
B0005, B0007, B0018

Testing battery:
B0006

Input features:
cycle, time, voltage, current, temperature, soc,
internal_resistance

Models:

1. Logistic Regression
Accuracy: 87.98%
Macro Precision: 85.25%
Macro Recall: 89.16%
Macro F1: 84.96%

2. XGBoost
Accuracy: 86.19%
Macro Precision: 82.40%
Macro Recall: 77.81%
Macro F1: 78.44%

3. BiLSTM + Attention
Accuracy: 94.05%
Macro Precision: 94.26%
Macro Recall: 88.03%
Macro F1: 90.63%

Error Analysis:

Logistic Regression:
The main errors involve WARNING predictions, including CRITICAL
and NORMAL samples being classified as WARNING.

XGBoost:
The largest error is NORMAL samples being classified as WARNING.
WARNING samples are also sometimes classified as CRITICAL.

BiLSTM + Attention:
The model achieves the strongest overall performance. Its main
remaining error is WARNING samples being classified as CRITICAL.

Conclusion:
The sequence-based BiLSTM + Attention model provides the strongest
overall performance among the evaluated models on the held-out
battery. This suggests that temporal information in battery
telemetry is useful for risk classification.
"""

summary_path = os.path.join(
    OUTPUT_DIR,
    "evaluation_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(summary)


# ============================================================
# COMPLETE
# ============================================================

print("\n===== EVALUATION COMPLETE =====")

print(
    f"Files generated in:\n{OUTPUT_DIR}"
)

print(
    "\nGenerated:"
)

print("- model_comparison.csv")
print("- model_comparison.png")
print("- confusion matrix plots")
print("- normalized confusion matrix plots")
print("- error_analysis.csv")
print("- evaluation_summary.txt")
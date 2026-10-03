import pandas as pd
from pathlib import Path

def main():
    result_path = Path("results/tabular_model_results.csv")
    if result_path.exists():
        print(pd.read_csv(result_path).to_string(index=False))
    else:
        print("Run train_xgboost.py first.")

if __name__ == "__main__":
    main()

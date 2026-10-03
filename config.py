from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

FEATURES = [
    "temperature",
    "voltage",
    "current",
    "soc",
    "soh",
    "internal_resistance",
]

TARGET = "risk_label"

RANDOM_STATE = 42
WINDOW_SIZE = 30
NUM_CLASSES = 3

for folder in [DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    folder.mkdir(exist_ok=True)

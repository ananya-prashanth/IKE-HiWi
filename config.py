from pathlib import Path

# Directory containing config.py
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"

SIM7200_DIR = DATA_DIR / "7200 sec Run"

LABEL_FILE = DATA_DIR / "Simulation output with conclusion of timed out cases.xlsx"

PROCESSED_DIR= PROJECT_ROOT/ "data" / "processed"

METADATA_FILE= (
    PROJECT_ROOT/ PROCESSED_DIR / "simulation_metadata_7200.csv"
)

CLEAN_METADATA_FILE=(
     PROJECT_ROOT/ PROCESSED_DIR / "simulation_metadata_7200_clean.csv"
)
RANDOM_STATE= 42

SPLIT_FILE=(
    PROCESSED_DIR/ "simulation_splits.csv"
)

MIN_SEQUENCE_LEN= 10

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "trajectory_diagnostics.csv"
)

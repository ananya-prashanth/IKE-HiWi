"""
build_metadata.py

Create a metadata table for all simulations in the
7200 sec Run dataset.

For each simulation we collect:

    - Simulation ID
    - 15 static input parameters
    - Number of unique timesteps
    - Maximum simulation time
"""

from pathlib import Path
import pandas as pd


# ==============================================================
# Project paths
# ==============================================================
from config import PROJECT_ROOT, DATA_DIR, SIM7200_DIR


LABEL_FILE = (
    DATA_DIR /
    "Simulation output with conclusion of timed out cases.xlsx"
)

OUTPUT_FILE = (
    DATA_DIR /
    "processed" /
    "simulation_metadata_7200.csv"
)


# ==============================================================
# Static simulation input parameters
# ==============================================================

INPUT_COLUMNS = [
    "Psys",
    "Ffuel",
    "Porosity",
    "Dparticle",
    "Alpha",
    "Rflat",
    "Tbed",
    "Decay",
    "Fstruct",
    "Rcake",
    "Topcake",
    "Botcake",
    "Porcake",
    "Partdiacake",
    "Hdebris",
]


# ==============================================================
# Check paths
# ==============================================================

print("Simulation directory:")
print(SIM7200_DIR)

print("\nExists:")
print(SIM7200_DIR.exists())

print("\nLabel file:")
print(LABEL_FILE)

print("\nExists:")
print(LABEL_FILE.exists())


if not SIM7200_DIR.exists():
    raise FileNotFoundError(
        f"Simulation directory not found:\n{SIM7200_DIR}"
    )

if not LABEL_FILE.exists():
    raise FileNotFoundError(
        f"Label file not found:\n{LABEL_FILE}"
    )


# ==============================================================
# Load input parameter table
# ==============================================================

print("\nLoading input parameter file...")

labels = pd.read_excel(LABEL_FILE)

labels = labels.set_index("Simulation No")

print(
    f"Number of simulations in label file: {len(labels)}"
)


# Check input columns
missing_columns = [
    column
    for column in INPUT_COLUMNS
    if column not in labels.columns
]

if missing_columns:
    raise ValueError(
        f"Missing input columns:\n{missing_columns}"
    )


# ==============================================================
# Find temperature files
# ==============================================================

temp_files = sorted(
    SIM7200_DIR.glob(
        "*.*.temp_unwrapped.csv"
    )
)

print(
    f"\nTemperature files found: {len(temp_files)}"
)


# ==============================================================
# Extract metadata
# ==============================================================

records = []

for i, temp_file in enumerate(temp_files, start=1):

    # ----------------------------------------------------------
    # Extract simulation ID from filename
    #
    # Example:
    #
    # 00042.00042.temp_unwrapped.csv
    #
    # simulation ID = 42
    # ----------------------------------------------------------

    try:
        simid = int(temp_file.name[:5])

    except ValueError:
        print(
            f"Skipping unexpected file: {temp_file.name}"
        )
        continue


    # ----------------------------------------------------------
    # Make sure the simulation exists in the input table
    # ----------------------------------------------------------

    if simid not in labels.index:

        print(
            f"WARNING: simulation {simid} "
            f"not found in label file"
        )

        continue


    # ----------------------------------------------------------
    # Read only the time column
    #
    # We do not need particle temperatures yet.
    # ----------------------------------------------------------

    time = pd.read_csv(
        temp_file,
        usecols=["time"]
    )["time"]


    if time.empty:

        print(
            f"WARNING: simulation {simid} "
            f"has an empty temperature file"
        )

        continue


    # ----------------------------------------------------------
    # Determine actual trajectory information
    # ----------------------------------------------------------

    sequence_length = time.nunique()

    maximum_time = time.max()


    # ----------------------------------------------------------
    # Get the 15 static input parameters
    # ----------------------------------------------------------

    input_values = labels.loc[
        simid,
        INPUT_COLUMNS
    ].to_dict()


    # ----------------------------------------------------------
    # Create one metadata record
    # ----------------------------------------------------------

    record = {
        "Simulation No": simid,
        **input_values,
        "sequence_length": sequence_length,
        "maximum_time": maximum_time,
    }

    records.append(record)


    # Progress information
    if i % 100 == 0:

        print(
            f"Processed {i}/{len(temp_files)} files"
        )


# ==============================================================
# Create metadata DataFrame
# ==============================================================

metadata = pd.DataFrame(records)


metadata = metadata.sort_values(
    "Simulation No"
).reset_index(drop=True)


# ==============================================================
# Basic checks
# ==============================================================

print("\n========================================")
print("METADATA SUMMARY")
print("========================================")

print(
    f"Number of simulations: {len(metadata)}"
)

print(
    f"Number of columns: {len(metadata.columns)}"
)

print("\nColumns:")

print(
    metadata.columns.tolist()
)


# ==============================================================
# Missing values
# ==============================================================

print("\nMissing values:")

print(
    metadata.isna().sum()
)


# ==============================================================
# Duplicate simulation IDs
# ==============================================================

duplicates = metadata[
    metadata["Simulation No"].duplicated()
]

print(
    f"\nDuplicate simulation IDs: "
    f"{len(duplicates)}"
)


# ==============================================================
# Sequence statistics
# ==============================================================

print("\nSequence length statistics:")

print(
    metadata["sequence_length"].describe()
)


print("\nMaximum simulation time statistics:")

print(
    metadata["maximum_time"].describe()
)


# ==============================================================
# Save
# ==============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

metadata.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved metadata to:")

print(OUTPUT_FILE)
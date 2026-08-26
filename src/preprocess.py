'''
simulation_metadata_7200.csv
       ↓
remove invalid/very short trajectories
       ↓
create log1p(sequence_length)
       ↓
create duration groups
       ↓
train/validation/test split
       ↓
save the split information

Important:
The split is performed BEFORE any time-series data is flattened.
This prevents timesteps from the same simulation appearing in
different dataset splits.

'''

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#-------------------------
#Config
#-------------------------

# Project root
from config import PROJECT_ROOT, PROCESSED_DIR, METADATA_FILE, RANDOM_STATE, MIN_SEQUENCE_LEN

#min number of timesteps required for a useful trajectory


#random seed for reproducibility


# ------------------------------------------------------------------
# Load metadata
# ------------------------------------------------------------------

print("Loading metadata...")

metadata = pd.read_csv(METADATA_FILE)

print(f"Loaded {len(metadata)} simulations.")


# ------------------------------------------------------------------
# Basic validation
# ------------------------------------------------------------------

required_columns = [
    "Simulation No",
    "sequence_length",
    "maximum_time",
]

missing_columns = [
    column
    for column in required_columns
    if column not in metadata.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ------------------------------------------------------------------
# Remove very short trajectories
# ------------------------------------------------------------------

print("\nFiltering short trajectories...")

before = len(metadata)

metadata_clean = metadata[
    metadata["sequence_length"] >= MIN_SEQUENCE_LEN
].copy()

removed = before - len(metadata_clean)

print(f"Minimum sequence length: {MIN_SEQUENCE_LEN}")
print(f"Removed simulations: {removed}")
print(f"Remaining simulations: {len(metadata_clean)}")


# ------------------------------------------------------------------
# Create transformed sequence-length feature
# ------------------------------------------------------------------

metadata_clean["log_sequence_length"] = np.log1p(
    metadata_clean["sequence_length"]
)


# ------------------------------------------------------------------
# Create trajectory-length groups
# ------------------------------------------------------------------

"""
The document recommends grouping simulations according to trajectory length before splitting the dataset.

Using five quantile groups means that approximately the same number of simulations should fall into each group.

qcut(..., q=5) sorts the simulations by log_sequence_length and divides them into five quantile groups.

"""

metadata_clean["length_group"] = pd.qcut(
    metadata_clean["log_sequence_length"],
    q=5,
    labels=False,
    duplicates="drop",
)

print("\nTrajectory length groups:")

print(
    metadata_clean["length_group"]
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------------
# Split into train and temporary sets
# ------------------------------------------------------------------

"""
First split:
80% training
20% temporary

The split is stratified using the trajectory-length group so that
short and long simulations remain represented in both sets.
"""

train_df, temp_df = train_test_split(
    metadata_clean,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=metadata_clean["length_group"],
)


# ------------------------------------------------------------------
# Split temporary set into validation and test
# ------------------------------------------------------------------

"""
The temporary 20% is divided equally:

10% validation
10% test
"""

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df["length_group"],
)


# ------------------------------------------------------------------
# Add split labels
# ------------------------------------------------------------------

train_df = train_df.copy()
validation_df = validation_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
validation_df["split"] = "validation"
test_df["split"] = "test"


# ------------------------------------------------------------------
# Combine datasets
# ------------------------------------------------------------------

metadata_split = pd.concat(
    [
        train_df,
        validation_df,
        test_df,
    ],
    ignore_index=True,
)


# ------------------------------------------------------------------
# Print split statistics
# ------------------------------------------------------------------

print("\nDataset split:")

print(
    metadata_split["split"]
    .value_counts()
)


print("\nSequence-length statistics by split:")

print(
    metadata_split
    .groupby("split")["sequence_length"]
    .describe()
)


print("\nLength-group distribution by split:")

print(
    pd.crosstab(
        metadata_split["length_group"],
        metadata_split["split"],
    )
)


# ------------------------------------------------------------------
# Verify that every simulation appears exactly once
# ------------------------------------------------------------------

duplicate_ids = (
    metadata_split["Simulation No"]
    .duplicated()
    .sum()
)

if duplicate_ids != 0:
    raise ValueError(
        f"Found {duplicate_ids} duplicated simulation IDs."
    )

print(
    "\nSimulation IDs are unique across the dataset."
)


# ------------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------------

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

clean_file = (
    PROCESSED_DIR
    / "simulation_metadata_7200_clean.csv"
)

split_file = (
    PROCESSED_DIR
    / "simulation_splits.csv"
)

metadata_clean.to_csv(
    clean_file,
    index=False,
)

metadata_split.to_csv(
    split_file,
    index=False,
)

# Create a simple table containing only the simulation ID
# and its assigned dataset split.

split_ids = metadata_split[
    ["Simulation No", "split"]
].copy()

split_ids.to_csv(
    split_file,
    index=False,
)


# ------------------------------------------------------------------
# Finished
# ------------------------------------------------------------------

print("\nSaved:")
print(clean_file)
print(split_file)

print("\nPreprocessing complete.")



"""
trajectory_preprocess.py

Create a sampled particle-level trajectory dataset for
temperature prediction.

Each sample represents:

    one simulation
    one particle
    one recorded physical time

Features:
    15 static simulation parameters
    x, y, z particle coordinates
    actual physical time
    normalized and transformed time features

Target:
    temperature

Important:
    Train/validation/test splitting is performed at the
    simulation level before trajectory samples are created.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config import SPLIT_FILE, OUTPUT_DIR,RANDOM_STATE, CLEAN_METADATA_FILE, SIM7200_DIR


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------


RANDOM_STATE = 42

# Number of particle-time observations sampled
# from each simulation.
SAMPLES_PER_SIMULATION = 500


# ------------------------------------------------------------------
# Static simulation parameters
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Load metadata
# ------------------------------------------------------------------

print("Loading metadata...")

metadata = pd.read_csv(
    CLEAN_METADATA_FILE
)

splits = pd.read_csv(
    SPLIT_FILE
)

print(
    f"Metadata simulations: {len(metadata)}"
)

print(
    f"Split assignments: {len(splits)}"
)


# ------------------------------------------------------------------
# Merge metadata with split information
# ------------------------------------------------------------------

metadata = metadata.merge(
    splits[
        [
            "Simulation No",
            "split",
        ]
    ],
    on="Simulation No",
    how="inner",
)

print(
    f"Final simulations: {len(metadata)}"
)


# ------------------------------------------------------------------
# Validate
# ------------------------------------------------------------------

required_columns = (
    [
        "Simulation No",
        "maximum_time",
        "split",
    ]
    + INPUT_COLUMNS
)

missing = [
    column
    for column in required_columns
    if column not in metadata.columns
]

if missing:
    raise ValueError(
        f"Missing metadata columns: {missing}"
    )


# ------------------------------------------------------------------
# Prepare simulation lists
# ------------------------------------------------------------------

simulation_groups = {
    split_name: group.copy()
    for split_name, group
    in metadata.groupby("split")
}

for split_name, group in simulation_groups.items():

    print(
        f"{split_name}: "
        f"{len(group)} simulations"
    )


# ------------------------------------------------------------------
# Process one simulation
# ------------------------------------------------------------------

def process_simulation(
    metadata_row,
    n_samples,
    rng,
):
    """
    Load one simulation and create sampled
    particle-time observations.
    """

    simulation_no = int(
        metadata_row["Simulation No"]
    )

    # --------------------------------------------------------------
    # Temperature file
    # --------------------------------------------------------------

    filename = (
        f"{simulation_no:05d}."
        f"{simulation_no:05d}."
        "temp_unwrapped.csv"
    )

    filepath = SIM7200_DIR / filename

    if not filepath.exists():

        raise FileNotFoundError(
            f"Temperature file not found:\n{filepath}"
        )

    # --------------------------------------------------------------
    # Load trajectory
    # --------------------------------------------------------------

    temp = pd.read_csv(filepath)

    # --------------------------------------------------------------
    # Maximum physical time
    # --------------------------------------------------------------

    maximum_time = float(
        metadata_row["maximum_time"]
    )

    # --------------------------------------------------------------
    # Normalized time
    # --------------------------------------------------------------

    temp["t_norm"] = (
        temp["time"]
        / maximum_time
    )

    # --------------------------------------------------------------
    # Time transformations
    # --------------------------------------------------------------

    temp["log1p_t_norm"] = np.log1p(
        temp["t_norm"]
    )

    temp["sqrt_t_norm"] = np.sqrt(
        temp["t_norm"]
    )

    temp["t_norm_squared"] = (
        temp["t_norm"] ** 2
    )

    temp["log1p_maximum_time"] = np.log1p(
        maximum_time
    )

    # --------------------------------------------------------------
    # Add static simulation parameters
    # --------------------------------------------------------------

    for column in INPUT_COLUMNS:

        temp[column] = metadata_row[
            column
        ]

    # --------------------------------------------------------------
    # Simulation identifier
    # --------------------------------------------------------------

    temp["Simulation No"] = simulation_no

    # --------------------------------------------------------------
    # Stratified temporal sampling
    # --------------------------------------------------------------

    # Divide normalized time into 10 bins.
    temp["time_bin"] = pd.cut(
        temp["t_norm"],
        bins=np.linspace(
            0,
            1,
            11
        ),
        include_lowest=True,
        labels=False,
    )

    sampled_groups = []

    n_bins = 10

    base_samples = (
        n_samples // n_bins
    )

    remainder = (
        n_samples % n_bins
    )

    for bin_id in range(n_bins):

        group = temp[
            temp["time_bin"] == bin_id
        ]

        if len(group) == 0:
            continue

        n_bin_samples = base_samples

        if bin_id < remainder:
            n_bin_samples += 1

        n_bin_samples = min(
            n_bin_samples,
            len(group)
        )

        sampled = group.sample(
            n=n_bin_samples,
            random_state=int(
                rng.integers(
                    0,
                    2**32 - 1
                )
            ),
        )

        sampled_groups.append(
            sampled
        )

    # --------------------------------------------------------------
    # Combine sampled observations
    # --------------------------------------------------------------

    if not sampled_groups:

        return pd.DataFrame()

    sampled = pd.concat(
        sampled_groups,
        ignore_index=True,
    )

    # --------------------------------------------------------------
    # Remove helper column
    # --------------------------------------------------------------

    sampled = sampled.drop(
        columns=["time_bin"]
    )

    return sampled


# ------------------------------------------------------------------
# Process each split
# ------------------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

rng = np.random.default_rng(
    RANDOM_STATE
)


for split_name in [
    "train",
    "validation",
    "test",
]:

    print("\n")
    print("=" * 60)
    print(
        f"Processing {split_name}"
    )
    print("=" * 60)

    group = simulation_groups[
        split_name
    ]

    sampled_simulations = []

    for i, (_, row) in enumerate(
        group.iterrows()
    ):

        sampled = process_simulation(
            metadata_row=row,
            n_samples=SAMPLES_PER_SIMULATION,
            rng=rng,
        )

        if len(sampled) > 0:

            sampled_simulations.append(
                sampled
            )

        if (i + 1) % 100 == 0:

            print(
                f"Processed "
                f"{i + 1}/{len(group)}"
            )

    # --------------------------------------------------------------
    # Combine simulations
    # --------------------------------------------------------------

    trajectory = pd.concat(
        sampled_simulations,
        ignore_index=True,
    )

    # --------------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------------

    feature_columns = (
        INPUT_COLUMNS
        + [
            "x",
            "y",
            "z",
            "time",
            "t_norm",
            "log1p_t_norm",
            "sqrt_t_norm",
            "t_norm_squared",
            "log1p_maximum_time",
        ]
    )

    output_columns = (
        [
            "Simulation No",
            "particle_id",
        ]
        + feature_columns
        + [
            "temperature",
        ]
    )

    trajectory = trajectory[
        output_columns
    ]

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        /trajectory
        / f"trajectory_{split_name}.parquet"
    )

    trajectory.to_parquet(
        output_file,
        index=False,
    )

    print(
        f"\nSaved {split_name}:"
    )

    print(
        output_file
    )

    print(
        f"Rows: {len(trajectory):,}"
    )


print("\n")
print("=" * 60)
print("Trajectory preprocessing complete.")
print("=" * 60)
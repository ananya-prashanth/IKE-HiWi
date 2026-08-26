"""
trajectory_diagnostics.py

Inspect the size and structure of the temperature trajectories
across the 7200 s simulation dataset.

For each simulation, record:
    - number of rows
    - number of particles
    - number of time points
    - maximum simulation time
    - whether every particle has the same number of time points
"""

from pathlib import Path

import pandas as pd

from config import SIM7200_DIR, OUTPUT_DIR, OUTPUT_FILE


# ------------------------------------------------------------------
# Find temperature files
# ------------------------------------------------------------------

temperature_files = sorted(
    SIM7200_DIR.glob("*.temp_unwrapped.csv")
)

print(
    f"Found {len(temperature_files)} temperature files."
)


if len(temperature_files) == 0:
    raise FileNotFoundError(
        f"No temperature files found in {SIM7200_DIR}"
    )


# ------------------------------------------------------------------
# Inspect simulations
# ------------------------------------------------------------------

records = []


for i, file in enumerate(temperature_files):

    # --------------------------------------------------------------
    # Extract simulation number
    # --------------------------------------------------------------

    simulation_no = int(
        file.name.split(".")[0]
    )

    # --------------------------------------------------------------
    # Load temperature data
    # --------------------------------------------------------------

    temp = pd.read_csv(file)

    # --------------------------------------------------------------
    # Basic quantities
    # --------------------------------------------------------------

    n_rows = len(temp)

    n_particles = (
        temp["particle_id"]
        .nunique()
    )

    n_time_points = (
        temp["time"]
        .nunique()
    )

    maximum_time = (
        temp["time"]
        .max()
    )

    minimum_time = (
        temp["time"]
        .min()
    )

    # --------------------------------------------------------------
    # Check temporal completeness
    # --------------------------------------------------------------

    particle_time_counts = (
        temp
        .groupby("particle_id")["time"]
        .nunique()
    )

    min_particle_times = (
        particle_time_counts.min()
    )

    max_particle_times = (
        particle_time_counts.max()
    )

    complete_particle_grid = (
        min_particle_times
        == max_particle_times
        == n_time_points
    )

    # --------------------------------------------------------------
    # Check expected row count
    # --------------------------------------------------------------

    expected_rows = (
        n_particles
        * n_time_points
    )

    rectangular_grid = (
        n_rows == expected_rows
    )

    # --------------------------------------------------------------
    # Store result
    # --------------------------------------------------------------

    records.append(
        {
            "Simulation No": simulation_no,
            "n_rows": n_rows,
            "n_particles": n_particles,
            "n_time_points": n_time_points,
            "minimum_time": minimum_time,
            "maximum_time": maximum_time,
            "min_particle_time_count": (
                min_particle_times
            ),
            "max_particle_time_count": (
                max_particle_times
            ),
            "complete_particle_grid": (
                complete_particle_grid
            ),
            "rectangular_grid": (
                rectangular_grid
            ),
        }
    )

    # --------------------------------------------------------------
    # Progress
    # --------------------------------------------------------------

    if (i + 1) % 100 == 0:

        print(
            f"Processed "
            f"{i + 1}/{len(temperature_files)}"
        )


# ------------------------------------------------------------------
# Create DataFrame
# ------------------------------------------------------------------

diagnostics = pd.DataFrame(records)

diagnostics = diagnostics.sort_values(
    "Simulation No"
).reset_index(drop=True)


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

print("\n")
print("=" * 60)
print("TRAJECTORY DIAGNOSTICS")
print("=" * 60)

print(
    f"\nSimulations inspected: "
    f"{len(diagnostics)}"
)

print("\nParticles per simulation:")
print(
    diagnostics["n_particles"].describe()
)

print("\nTime points per simulation:")
print(
    diagnostics["n_time_points"].describe()
)

print("\nRows per simulation:")
print(
    diagnostics["n_rows"].describe()
)

print("\nMaximum simulation time:")
print(
    diagnostics["maximum_time"].describe()
)

print("\nComplete particle grids:")
print(
    diagnostics["complete_particle_grid"]
    .value_counts()
)

print("\nRectangular grids:")
print(
    diagnostics["rectangular_grid"]
    .value_counts()
)


# ------------------------------------------------------------------
# Check unusual simulations
# ------------------------------------------------------------------

print("\nSimulations with incomplete particle grids:")

incomplete = diagnostics[
    ~diagnostics["complete_particle_grid"]
]

if len(incomplete) == 0:

    print("None")

else:

    print(
        incomplete[
            [
                "Simulation No",
                "n_particles",
                "n_time_points",
                "min_particle_time_count",
                "max_particle_time_count",
            ]
        ].head(20)
    )


# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

diagnostics.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)
"""
plot_trajectory_results.py

Visual inspection of Task 3 trajectory predictions.

Creates:
1. Test R2 comparison for all predicted variables.
2. Actual vs predicted trajectories for representative variables.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from config import PROJECT_ROOT, MODEL_DIR, PREDICTION_DIR

OUTPUT_DIR = (
    MODEL_DIR / "plots"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Load metrics
# ============================================================

metrics = pd.read_csv(
    MODEL_DIR / "trajectory_all_metrics.csv"
)

print(metrics)


# ============================================================
# Plot 1: R2 for all variables
# ============================================================

plot_df = metrics.sort_values(
    "test_R2",
    ascending=True
)

plt.figure(figsize=(10, 8))

plt.barh(
    plot_df["target"],
    plot_df["test_R2"]
)

plt.xlabel("Test R²")
plt.ylabel("Balance variable")
plt.title(
    "Task 3 baseline — Test R² by physical variable"
)

plt.xlim(0, 1.05)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "test_R2_all_variables.png",
    dpi=200
)

plt.show()


# ============================================================
# Variables to inspect
# ============================================================

selected_targets = [
    "mg","ml","mp",
        "eg","el","ep",
        "fgin","fgout",
        "flin","flout",
        "qgin","qgout",
        "qlin","qlout",
        "qpin","qpout",
        "qsi","qgi","qli",
        "qsg","qsl","pow"
]


# ============================================================
# Plot representative trajectories
# ============================================================

for target in selected_targets:

    prediction_file = (
        PREDICTION_DIR
        / f"trajectory_{target}_predictions_test.parquet"
    )

    if not prediction_file.exists():

        print(
            f"Skipping {target}: file not found"
        )

        continue

    df = pd.read_parquet(
        prediction_file
    )

    # --------------------------------------------------------
    # Pick a representative simulation
    # --------------------------------------------------------

    counts = (
        df.groupby("Simulation No")
        .size()
        .sort_values(
            ascending=False
        )
    )

    sim_id = counts.index[0]

    sim = (
        df[
            df["Simulation No"] == sim_id
        ]
        .sort_values("time")
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(
        sim["time"],
        sim[target],
        label="Actual"
    )

    plt.plot(
        sim["time"],
        sim[f"predicted_{target}"],
        "--",
        label="Predicted"
    )

    plt.xlabel("Time [s]")
    plt.ylabel(target)

    plt.title(
        f"{target}: actual vs predicted "
        f"(simulation {sim_id})"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    output_file = (
        OUTPUT_DIR
        / f"{target}_trajectory.png"
    )

    plt.savefig(
        output_file,
        dpi=200
    )

    plt.show()

    print(
        f"{target}: simulation {sim_id}"
    )


print()
print("Plots saved to:")
print(OUTPUT_DIR)
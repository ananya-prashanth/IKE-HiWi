"""
qsl_state_analysis.py

Analyze relationships between active qsl and the physical state
variables in the balance trajectory.

This is a diagnostic only.
It does NOT train a new model.

Goal:
    Determine whether qsl magnitude is strongly related to the
    evolving physical state, rather than only the static simulation
    parameters and time.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr, pearsonr


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path("/home/ikeaprash/project")

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "trajectory"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "models"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Configuration
# ============================================================

QSL_THRESHOLD = 10_000.0

TARGET = "qsl"

STATE_VARIABLES = [
    "mg",
    "ml",
    "mp",
    "eg",
    "el",
    "ep",
    "fgin",
    "fgout",
    "flin",
    "flout",
    "qgin",
    "qgout",
    "qlin",
    "qlout",
    "qpin",
    "qpout",
    "qsi",
    "qgi",
    "qli",
    "qsg",
    "pow",
]


# ============================================================
# Load data
# ============================================================

print("=" * 70)
print("QSL PHYSICAL-STATE ANALYSIS")
print("=" * 70)

print("\nLoading trajectory data...")

train = pd.read_parquet(
    DATA_DIR / "trajectory_train.parquet"
)

validation = pd.read_parquet(
    DATA_DIR / "trajectory_validation.parquet"
)

test = pd.read_parquet(
    DATA_DIR / "trajectory_test.parquet"
)

print(
    f"Train:       {len(train):,} rows"
)

print(
    f"Validation:  {len(validation):,} rows"
)

print(
    f"Test:        {len(test):,} rows"
)


# ============================================================
# Use training data for discovery
# ============================================================

print("\nSelecting meaningful qsl points...")

active_train = train[
    train[TARGET].abs() > QSL_THRESHOLD
].copy()

print(
    f"Active training rows: {len(active_train):,}"
)

print(
    f"Active fraction: "
    f"{len(active_train) / len(train):.6f}"
)


# ============================================================
# Correlation analysis
# ============================================================

print("\nCalculating correlations...")

results = []

qsl = active_train[TARGET]

for variable in STATE_VARIABLES:

    x = active_train[variable]

    # Remove NaN / infinite values
    mask = (
        np.isfinite(x.to_numpy())
        &
        np.isfinite(qsl.to_numpy())
    )

    x_valid = x.to_numpy()[mask]
    qsl_valid = qsl.to_numpy()[mask]

    if len(x_valid) < 10:
        continue

    # Pearson
    pearson_r, pearson_p = pearsonr(
        x_valid,
        qsl_valid,
    )

    # Spearman
    spearman_r, spearman_p = spearmanr(
        x_valid,
        qsl_valid,
    )

    results.append(
        {
            "variable": variable,

            "pearson_r": pearson_r,
            "pearson_abs_r": abs(pearson_r),
            "pearson_p": pearson_p,

            "spearman_r": spearman_r,
            "spearman_abs_r": abs(spearman_r),
            "spearman_p": spearman_p,
        }
    )


correlations = pd.DataFrame(
    results
)

correlations = correlations.sort_values(
    "spearman_abs_r",
    ascending=False,
)


# ============================================================
# Print correlation results
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION WITH ACTIVE qsl")
print("=" * 70)

print(
    correlations[
        [
            "variable",
            "pearson_r",
            "spearman_r",
            "spearman_abs_r",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# qsl magnitude correlation
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION WITH |qsl|")
print("=" * 70)

active_train["qsl_abs"] = (
    active_train[TARGET].abs()
)

qsl_abs = active_train["qsl_abs"]

magnitude_results = []

for variable in STATE_VARIABLES:

    x = active_train[variable]

    mask = (
        np.isfinite(x.to_numpy())
        &
        np.isfinite(qsl_abs.to_numpy())
    )

    x_valid = x.to_numpy()[mask]
    y_valid = qsl_abs.to_numpy()[mask]

    if len(x_valid) < 10:
        continue

    pearson_r, _ = pearsonr(
        x_valid,
        y_valid,
    )

    spearman_r, _ = spearmanr(
        x_valid,
        y_valid,
    )

    magnitude_results.append(
        {
            "variable": variable,
            "pearson_r": pearson_r,
            "pearson_abs_r": abs(pearson_r),
            "spearman_r": spearman_r,
            "spearman_abs_r": abs(spearman_r),
        }
    )


magnitude_correlations = pd.DataFrame(
    magnitude_results
)

magnitude_correlations = (
    magnitude_correlations
    .sort_values(
        "spearman_abs_r",
        ascending=False,
    )
)

print(
    magnitude_correlations.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# Correlation with log magnitude
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION WITH log1p(|qsl|)")
print("=" * 70)

log_qsl = np.log1p(
    active_train[TARGET].abs()
)

log_results = []

for variable in STATE_VARIABLES:

    x = active_train[variable]

    mask = (
        np.isfinite(x.to_numpy())
        &
        np.isfinite(log_qsl.to_numpy())
    )

    x_valid = x.to_numpy()[mask]
    y_valid = log_qsl.to_numpy()[mask]

    if len(x_valid) < 10:
        continue

    pearson_r, _ = pearsonr(
        x_valid,
        y_valid,
    )

    spearman_r, _ = spearmanr(
        x_valid,
        y_valid,
    )

    log_results.append(
        {
            "variable": variable,
            "pearson_r": pearson_r,
            "pearson_abs_r": abs(pearson_r),
            "spearman_r": spearman_r,
            "spearman_abs_r": abs(spearman_r),
        }
    )


log_correlations = pd.DataFrame(
    log_results
)

log_correlations = (
    log_correlations
    .sort_values(
        "spearman_abs_r",
        ascending=False,
    )
)

print(
    log_correlations.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# Compare active vs inactive state distributions
# ============================================================

print("\n" + "=" * 70)
print("ACTIVE vs INACTIVE STATE COMPARISON")
print("=" * 70)

# Sample inactive points to keep this analysis manageable
inactive_train = train[
    train[TARGET].abs() <= QSL_THRESHOLD
]

if len(inactive_train) > len(active_train):

    inactive_sample = inactive_train.sample(
        n=len(active_train),
        random_state=42,
    )

else:

    inactive_sample = inactive_train


comparison = []

for variable in STATE_VARIABLES:

    active_values = (
        active_train[variable]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    inactive_values = (
        inactive_sample[variable]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    if len(active_values) == 0:
        continue

    if len(inactive_values) == 0:
        continue

    comparison.append(
        {
            "variable": variable,

            "active_median":
                active_values.median(),

            "inactive_median":
                inactive_values.median(),

            "active_mean":
                active_values.mean(),

            "inactive_mean":
                inactive_values.mean(),

            "active_std":
                active_values.std(),

            "inactive_std":
                inactive_values.std(),
        }
    )


comparison = pd.DataFrame(
    comparison
)

print(
    comparison.to_string(
        index=False,
    )
)


# ============================================================
# Save results
# ============================================================

correlation_file = (
    OUTPUT_DIR
    / "qsl_state_correlations.csv"
)

magnitude_file = (
    OUTPUT_DIR
    / "qsl_magnitude_correlations.csv"
)

log_file = (
    OUTPUT_DIR
    / "qsl_log_magnitude_correlations.csv"
)

comparison_file = (
    OUTPUT_DIR
    / "qsl_active_inactive_state_comparison.csv"
)


correlations.to_csv(
    correlation_file,
    index=False,
)

magnitude_correlations.to_csv(
    magnitude_file,
    index=False,
)

log_correlations.to_csv(
    log_file,
    index=False,
)

comparison.to_csv(
    comparison_file,
    index=False,
)


print("\nSaved:")

print(correlation_file)
print(magnitude_file)
print(log_file)
print(comparison_file)


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
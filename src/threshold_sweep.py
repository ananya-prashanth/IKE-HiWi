from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


PROJECT_ROOT = Path("/home/ikeaprash/project")

PREDICTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "predictions"
    / "qsl_specialized_predictions_test.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "predictions"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Load predictions
# ------------------------------------------------------------

print("Loading qsl predictions...")

df = pd.read_parquet(
    PREDICTION_FILE
)

print(f"Rows: {len(df):,}")
print()
print(df.columns.tolist())


# ------------------------------------------------------------
# Required columns
# ------------------------------------------------------------

required = [
    "qsl",
    "qsl_active_probability",
    "predicted_qsl",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )


y_true = df["qsl"].to_numpy()

probability = (
    df["qsl_active_probability"]
    .to_numpy()
)

regression_prediction = (
    df["predicted_qsl"]
    .to_numpy()
)

actual_active = (
    np.abs(y_true) > 10_000
).astype(np.int8)


# ------------------------------------------------------------
# Thresholds
# ------------------------------------------------------------

thresholds = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]


# ------------------------------------------------------------
# Evaluate thresholds
# ------------------------------------------------------------

results = []

for threshold in thresholds:

    predicted_active = (
        probability >= threshold
    ).astype(np.int8)

    # Reconstruct qsl trajectory
    predicted_qsl = np.where(
        predicted_active == 1,
        regression_prediction,
        0.0,
    )

    # Classification metrics
    precision = precision_score(
        actual_active,
        predicted_active,
        zero_division=0,
    )

    recall = recall_score(
        actual_active,
        predicted_active,
        zero_division=0,
    )

    f1 = f1_score(
        actual_active,
        predicted_active,
        zero_division=0,
    )

    # Full trajectory metrics
    mae = mean_absolute_error(
        y_true,
        predicted_qsl,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predicted_qsl,
        )
    )

    r2 = r2_score(
        y_true,
        predicted_qsl,
    )

    # Active-point metrics
    active_mask = (
        actual_active == 1
    )

    active_mae = mean_absolute_error(
        y_true[active_mask],
        predicted_qsl[active_mask],
    )

    active_rmse = np.sqrt(
        mean_squared_error(
            y_true[active_mask],
            predicted_qsl[active_mask],
        )
    )

    active_r2 = r2_score(
        y_true[active_mask],
        predicted_qsl[active_mask],
    )

    results.append(
        {
            "threshold": threshold,

            "precision": precision,
            "recall": recall,
            "f1": f1,

            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,

            "active_MAE": active_mae,
            "active_RMSE": active_rmse,
            "active_R2": active_r2,

            "predicted_active_fraction":
                predicted_active.mean(),
        }
    )


results = pd.DataFrame(
    results
)


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print("\n" + "=" * 100)
print("QSL THRESHOLD SWEEP")
print("=" * 100)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ------------------------------------------------------------
# Best thresholds
# ------------------------------------------------------------

best_r2 = results.loc[
    results["R2"].idxmax()
]

best_mae = results.loc[
    results["MAE"].idxmin()
]

best_rmse = results.loc[
    results["RMSE"].idxmin()
]

best_f1 = results.loc[
    results["f1"].idxmax()
]


print("\n" + "=" * 100)
print("BEST THRESHOLDS")
print("=" * 100)

print(
    "\nBest full-trajectory R2:"
)

print(best_r2.to_string())

print(
    "\nBest full-trajectory MAE:"
)

print(best_mae.to_string())

print(
    "\nBest full-trajectory RMSE:"
)

print(best_rmse.to_string())

print(
    "\nBest classification F1:"
)

print(best_f1.to_string())


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

output_file = (
    OUTPUT_DIR
    / "qsl_threshold_sweep.csv"
)

results.to_csv(
    output_file,
    index=False,
)

print(
    f"\nSaved: {output_file}"
)
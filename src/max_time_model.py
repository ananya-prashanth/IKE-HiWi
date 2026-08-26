"""
maximum_time_model.py

Predict the maximum physical simulation time from the
static simulation input parameters.

Target:
    log1p(maximum_time)

Important:
    maximum_time is the physical time corresponding to the
    final recorded simulation state.

    We do NOT assume a fixed relationship between
    sequence_length and maximum_time.
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import PROCESSED_DIR, RANDOM_STATE, CLEAN_METADATA_FILE, SPLIT_FILE

# ------------------------------------------------------------------
# Static input parameters
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
# Load data
# ------------------------------------------------------------------

print("Loading metadata...")

metadata = pd.read_csv(
    CLEAN_METADATA_FILE
)

splits = pd.read_csv(
    SPLIT_FILE
)

print(f"Metadata simulations: {len(metadata)}")
print(f"Split assignments:    {len(splits)}")


# ------------------------------------------------------------------
# Merge split information
# ------------------------------------------------------------------

data = metadata.merge(
    splits,
    on="Simulation No",
    how="inner",
)

print(f"Final dataset:         {len(data)}")

# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

required_columns = (
    INPUT_COLUMNS
    + [
        "maximum_time",
        "split",
    ]
)

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


if data[required_columns].isna().sum().sum() > 0:
    raise ValueError(
        "Missing values detected."
    )

# ------------------------------------------------------------------
# Create target
# ------------------------------------------------------------------

data["target"] = np.log1p(
    data["maximum_time"]
)


# ------------------------------------------------------------------
# Train / validation / test
# ------------------------------------------------------------------

train = data[
    data["split"] == "train"
].copy()

validation = data[
    data["split"] == "validation"
].copy()

test = data[
    data["split"] == "test"
].copy()


X_train = train[INPUT_COLUMNS]
y_train = train["target"]

X_validation = validation[INPUT_COLUMNS]
y_validation = validation["target"]

X_test = test[INPUT_COLUMNS]
y_test = test["target"]


print("\nDataset sizes:")
print(f"Train:       {len(train)}")
print(f"Validation:  {len(validation)}")
print(f"Test:        {len(test)}")

# ------------------------------------------------------------------
# P50 model
# ------------------------------------------------------------------

print("\nTraining P50 maximum-time model...")

model_p50 = HistGradientBoostingRegressor(
    loss="quantile",
    quantile=0.50,
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=RANDOM_STATE,
)

model_p50.fit(
    X_train,
    y_train,
)


# ------------------------------------------------------------------
# P75 model
# ------------------------------------------------------------------

print("Training P75 maximum-time model...")

model_p75 = HistGradientBoostingRegressor(
    loss="quantile",
    quantile=0.75,
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=RANDOM_STATE,
)

model_p75.fit(
    X_train,
    y_train,
)


# ------------------------------------------------------------------
# Predictions
# ------------------------------------------------------------------

pred_p50_log = model_p50.predict(
    X_test
)

pred_p75_log = model_p75.predict(
    X_test
)


# ------------------------------------------------------------------
# Convert back to seconds
# ------------------------------------------------------------------

pred_p50 = np.expm1(
    pred_p50_log
)

pred_p75 = np.expm1(
    pred_p75_log
)

actual_time = test[
    "maximum_time"
].to_numpy()


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

print("\nP50 model")
print("---------")

print(
    "MAE (log maximum time):",
    mean_absolute_error(
        y_test,
        pred_p50_log
    )
)

print(
    "RMSE (log maximum time):",
    np.sqrt(
        mean_squared_error(
            y_test,
            pred_p50_log
        )
    )
)

print(
    "MAE (maximum time [s]):",
    mean_absolute_error(
        actual_time,
        pred_p50
    )
)


print("\nP75 model")
print("---------")

print(
    "MAE (log maximum time):",
    mean_absolute_error(
        y_test,
        pred_p75_log
    )
)

print(
    "RMSE (log maximum time):",
    np.sqrt(
        mean_squared_error(
            y_test,
            pred_p75_log
        )
    )
)

print(
    "MAE (maximum time [s]):",
    mean_absolute_error(
        actual_time,
        pred_p75
    )
)


# ------------------------------------------------------------------
# Quantile coverage
# ------------------------------------------------------------------

coverage_p50 = (
    actual_time <= pred_p50
).mean()

coverage_p75 = (
    actual_time <= pred_p75
).mean()

print("\nQuantile coverage")
print("-----------------")

print(
    f"P50 coverage: {coverage_p50:.3f}"
)

print(
    f"P75 coverage: {coverage_p75:.3f}"
)


# ------------------------------------------------------------------
# Relative errors
# ------------------------------------------------------------------

relative_error_p50 = (
    np.abs(
        pred_p50 - actual_time
    )
    / actual_time
)

relative_error_p75 = (
    np.abs(
        pred_p75 - actual_time
    )
    / actual_time
)


print("\nRelative error")
print("--------------")

print(
    f"P50 MdAPE: "
    f"{np.median(relative_error_p50) * 100:.2f}%"
)

print(
    f"P75 MdAPE: "
    f"{np.median(relative_error_p75) * 100:.2f}%"
)


# ------------------------------------------------------------------
# Save predictions
# ------------------------------------------------------------------

results = test[
    [
        "Simulation No",
        "maximum_time",
        "sequence_length",
    ]
].copy()

results["predicted_maximum_time_p50"] = pred_p50
results["predicted_maximum_time_p75"] = pred_p75

results["relative_error_p50"] = (
    relative_error_p50
)

results["relative_error_p75"] = (
    relative_error_p75
)


prediction_file = (
    PROCESSED_DIR
    / "maximum_time_predictions.csv"
)

results.to_csv(
    prediction_file,
    index=False,
)

print("\nSaved predictions:")
print(prediction_file)
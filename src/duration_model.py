"""
duration_model.py

Train models to predict simulation duration from the
static simulation input parameters.

Target:
    log1p(sequence_length)

Models:
    P50 quantile model
    P75 quantile model

The train/validation/test split is inherited from
preprocess.py and is performed at the simulation level.
"""
from pathlib import Path
import numpy as np
import pandas as pd
'''
HistGradientBoostingRegressor is deliberately being used here INITIALLY 
as it would give us a clean quantile-regression baseline without adding another dependency.

it also directly supports: 
loss="quantile" 
quantile=0.50

AND

quantile=0.75

'''


from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error

#-----------
#CONFIG
#-----------

from config import PROCESSED_DIR, RANDOM_STATE, METADATA_FILE, SPLIT_FILE

# ------------------------------------------------------------------
# Input parameters
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
    METADATA_FILE
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
# Check for missing values
# ------------------------------------------------------------------

missing = data[
    INPUT_COLUMNS + ["sequence_length", "split"]
].isna().sum()

if missing.sum() > 0:
    print("\nMissing values:")
    print(missing[missing > 0])

    raise ValueError(
        "Missing values detected in duration-model data."
    )

# ------------------------------------------------------------------
# Create target
# ------------------------------------------------------------------

data["target"] = np.log1p(
    data["sequence_length"]
)


# ------------------------------------------------------------------
# Create train / validation / test sets
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
# Train P50 model
# ------------------------------------------------------------------

print("\nTraining P50 duration model...")

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
# Train P75 model
# ------------------------------------------------------------------

print("Training P75 duration model...")

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
# Convert back to sequence length
# ------------------------------------------------------------------

pred_p50 = np.expm1(
    pred_p50_log
)

pred_p75 = np.expm1(
    pred_p75_log
)

actual_length = test[
    "sequence_length"
].to_numpy()


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

print("\nP50 model")
print("---------")

print(
    "MAE (log sequence length):",
    mean_absolute_error(
        y_test,
        pred_p50_log
    )
)

print(
    "RMSE (log sequence length):",
    np.sqrt(
        mean_squared_error(
            y_test,
            pred_p50_log
        )
    )
)

print(
    "MAE (sequence length):",
    mean_absolute_error(
        actual_length,
        pred_p50
    )
)


print("\nP75 model")
print("---------")

print(
    "MAE (log sequence length):",
    mean_absolute_error(
        y_test,
        pred_p75_log
    )
)

print(
    "MAE (sequence length):",
    mean_absolute_error(
        actual_length,
        pred_p75
    )
)




# ------------------------------------------------------------------
# Store predictions
# ------------------------------------------------------------------

results = test[
    [
        "Simulation No",
        "sequence_length",
        "maximum_time",
    ]
].copy()

results["predicted_length_p50"] = pred_p50
results["predicted_length_p75"] = pred_p75

results["predicted_time_p50"] = (
    pred_p50 * 10.0   # multiplying by 10 based on the observation that particle outputs are approx. every 10 minutes.
)

results["predicted_time_p75"] = (
    pred_p75 * 10.0
)

results["relative_error_p50"] = (
    np.abs(
        results["predicted_length_p50"]
        - results["sequence_length"]
    )
    / results["sequence_length"]
)

results["relative_error_p75"] = (
    np.abs(
        results["predicted_length_p75"]
        - results["sequence_length"]
    )
    / results["sequence_length"]
)

print("P50 relative error:")
print(results["relative_error_p50"].describe())

print("\nP75 relative error:")
print(results["relative_error_p75"].describe())


mdape_p50 = results["relative_error_p50"].median() * 100
mdape_p75 = results["relative_error_p75"].median() * 100

print(f"P50 MdAPE: {mdape_p50:.2f}%")
print(f"P75 MdAPE: {mdape_p75:.2f}%")


# ------------------------------------------------------------------
# Save predictions
# ------------------------------------------------------------------

prediction_file = (
    PROCESSED_DIR
    / "duration_predictions.csv"
)

results.to_csv(
    prediction_file,
    index=False,
)

print("\nSaved predictions:")
print(prediction_file)

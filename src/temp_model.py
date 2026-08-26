"""
temperature_model.py

Baseline temperature prediction model.

Inputs:
    - 15 static COCOMO simulation parameters
    - particle position: x, y, z
    - actual physical time
    - time-derived features

Target:
    temperature

The train/validation/test split is already performed at the
simulation level in simulation_splits.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
)

# --------------------------------------------------------
# Config
# ---------------------------------------------------------

from config import OUTPUT_DIR

TRAJECTORY_DIR = (
    OUTPUT_DIR / "trajectory"
)

MODEL_DIR = (
    OUTPUT_DIR / "models"
)

PREDICTION_DIR = (
    OUTPUT_DIR / "predictions"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ---------------------------------------------------------
# Features
# ---------------------------------------------------------

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

TARGET = "temperature"

# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------

print( "loading trajectory details:")

train= pd.read_parquet(
    TRAJECTORY_DIR / "trajectory_train.parquet"
)

validation= pd.read_parquet(
    TRAJECTORY_DIR/"trajectory_validation.parquet"
)

test= pd.read_parquet(
    TRAJECTORY_DIR/ "trajectory_test.parquet"
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

# --------------------------------------------------------
# Prepare x and y
# --------------------------------------------------------

x_train= train[INPUT_COLUMNS]
y_train= train[TARGET]

x_validation= validation[INPUT_COLUMNS]
y_validation= validation[TARGET]

x_test= test[INPUT_COLUMNS]
y_test= test[TARGET]

print(
    f"\nNumber of features: "
    f"{len(INPUT_COLUMNS)}"
)

print(
    f"Target: {TARGET}"
)


# ------------------------------------------------------------------
# Train LightGBM
# ------------------------------------------------------------------

print("\n Training LightGBM temperature model:")

model= lgb.LGBMRegressor(
    objective = "regression",
    n_estimators= 3000,
    learning_rate= 0.05,
    num_leaves= 127,
    max_depth= -1,
    subsample= 0.8,
    colsample_bytree= 0.8,
    reg_alpha= 0.1,
    reg_lambda= 0.1,
    random_state=42,
    n_jobs= -1,
)

model.fit(
    x_train, y_train,
    eval_set=[
        (x_train, y_train),
        (x_validation, y_validation),
    ],
    eval_names=[
        "train", "validation",
    ],
    callbacks= [
        lgb.early_stopping(
        stopping_rounds=100,
        verbose=True,
    ),
    lgb.log_evaluation(
        period=100
    ),
    ],
)


#------------------------------------------------------------
# Predictions
# -----------------------------------------------------------

print("\n Generating predictions:")

train_pred= model.predict(
    x_train, num_iteration= model.best_iteration_
)

validation_pred= model.predict(
    x_validation, num_iteration= model.best_iteration_
)

test_pred= model.predict(
    x_test, num_iteration=model.best_iteration_
)

#--------------------------------------------------------------
# metrics
#--------------------------------------------------------------

def calculate_metrics(
    y_true,
    y_pred,
):
    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


train_metrics = calculate_metrics(
    y_train,
    train_pred,
)

validation_metrics = calculate_metrics(
    y_validation,
    validation_pred,
)

test_metrics = calculate_metrics(
    y_test,
    test_pred,
)


# ------------------------------------------------------------------
# Print metrics
# ------------------------------------------------------------------

print("\n")
print("=" * 60)
print("TEMPERATURE MODEL RESULTS")
print("=" * 60)

print(
    f"\nBest iteration: "
    f"{model.best_iteration_}"
)

print("\nTrain:")
for name, value in train_metrics.items():
    print(
        f"{name}: {value:.4f}"
    )

print("\nValidation:")
for name, value in validation_metrics.items():
    print(
        f"{name}: {value:.4f}"
    )

print("\nTest:")
for name, value in test_metrics.items():
    print(
        f"{name}: {value:.4f}"
    )


# ------------------------------------------------------------------
# Temperature-bin evaluation
# ------------------------------------------------------------------

print("\n")
print("=" * 60)
print("TEST PERFORMANCE BY TEMPERATURE")
print("=" * 60)

test_results = test[
    [
        "Simulation No",
        "particle_id",
        "time",
        "temperature",
    ]
].copy()

test_results["prediction"] = test_pred

test_results["absolute_error"] = (
    test_results["prediction"]
    - test_results["temperature"]
).abs()


bins = [
    0,
    500,
    750,
    1000,
    1250,
    1500,
    1750,
    np.inf,
]

labels = [
    "0-500",
    "500-750",
    "750-1000",
    "1000-1250",
    "1250-1500",
    "1500-1750",
    "1750+",
]

test_results["temperature_bin"] = pd.cut(
    test_results["temperature"],
    bins=bins,
    labels=labels,
    include_lowest=True,
)

bin_results = (
    test_results
    .groupby(
        "temperature_bin",
        observed=True,
    )
    .agg(
        N=("temperature", "size"),
        MAE=("absolute_error", "mean"),
    )
)

print(bin_results)


# ------------------------------------------------------------------
# Save predictions
# ------------------------------------------------------------------

prediction_file = (
    PREDICTION_DIR
    / "temperature_predictions_test.parquet"
)

test_results.to_parquet(
    prediction_file,
    index=False,
)


# ------------------------------------------------------------------
# Feature importance
# ------------------------------------------------------------------

feature_importance = pd.DataFrame(
    {
        "feature": INPUT_COLUMNS,
        "importance": model.feature_importances_,
    }
).sort_values(
    "importance",
    ascending=False,
)

print("\n")
print("=" * 60)
print("TOP FEATURE IMPORTANCE")
print("=" * 60)

print(
    feature_importance.head(20)
)


feature_importance_file = (
    MODEL_DIR
    / "temperature_feature_importance.csv"
)

feature_importance.to_csv(
    feature_importance_file,
    index=False,
)


# ------------------------------------------------------------------
# Save model
# ------------------------------------------------------------------

model_file = (
    MODEL_DIR
    / "temperature_lightgbm.txt"
)

model.booster_.save_model(
    str(model_file)
)

print("\nSaved:")
print(model_file)
print(prediction_file)
print(feature_importance_file)

print("\nTemperature model complete.")
"""
trajectory_model.py

Task 3 - Baseline trajectory model.

Predict one physical variable from the balance files as a function of:

    - 15 initial simulation parameters
    - actual simulation time
    - normalized time features
    - maximum simulation time

First target:
    eg = internal energy of the gas phase [J]

This is a baseline LightGBM model.

The full trajectory datasets are retained for evaluation, while
training uses a fixed number of representative timesteps per
simulation to keep memory and training time manageable.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# Paths
# ============================================================

from config import PREDICTION_DIR, PROJECT_ROOT, MODEL_DIR

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "trajectory"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Files
# ============================================================

TRAIN_FILE = (
    DATA_DIR
    / "trajectory_train.parquet"
)

VALIDATION_FILE = (
    DATA_DIR
    / "trajectory_validation.parquet"
)

TEST_FILE = (
    DATA_DIR
    / "trajectory_test.parquet"
)


# ============================================================
# Configuration
# ============================================================

TARGET = "eg"

# Number of trajectory points sampled from each training
# simulation.
POINTS_PER_SIMULATION = 150

RANDOM_STATE = 42


# ============================================================
# Features
# ============================================================

STATIC_FEATURES = [
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


TIME_FEATURES = [
    "time",
    "t_norm",
    "log1p_t_norm",
    "sqrt_t_norm",
    "t_norm_squared",
    "log1p_maximum_time",
]


FEATURES = (
    STATIC_FEATURES
    + TIME_FEATURES
)


# ============================================================
# Load datasets
# ============================================================

print("Loading trajectory datasets...")

train = pd.read_parquet(
    TRAIN_FILE
)

validation = pd.read_parquet(
    VALIDATION_FILE
)

test = pd.read_parquet(
    TEST_FILE
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

print(
    f"Number of features: {len(FEATURES)}"
)

print(
    f"Target: {TARGET}"
)


# ============================================================
# Check required columns
# ============================================================

required = (
    FEATURES
    + [
        "Simulation No",
        TARGET,
    ]
)

for dataset_name, dataset in [
    ("train", train),
    ("validation", validation),
    ("test", test),
]:

    missing = [
        col
        for col in required
        if col not in dataset.columns
    ]

    if missing:

        raise ValueError(
            f"{dataset_name} is missing: "
            f"{missing}"
        )


# ============================================================
# Sample training trajectories
# ============================================================

def sample_per_simulation(
    data,
    points_per_simulation,
    random_state,
):
    """
    Sample a fixed number of trajectory points per simulation.

    Sampling is performed independently for each simulation so
    that long simulations do not dominate the training set.
    """

    sampled = (
        data
        .groupby(
            "Simulation No",
            group_keys=False
        )
        .apply(
            lambda group:
            group.sample(
                n=min(
                    points_per_simulation,
                    len(group)
                ),
                random_state=random_state
            )
        )
        .reset_index(drop=True)
    )

    return sampled


print()
print(
    "Sampling training trajectories..."
)

train_sample = sample_per_simulation(
    train,
    POINTS_PER_SIMULATION,
    RANDOM_STATE,
)

print(
    f"Training rows after sampling: "
    f"{len(train_sample):,}"
)

print(
    f"Training simulations: "
    f"{train_sample['Simulation No'].nunique()}"
)


# ============================================================
# Prepare X/y
# ============================================================

X_train = train_sample[
    FEATURES
]

y_train = train_sample[
    TARGET
]

X_validation = validation[
    FEATURES
]

y_validation = validation[
    TARGET
]


# ============================================================
# Train LightGBM
# ============================================================

print()
print("=" * 60)
print("Training LightGBM trajectory model")
print("=" * 60)

model = lgb.LGBMRegressor(
    objective="regression",

    n_estimators=3000,

    learning_rate=0.05,

    num_leaves=63,

    max_depth=-1,

    subsample=0.8,

    colsample_bytree=0.8,

    reg_alpha=0.0,

    reg_lambda=0.0,

    random_state=RANDOM_STATE,

    n_jobs=64,
)

model.fit(
    X_train,
    y_train,

    eval_set=[
        (
            X_validation,
            y_validation
        )
    ],

    callbacks=[
        lgb.early_stopping(
            stopping_rounds=100
        ),
        lgb.log_evaluation(
            period=100
        ),
    ],
)


# ============================================================
# Predictions
# ============================================================

print()
print("Generating predictions...")

train_pred = model.predict(
    X_train
)

validation_pred = model.predict(
    X_validation
)

X_test = test[
    FEATURES
]

y_test = test[
    TARGET
]

test_pred = model.predict(
    X_test
)


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    actual,
    predicted,
):
    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    # Avoid division by zero
    nonzero = actual != 0

    if nonzero.any():

        ape = (
            np.abs(
                (
                    predicted[nonzero]
                    -
                    actual[nonzero]
                )
                /
                actual[nonzero]
            )
            * 100
        )

        mdape = np.median(
            ape
        )

    else:

        mdape = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MdAPE": mdape,
    }


train_metrics = calculate_metrics(
    y_train.to_numpy(),
    train_pred,
)

validation_metrics = calculate_metrics(
    y_validation.to_numpy(),
    validation_pred,
)

test_metrics = calculate_metrics(
    y_test.to_numpy(),
    test_pred,
)


# ============================================================
# Print results
# ============================================================

print()
print("=" * 60)
print("EG TRAJECTORY MODEL RESULTS")
print("=" * 60)

print(
    f"\nBest iteration: "
    f"{model.best_iteration_}"
)

print("\nTrain:")

for key, value in train_metrics.items():

    print(
        f"{key}: {value:.6f}"
    )

print("\nValidation:")

for key, value in validation_metrics.items():

    print(
        f"{key}: {value:.6f}"
    )

print("\nTest:")

for key, value in test_metrics.items():

    print(
        f"{key}: {value:.6f}"
    )


# ============================================================
# Feature importance
# ============================================================

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_,
})

importance = (
    importance
    .sort_values(
        "importance",
        ascending=False
    )
)

print()
print("=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# Save model
# ============================================================

model_path = (
    MODEL_DIR
    / "trajectory_eg_lightgbm.txt"
)

model.booster_.save_model(
    str(model_path)
)


# ============================================================
# Save test predictions
# ============================================================

prediction_output = test[
    [
        "Simulation No",
        "time",
        "t_norm",
        TARGET,
    ]
].copy()

prediction_output[
    "predicted_eg"
] = test_pred

prediction_path = (
    PREDICTION_DIR
    / "trajectory_eg_predictions_test.parquet"
)

prediction_output.to_parquet(
    prediction_path,
    index=False
)


# ============================================================
# Save feature importance
# ============================================================

importance_path = (
    MODEL_DIR
    / "trajectory_eg_feature_importance.csv"
)

importance.to_csv(
    importance_path,
    index=False
)


# ============================================================
# Save metrics
# ============================================================

metrics_output = pd.DataFrame({
    "split": [
        "train",
        "validation",
        "test",
    ],

    "MAE": [
        train_metrics["MAE"],
        validation_metrics["MAE"],
        test_metrics["MAE"],
    ],

    "RMSE": [
        train_metrics["RMSE"],
        validation_metrics["RMSE"],
        test_metrics["RMSE"],
    ],

    "R2": [
        train_metrics["R2"],
        validation_metrics["R2"],
        test_metrics["R2"],
    ],

    "MdAPE": [
        train_metrics["MdAPE"],
        validation_metrics["MdAPE"],
        test_metrics["MdAPE"],
    ],
})

metrics_path = (
    MODEL_DIR
    / "trajectory_eg_metrics.csv"
)

metrics_output.to_csv(
    metrics_path,
    index=False
)


print()
print("=" * 60)
print("SAVED")
print("=" * 60)

print(
    model_path
)

print(
    prediction_path
)

print(
    importance_path
)

print(
    metrics_path
)

print()
print("Trajectory baseline complete.")
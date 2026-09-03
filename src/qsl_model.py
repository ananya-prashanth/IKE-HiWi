"""
qsl_model.py

Specialized Task 3 model for qsl.

qsl has a sparse/event-like structure:
    - qsl starts at zero
    - most timesteps are zero
    - meaningful nonzero values are generally > 10,000 W in magnitude
    - active simulations can switch between inactive and active periods

Model structure
---------------
1. Classification:
       parameters + time features
              -> P(qsl is active)

2. Regression:
       parameters + time features
              -> signed qsl value
       trained only on active points

3. Reconstruction:
       inactive -> qsl = 0
       active   -> predicted signed qsl

The existing single-regression qsl model remains the baseline.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import lightgbm as lgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# Paths
# ============================================================

from config import PROJECT_ROOT, PROCESSED_DIR, TRAJECTORY_DIR, MODEL_DIR, PREDICTION_DIR


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Configuration
# ============================================================

TARGET = "qsl"

ACTIVE_THRESHOLD = 10_000.0

RANDOM_STATE = 42

# Number of trajectory points sampled per simulation.
SAMPLES_PER_SIMULATION = 150

# Parallelism
N_THREADS = 32


# ============================================================
# Features
# ============================================================

FEATURES = [
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
    "time",
    "t_norm",
    "log1p_t_norm",
    "sqrt_t_norm",
    "t_norm_squared",
    "log1p_maximum_time",
]


# ============================================================
# Utility functions
# ============================================================

def add_activity_target(df):
    """
    Create binary qsl activity target.

    Active means |qsl| > ACTIVE_THRESHOLD.
    """

    df = df.copy()

    df["qsl_active"] = (
        df[TARGET].abs() > ACTIVE_THRESHOLD
    ).astype(np.int8)

    return df


def sample_trajectories(df):
    """
    Sample a fixed number of points from each simulation.

    Sampling is performed separately for each simulation so that
    long simulations do not dominate training.
    """

    sampled = (
        df.groupby(
            "Simulation No",
            group_keys=False,
        )
        .apply(
            lambda g: g.sample(
                n=min(
                    SAMPLES_PER_SIMULATION,
                    len(g),
                ),
                random_state=RANDOM_STATE,
            )
        )
        .reset_index(drop=True)
    )

    return sampled


def calculate_metrics(y_true, y_pred):
    """
    Regression metrics.
    """

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

    return mae, rmse, r2


# ============================================================
# Load data
# ============================================================

print("=" * 70)
print("QSL SPECIALIZED MODEL")
print("=" * 70)

print("\nLoading trajectory datasets...")

train = pd.read_parquet(
    TRAJECTORY_DIR / "trajectory_train.parquet"
)

validation = pd.read_parquet(
    TRAJECTORY_DIR / "trajectory_validation.parquet"
)

test = pd.read_parquet(
    TRAJECTORY_DIR / "trajectory_test.parquet"
)

print(f"Train:       {len(train):,} rows")
print(f"Validation:  {len(validation):,} rows")
print(f"Test:        {len(test):,} rows")


# ============================================================
# Create activity target
# ============================================================

print("\nCreating qsl activity target...")

train = add_activity_target(train)
validation = add_activity_target(validation)
test = add_activity_target(test)

print(
    "\nTraining activity distribution:"
)

print(
    train["qsl_active"]
    .value_counts()
    .rename(
        index={
            0: "Inactive",
            1: "Active",
        }
    )
)

print(
    "\nValidation activity distribution:"
)

print(
    validation["qsl_active"]
    .value_counts()
    .rename(
        index={
            0: "Inactive",
            1: "Active",
        }
    )
)

print(
    "\nTest activity distribution:"
)

print(
    test["qsl_active"]
    .value_counts()
    .rename(
        index={
            0: "Inactive",
            1: "Active",
        }
    )
)


# ============================================================
# Sample training trajectories
# ============================================================

print("\nSampling training trajectories...")

train_sample = sample_trajectories(
    train
)

print(
    f"Training rows: {len(train_sample):,}"
)

print(
    "Training simulations:",
    train_sample["Simulation No"].nunique(),
)


# ============================================================
# Prepare classification data
# ============================================================

X_train = train_sample[FEATURES]
y_train_cls = train_sample["qsl_active"]

X_val = validation[FEATURES]
y_val_cls = validation["qsl_active"]

X_test = test[FEATURES]
y_test_cls = test["qsl_active"]


# ============================================================
# Stage 1 — Activity classifier
# ============================================================

print("\n" + "=" * 70)
print("STAGE 1 — QSL ACTIVITY CLASSIFIER")
print("=" * 70)

classifier = lgb.LGBMClassifier(
    objective="binary",

    n_estimators=3000,

    learning_rate=0.03,

    num_leaves=64,

    max_depth=-1,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=RANDOM_STATE,

    n_jobs=N_THREADS,

    verbosity=-1,
)

classifier.fit(
    X_train,
    y_train_cls,

    eval_set=[
        (
            X_val,
            y_val_cls,
        )
    ],

    callbacks=[
        lgb.early_stopping(
            100,
            verbose=True,
        )
    ],
)

# ------------------------------------------------------------
# Classification predictions
# ------------------------------------------------------------

test_probability = classifier.predict_proba(
    X_test
)[:, 1]

# Default classification threshold
CLASSIFICATION_THRESHOLD = 0.2

test_active_pred = (
    test_probability
    >= CLASSIFICATION_THRESHOLD
).astype(np.int8)


# ============================================================
# Classification metrics
# ============================================================

accuracy = accuracy_score(
    y_test_cls,
    test_active_pred,
)

precision = precision_score(
    y_test_cls,
    test_active_pred,
    zero_division=0,
)

recall = recall_score(
    y_test_cls,
    test_active_pred,
    zero_division=0,
)

f1 = f1_score(
    y_test_cls,
    test_active_pred,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_test_cls,
    test_probability,
)

cm = confusion_matrix(
    y_test_cls,
    test_active_pred,
)

print("\nClassification results:")

print(
    f"Accuracy:  {accuracy:.6f}"
)

print(
    f"Precision: {precision:.6f}"
)

print(
    f"Recall:    {recall:.6f}"
)

print(
    f"F1:        {f1:.6f}"
)

print(
    f"ROC-AUC:   {roc_auc:.6f}"
)

print("\nConfusion matrix:")

print(cm)


# ============================================================
# Stage 2 — Active-value regression
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2 — ACTIVE QSL REGRESSION")
print("=" * 70)

active_train = train_sample[
    train_sample["qsl_active"] == 1
]

active_validation = validation[
    validation["qsl_active"] == 1
]

active_test = test[
    test["qsl_active"] == 1
]

print(
    f"Active training rows: "
    f"{len(active_train):,}"
)

print(
    f"Active validation rows: "
    f"{len(active_validation):,}"
)

print(
    f"Active test rows: "
    f"{len(active_test):,}"
)

X_train_reg = active_train[FEATURES]
y_train_reg = active_train[TARGET]

X_val_reg = active_validation[FEATURES]
y_val_reg = active_validation[TARGET]

X_test_reg = active_test[FEATURES]
y_test_reg = active_test[TARGET]


regressor = lgb.LGBMRegressor(
    objective="regression",

    n_estimators=3000,

    learning_rate=0.03,

    num_leaves=64,

    max_depth=-1,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=RANDOM_STATE,

    n_jobs=N_THREADS,

    verbosity=-1,
)

regressor.fit(
    X_train_reg,
    y_train_reg,

    eval_set=[
        (
            X_val_reg,
            y_val_reg,
        )
    ],

    callbacks=[
        lgb.early_stopping(
            100,
            verbose=True,
        )
    ],
)


# ============================================================
# Reconstruct full qsl trajectory
# ============================================================

print("\n" + "=" * 70)
print("RECONSTRUCTING FULL QSL TEST TRAJECTORY")
print("=" * 70)

regression_prediction = regressor.predict(
    X_test
)

# Classifier decides whether the value is active.
# Regression supplies the signed value.
qsl_prediction = np.where(
    test_active_pred == 1,
    regression_prediction,
    0.0,
)


# ============================================================
# Full trajectory metrics
# ============================================================

actual = test[TARGET].to_numpy()

mae, rmse, r2 = calculate_metrics(
    actual,
    qsl_prediction,
)

print("\nFull reconstructed qsl metrics:")

print(
    f"MAE:  {mae:.6g}"
)

print(
    f"RMSE: {rmse:.6g}"
)

print(
    f"R²:   {r2:.6f}"
)


# ============================================================
# Active-point regression metrics
# ============================================================

actual_active = actual[
    y_test_cls.to_numpy() == 1
]

pred_active = qsl_prediction[
    y_test_cls.to_numpy() == 1
]

active_mae, active_rmse, active_r2 = (
    calculate_metrics(
        actual_active,
        pred_active,
    )
)

print("\nActive-point metrics:")

print(
    f"MAE:  {active_mae:.6g}"
)

print(
    f"RMSE: {active_rmse:.6g}"
)

print(
    f"R²:   {active_r2:.6f}"
)


# ============================================================
# Save reconstructed predictions
# ============================================================

prediction_output = test[
    [
        "Simulation No",
        "time",
        TARGET,
    ]
].copy()

prediction_output[
    "qsl_active_probability"
] = test_probability

prediction_output[
    "qsl_active_predicted"
] = test_active_pred

prediction_output[
    "predicted_qsl"
] = qsl_prediction

prediction_file = (
    PREDICTION_DIR
    / "qsl_specialized_predictions_test.parquet"
)

prediction_output.to_parquet(
    prediction_file,
    index=False,
)

print(
    "\nSaved predictions:"
)

print(prediction_file)


# ============================================================
# Save models
# ============================================================

classifier_file = (
    MODEL_DIR
    / "qsl_activity_lightgbm.txt"
)

regressor_file = (
    MODEL_DIR
    / "qsl_active_regression_lightgbm.txt"
)

classifier.booster_.save_model(
    classifier_file
)

regressor.booster_.save_model(
    regressor_file
)

print("\nSaved models:")

print(classifier_file)
print(regressor_file)


# ============================================================
# Save metrics
# ============================================================

metrics = pd.DataFrame(
    [
        {
            "model": "qsl_activity_classifier",
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
        },
        {
            "model": "qsl_full_reconstruction",
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        },
        {
            "model": "qsl_active_regression",
            "MAE": active_mae,
            "RMSE": active_rmse,
            "R2": active_r2,
        },
    ]
)

metrics_file = (
    MODEL_DIR
    / "qsl_specialized_metrics.csv"
)

metrics.to_csv(
    metrics_file,
    index=False,
)

print(
    "\nSaved metrics:"
)

print(metrics_file)

print("\n" + "=" * 70)
print("QSL SPECIALIZED MODEL COMPLETE")
print("=" * 70)
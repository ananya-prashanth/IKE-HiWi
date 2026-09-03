"""
qsl_state_model.py

Compare two qsl models:

MODEL A — baseline
    simulation parameters + time -> qsl

MODEL B — state augmented
    simulation parameters + time
    + mg, ml, mp, eg, el, ep
    -> qsl

Both use:
    - qsl activity threshold = 10,000 W
    - classifier + active-regime regression
    - classification threshold = 0.35
    - same train/validation/test split
    - same trajectory sampling

This experiment uses TRUE physical state variables.
It is therefore a diagnostic experiment to determine whether
the evolving thermodynamic state contains useful information
for qsl prediction.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# Paths
# ============================================================

from config import PROJECT_ROOT, TRAJECTORY_DIR, MODEL_DIR, PREDICTION_DIR


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

CLASSIFICATION_THRESHOLD = 0.35

RANDOM_STATE = 42

SAMPLES_PER_SIMULATION = 150

N_THREADS = 32


# ============================================================
# Features
# ============================================================

BASE_FEATURES = [
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


STATE_FEATURES = [
    "mg",
    "ml",
    "mp",
    "eg",
    "el",
    "ep",
]


STATE_FEATURES_AVAILABLE = [
    x for x in STATE_FEATURES
    if x in BASE_FEATURES
]


# ============================================================
# Functions
# ============================================================

def add_activity_target(df):
    df = df.copy()

    df["qsl_active"] = (
        df[TARGET].abs()
        > ACTIVE_THRESHOLD
    ).astype(np.int8)

    return df


def sample_trajectories(df):

    return (
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


def regression_metrics(y_true, y_pred):

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


def train_qsl_model(
    train,
    validation,
    test,
    features,
    model_name,
):

    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    print(
        f"\nNumber of features: "
        f"{len(features)}"
    )

    print(
        "\nFeatures:"
    )

    print(
        features
    )

    # --------------------------------------------------------
    # Sample training trajectories
    # --------------------------------------------------------

    train_sample = sample_trajectories(
        train
    )

    print(
        f"\nTraining rows: "
        f"{len(train_sample):,}"
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    X_train = train_sample[features]
    y_train = train_sample["qsl_active"]

    X_val = validation[features]
    y_val = validation["qsl_active"]

    X_test = test[features]
    y_test = test["qsl_active"]

    print(
        "\nTraining activity classifier..."
    )

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
        y_train,

        eval_set=[
            (
                X_val,
                y_val,
            )
        ],

        callbacks=[
            lgb.early_stopping(
                100,
                verbose=True,
            )
        ],
    )

    # --------------------------------------------------------
    # Classification predictions
    # --------------------------------------------------------

    probability = classifier.predict_proba(
        X_test
    )[:, 1]

    predicted_active = (
        probability
        >= CLASSIFICATION_THRESHOLD
    ).astype(np.int8)

    precision = precision_score(
        y_test,
        predicted_active,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predicted_active,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predicted_active,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probability,
    )

    print(
        "\nClassification:"
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

    # --------------------------------------------------------
    # Active regression
    # --------------------------------------------------------

    active_train = train_sample[
        train_sample["qsl_active"] == 1
    ]

    active_validation = validation[
        validation["qsl_active"] == 1
    ]

    print(
        f"\nActive training rows: "
        f"{len(active_train):,}"
    )

    print(
        f"Active validation rows: "
        f"{len(active_validation):,}"
    )

    X_train_reg = active_train[features]

    y_train_reg = active_train[TARGET]

    X_val_reg = active_validation[features]

    y_val_reg = active_validation[TARGET]

    print(
        "\nTraining active qsl regressor..."
    )

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

    # --------------------------------------------------------
    # Regression prediction
    # --------------------------------------------------------

    regression_prediction = (
        regressor.predict(
            X_test
        )
    )

    # --------------------------------------------------------
    # Reconstruct qsl
    # --------------------------------------------------------

    predicted_qsl = np.where(
        predicted_active == 1,
        regression_prediction,
        0.0,
    )

    actual_qsl = (
        test[TARGET]
        .to_numpy()
    )

    # --------------------------------------------------------
    # Full trajectory metrics
    # --------------------------------------------------------

    mae, rmse, r2 = regression_metrics(
        actual_qsl,
        predicted_qsl,
    )

    # --------------------------------------------------------
    # Active point metrics
    # --------------------------------------------------------

    active_mask = (
        y_test.to_numpy() == 1
    )

    active_mae, active_rmse, active_r2 = (
        regression_metrics(
            actual_qsl[active_mask],
            predicted_qsl[active_mask],
        )
    )

    print(
        "\nFull qsl trajectory:"
    )

    print(
        f"MAE:  {mae:.6g}"
    )

    print(
        f"RMSE: {rmse:.6g}"
    )

    print(
        f"R2:   {r2:.6f}"
    )

    print(
        "\nActive qsl:"
    )

    print(
        f"MAE:  {active_mae:.6g}"
    )

    print(
        f"RMSE: {active_rmse:.6g}"
    )

    print(
        f"R2:   {active_r2:.6f}"
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    predictions = test[
        [
            "Simulation No",
            "time",
            "qsl",
        ]
    ].copy()

    predictions[
        "qsl_active_probability"
    ] = probability

    predictions[
        "qsl_active_predicted"
    ] = predicted_active

    predictions[
        "predicted_qsl"
    ] = predicted_qsl

    prediction_file = (
        PREDICTION_DIR
        / f"qsl_{model_name}_predictions_test.parquet"
    )

    predictions.to_parquet(
        prediction_file,
        index=False,
    )

    # --------------------------------------------------------
    # Save models
    # --------------------------------------------------------

    classifier_file = (
        MODEL_DIR
        / f"qsl_{model_name}_activity.txt"
    )

    regressor_file = (
        MODEL_DIR
        / f"qsl_{model_name}_regression.txt"
    )

    classifier.booster_.save_model(
        classifier_file
    )

    regressor.booster_.save_model(
        regressor_file
    )

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {
        "model": model_name,

        "features": len(features),

        "classifier_precision": precision,

        "classifier_recall": recall,

        "classifier_f1": f1,

        "classifier_roc_auc": roc_auc,

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2,

        "active_MAE": active_mae,

        "active_RMSE": active_rmse,

        "active_R2": active_r2,

        "predicted_active_fraction":
            predicted_active.mean(),

        "classifier_best_iteration":
            classifier.best_iteration_,

        "regressor_best_iteration":
            regressor.best_iteration_,
    }


# ============================================================
# Load datasets
# ============================================================

print("=" * 70)
print("QSL STATE-AUGMENTED EXPERIMENT")
print("=" * 70)

print(
    "\nLoading trajectory datasets..."
)

train = pd.read_parquet(
    TRAJECTORY_DIR
    / "trajectory_train.parquet"
)

validation = pd.read_parquet(
    TRAJECTORY_DIR
    / "trajectory_validation.parquet"
)

test = pd.read_parquet(
    TRAJECTORY_DIR
    / "trajectory_test.parquet"
)

print(
    f"Train:       {len(train):,}"
)

print(
    f"Validation:  {len(validation):,}"
)

print(
    f"Test:        {len(test):,}"
)


# ============================================================
# Activity target
# ============================================================

train = add_activity_target(
    train
)

validation = add_activity_target(
    validation
)

test = add_activity_target(
    test
)


# ============================================================
# Verify state columns
# ============================================================

missing_state = [
    x for x in STATE_FEATURES
    if x not in train.columns
]

if missing_state:

    raise ValueError(
        "Missing state variables: "
        f"{missing_state}"
    )


# ============================================================
# Define models
# ============================================================

baseline_features = BASE_FEATURES.copy()

state_features = (
    BASE_FEATURES
    + STATE_FEATURES
)


# ============================================================
# Train baseline
# ============================================================

baseline_result = train_qsl_model(

    train=train,

    validation=validation,

    test=test,

    features=baseline_features,

    model_name="baseline",
)


# ============================================================
# Train state-augmented model
# ============================================================

state_result = train_qsl_model(

    train=train,

    validation=validation,

    test=test,

    features=state_features,

    model_name="state",
)


# ============================================================
# Comparison
# ============================================================

comparison = pd.DataFrame(
    [
        baseline_result,
        state_result,
    ]
)

print(
    "\n" + "=" * 70
)

print(
    "FINAL MODEL COMPARISON"
)

print(
    "=" * 70
)

print(
    comparison[
        [
            "model",
            "features",
            "classifier_precision",
            "classifier_recall",
            "classifier_f1",
            "classifier_roc_auc",
            "MAE",
            "RMSE",
            "R2",
            "active_MAE",
            "active_RMSE",
            "active_R2",
            "predicted_active_fraction",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# Improvement
# ============================================================

baseline_r2 = baseline_result["R2"]
state_r2 = state_result["R2"]

baseline_mae = baseline_result["MAE"]
state_mae = state_result["MAE"]

baseline_rmse = baseline_result["RMSE"]
state_rmse = state_result["RMSE"]

print(
    "\n" + "=" * 70
)

print(
    "STATE AUGMENTATION IMPROVEMENT"
)

print(
    "=" * 70
)

print(
    f"\nR2:"
)

print(
    f"Baseline:        {baseline_r2:.6f}"
)

print(
    f"State augmented: {state_r2:.6f}"
)

print(
    f"Change:          "
    f"{state_r2 - baseline_r2:+.6f}"
)

print(
    "\nMAE:"
)

print(
    f"Baseline:        {baseline_mae:.6g}"
)

print(
    f"State augmented: {state_mae:.6g}"
)

print(
    f"Change:          "
    f"{state_mae - baseline_mae:+.6g}"
)

print(
    "\nRMSE:"
)

print(
    f"Baseline:        {baseline_rmse:.6g}"
)

print(
    f"State augmented: {state_rmse:.6g}"
)

print(
    f"Change:          "
    f"{state_rmse - baseline_rmse:+.6g}"
)


# ============================================================
# Save comparison
# ============================================================

comparison_file = (
    MODEL_DIR
    / "qsl_state_model_comparison.csv"
)

comparison.to_csv(
    comparison_file,
    index=False,
)

print(
    f"\nSaved comparison:"
)

print(
    comparison_file
)

print(
    "\nExperiment complete."
)
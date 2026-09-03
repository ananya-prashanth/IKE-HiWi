"""
traj_all_models.py

Task 3 - Full physical trajectory prediction.

Predict the progression of the 23 physical variables contained
in the balance files throughout each simulation.

Features:
    - 15 static simulation parameters
    - actual simulation time
    - normalized/time-derived features

Targets:
    - 23 balance-file physical variables

Model:
    - LightGBM regression
    - one independent model per target

Parallelization:
    - multiple target models are trained simultaneously
    - each LightGBM model uses a controlled number of CPU threads

Important:
    Training uses a fixed number of representative trajectory
    points per simulation. Validation and test use the complete
    trajectories.

Particle temperature files are NOT used.
"""


# ============================================================
# Imports
# ============================================================

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# Configuration
# ============================================================

from config import PROJECT_ROOT, MODEL_DIR, PREDICTION_DIR
 
DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "trajectory"
)




# ------------------------------------------------------------
# Parallelization
# ------------------------------------------------------------

N_PARALLEL_TARGETS = 8

LIGHTGBM_THREADS = 32


# ------------------------------------------------------------
# Training sampling
# ------------------------------------------------------------

POINTS_PER_SIMULATION = 150

RANDOM_STATE = 42


# ------------------------------------------------------------
# LightGBM settings
# ------------------------------------------------------------

N_ESTIMATORS = 3000

LEARNING_RATE = 0.05

NUM_LEAVES = 63

EARLY_STOPPING_ROUNDS = 100


# ============================================================
# Input files
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
# Feature definitions
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
# Targets
# ============================================================

TARGETS = [
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
    "qsl",
    "pow",
]


# ============================================================
# Output directories
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Utility functions
# ============================================================

def sample_training_data(
    data,
    points_per_simulation,
    random_state,
):
    """
    Sample approximately the same number of trajectory points
    from every simulation.

    This prevents simulations with many timesteps from dominating
    the training set.
    """

    rng = np.random.default_rng(
        random_state
    )

    groups = []

    for sim_id, group in data.groupby(
        "Simulation No",
        sort=False
    ):

        n = min(
            points_per_simulation,
            len(group)
        )

        # Use random integer positions rather than DataFrameGroupBy.apply.
        indices = rng.choice(
            len(group),
            size=n,
            replace=False
        )

        groups.append(
            group.iloc[indices]
        )

    return pd.concat(
        groups,
        ignore_index=True
    )


def calculate_metrics(
    actual,
    predicted,
):
    """
    Calculate robust regression metrics.

    MdAPE is only calculated on non-zero actual values.
    """

    actual = np.asarray(
        actual,
        dtype=np.float64
    )

    predicted = np.asarray(
        predicted,
        dtype=np.float64
    )

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

    # --------------------------------------------------------
    # Median absolute percentage error
    # --------------------------------------------------------

    nonzero = (
        np.abs(actual) > 1e-12
    )

    if np.any(nonzero):

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
            * 100.0
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


# ============================================================
# Train one target
# ============================================================

def train_target(
    target,
    train_sample,
    validation,
    test,
):
    """
    Train one LightGBM model and evaluate it.

    This function is executed in a separate process.
    """

    print(
        f"[{target}] Starting..."
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X_train = train_sample[
        FEATURES
    ]

    y_train = train_sample[
        target
    ]

    X_validation = validation[
        FEATURES
    ]

    y_validation = validation[
        target
    ]

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = lgb.LGBMRegressor(

        objective="regression",

        n_estimators=N_ESTIMATORS,

        learning_rate=LEARNING_RATE,

        num_leaves=NUM_LEAVES,

        max_depth=-1,

        subsample=0.8,

        colsample_bytree=0.8,

        reg_alpha=0.0,

        reg_lambda=0.0,

        random_state=RANDOM_STATE,

        n_jobs=LIGHTGBM_THREADS,

        verbosity=-1,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(

        X_train,
        y_train,

        eval_X=X_validation,

        eval_y=y_validation,

        callbacks=[
            lgb.early_stopping(
                EARLY_STOPPING_ROUNDS,
                verbose=False
            ),
        ],
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

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
        target
    ]

    test_pred = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    train_metrics = calculate_metrics(
        y_train,
        train_pred
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_pred
    )

    test_metrics = calculate_metrics(
        y_test,
        test_pred
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR
        / f"trajectory_{target}_lightgbm.txt"
    )

    model.booster_.save_model(
        str(model_path)
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_output = test[
        [
            "Simulation No",
            "time",
            "t_norm",
            target,
        ]
    ].copy()

    prediction_output[
        f"predicted_{target}"
    ] = test_pred

    prediction_path = (
        PREDICTION_DIR
        / f"trajectory_{target}_predictions_test.parquet"
    )

    prediction_output.to_parquet(
        prediction_path,
        index=False
    )

    # --------------------------------------------------------
    # Save feature importance
    # --------------------------------------------------------

    importance_path = (
        MODEL_DIR
        / f"trajectory_{target}_feature_importance.csv"
    )

    importance.to_csv(
        importance_path,
        index=False
    )

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    result = {
        "target": target,

        "best_iteration":
            model.best_iteration_,

        "train_MAE":
            train_metrics["MAE"],

        "train_RMSE":
            train_metrics["RMSE"],

        "train_R2":
            train_metrics["R2"],

        "train_MdAPE":
            train_metrics["MdAPE"],

        "validation_MAE":
            validation_metrics["MAE"],

        "validation_RMSE":
            validation_metrics["RMSE"],

        "validation_R2":
            validation_metrics["R2"],

        "validation_MdAPE":
            validation_metrics["MdAPE"],

        "test_MAE":
            test_metrics["MAE"],

        "test_RMSE":
            test_metrics["RMSE"],

        "test_R2":
            test_metrics["R2"],

        "test_MdAPE":
            test_metrics["MdAPE"],
    }

    print(
        f"[{target}] Finished "
        f"(iteration "
        f"{model.best_iteration_})"
    )

    return result


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("TASK 3 — FULL TRAJECTORY MODEL")
    print("=" * 70)

    print()
    print(
        f"Parallel targets: "
        f"{N_PARALLEL_TARGETS}"
    )

    print(
        f"LightGBM threads/model: "
        f"{LIGHTGBM_THREADS}"
    )

    print(
        f"Total requested CPU threads: "
        f"{N_PARALLEL_TARGETS * LIGHTGBM_THREADS}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print()
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
        f"Features:    {len(FEATURES)}"
    )

    print(
        f"Targets:     {len(TARGETS)}"
    )

    # --------------------------------------------------------
    # Check columns
    # --------------------------------------------------------

    required_columns = (
        ["Simulation No"]
        + FEATURES
        + TARGETS
    )

    for name, data in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:

        missing = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing:

            raise ValueError(
                f"{name} is missing columns: "
                f"{missing}"
            )

    # --------------------------------------------------------
    # Sample training data ONCE
    # --------------------------------------------------------

    print()
    print(
        "Sampling training trajectories..."
    )

    train_sample = sample_training_data(
        train,
        POINTS_PER_SIMULATION,
        RANDOM_STATE
    )

    print(
        f"Training rows: "
        f"{len(train_sample):,}"
    )

    print(
        f"Training simulations: "
        f"{train_sample['Simulation No'].nunique()}"
    )

    # --------------------------------------------------------
    # Train targets in parallel
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STARTING PARALLEL TARGET TRAINING")
    print("=" * 70)
    print()

    results = []

    with ProcessPoolExecutor(
        max_workers=N_PARALLEL_TARGETS
    ) as executor:

        futures = {
            executor.submit(
                train_target,
                target,
                train_sample,
                validation,
                test,
            ): target
            for target in TARGETS
        }

        for future in as_completed(
            futures
        ):

            target = futures[
                future
            ]

            try:

                result = future.result()

                results.append(
                    result
                )

                print(
                    f"\nCompleted: {target}"
                )

                print(
                    f"  Test R2: "
                    f"{result['test_R2']:.6f}"
                )

                print(
                    f"  Test MAE: "
                    f"{result['test_MAE']:.6g}"
                )

            except Exception as exc:

                print()
                print(
                    f"ERROR training {target}:"
                )

                print(
                    repr(exc)
                )

    # --------------------------------------------------------
    # Check results
    # --------------------------------------------------------

    if not results:

        raise RuntimeError(
            "No target models completed."
        )

    metrics = pd.DataFrame(
        results
    )

    metrics = (
        metrics
        .sort_values("target")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Save combined metrics
    # --------------------------------------------------------

    metrics_path = (
        MODEL_DIR
        / "trajectory_all_metrics.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ALL TARGET RESULTS")
    print("=" * 70)

    display_columns = [
        "target",
        "best_iteration",
        "test_MAE",
        "test_RMSE",
        "test_R2",
        "test_MdAPE",
    ]

    print(
        metrics[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Completed targets: "
        f"{len(results)}/{len(TARGETS)}"
    )

    print()
    print(
        f"Metrics saved:"
    )

    print(
        metrics_path
    )

    print()
    print("=" * 70)
    print("TASK 3 TRAINING COMPLETE")
    print("=" * 70)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
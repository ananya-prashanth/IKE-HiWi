"""
trajectory_preprocess.py

Create the ML dataset for Task 3 using balance trajectories.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from config import PROJECT_ROOT, PROCESSED_DIR, METADATA_FILE


STATIC_COLUMNS = [
    "Psys","Ffuel","Porosity","Dparticle","Alpha",
    "Rflat","Tbed","Decay","Fstruct","Rcake",
    "Topcake","Botcake","Porcake","Partdiacake","Hdebris"
]

TARGET_COLUMNS = [
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

TIME_FEATURES = [
    "time",
    "t_norm",
    "log1p_t_norm",
    "sqrt_t_norm",
    "t_norm_squared",
    "log1p_maximum_time"
]

metadata = pd.read_csv(METADATA_FILE)
metadata["Simulation No"] = metadata["Simulation No"].astype(int)
metadata = metadata[["Simulation No"] + STATIC_COLUMNS + ["maximum_time"]]


def build_dataset(split):

    print(f"\nLoading balance_{split}.parquet")

    balance = pd.read_parquet(
        PROCESSED_DIR / f"balance_{split}.parquet"
    )

    print(balance.shape)

    # Merge static parameters
    df = balance.merge(
        metadata,
        on="Simulation No",
        how="left"
    )

    # Time features
    df["t_norm"] = df["time"] / df["maximum_time"]
    df["log1p_t_norm"] = np.log1p(df["t_norm"])
    df["sqrt_t_norm"] = np.sqrt(df["t_norm"])
    df["t_norm_squared"] = df["t_norm"] ** 2
    df["log1p_maximum_time"] = np.log1p(df["maximum_time"])

    feature_columns = (
        ["Simulation No"]
        + STATIC_COLUMNS
        + TIME_FEATURES
    )

    final_columns = (
        feature_columns
        + TARGET_COLUMNS
    )

    df = df[final_columns]

    output = PROCESSED_DIR / "trajectory"/ f"trajectory_{split}.parquet"

    df.to_parquet(output, index=False)

    print("Saved:", output)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    return df


for split in ["train", "validation", "test"]:
    build_dataset(split)

print("\nTrajectory preprocessing complete.")
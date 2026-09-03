"""
convert_balance_to_parquet.py

One-time conversion of balance Excel files to Parquet.

The conversion is parallelized because the raw balance files are
independent of one another.

Already-converted individual Parquet files are reused, so an
interrupted conversion can be resumed without repeating work.

After all individual files have been converted, three consolidated
Parquet files are created:

    balance_train.parquet
    balance_validation.parquet
    balance_test.parquet

Only the balance-file variables required for Task 3 are retained.
Particle temperature files are not used.
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd


# ============================================================
# Paths
# ============================================================

from config import PROJECT_ROOT, PROCESSED_DIR, SPLIT_FILE, SIM7200_DIR
PROJECT_ROOT = Path("/home/ikeaprash/project")

INDIVIDUAL_DIR = (
    PROCESSED_DIR
    / "balance_parquet"
)



# ============================================================
# Settings
# ============================================================

N_WORKERS = 48


# ============================================================
# Balance variables
# ============================================================

BALANCE_COLUMNS = [
    "time",
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
# Worker function
# ============================================================

def convert_one(simid):
    """
    Convert one balance Excel file to Parquet.

    If the Parquet file already exists, it is reused.
    """

    output_path = (
        INDIVIDUAL_DIR
        / f"{simid:05d}.parquet"
    )

    # --------------------------------------------------------
    # Reuse existing conversion
    # --------------------------------------------------------

    if output_path.exists():

        return simid, "existing"

    # --------------------------------------------------------
    # Locate Excel file
    # --------------------------------------------------------

    excel_path = (
        SIM7200_DIR
        / f"{simid:05d}_balance_readable.xlsx"
    )

    if not excel_path.exists():

        return simid, "missing"

    # --------------------------------------------------------
    # Read only required columns
    # --------------------------------------------------------

    df = pd.read_excel(
        excel_path,
        usecols=BALANCE_COLUMNS
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    missing = [
        column
        for column in BALANCE_COLUMNS
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Simulation {simid}: "
            f"missing columns {missing}"
        )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for column in BALANCE_COLUMNS:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Sort by actual time
    # --------------------------------------------------------

    df = (
        df
        .dropna(subset=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Add simulation ID
    # --------------------------------------------------------

    df.insert(
        0,
        "Simulation No",
        simid
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_parquet(
        output_path,
        index=False
    )

    return simid, "converted"


# ============================================================
# Main
# ============================================================

def main():

    INDIVIDUAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load simulation split
    # --------------------------------------------------------

    print("Loading simulation splits...")

    splits = pd.read_csv(
        SPLIT_FILE
    )

    splits["Simulation No"] = (
        splits["Simulation No"]
        .astype(int)
    )

    print()
    print(
        splits["split"].value_counts()
    )

    # --------------------------------------------------------
    # Get all simulation IDs
    # --------------------------------------------------------

    sim_ids = sorted(
        splits["Simulation No"]
        .unique()
    )

    print()
    print(
        f"Total simulations: {len(sim_ids)}"
    )

    print(
        f"Workers: {N_WORKERS}"
    )

    # --------------------------------------------------------
    # Parallel conversion
    # --------------------------------------------------------

    counts = {
        "existing": 0,
        "converted": 0,
        "missing": 0,
        "error": 0,
    }

    print()
    print("Starting conversion...")
    print()

    with ProcessPoolExecutor(
        max_workers=N_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                convert_one,
                simid
            ): simid
            for simid in sim_ids
        }

        completed = 0

        for future in as_completed(futures):

            simid = futures[future]

            try:

                _, status = future.result()

                counts[status] += 1

            except Exception as exc:

                counts["error"] += 1

                print(
                    f"\nERROR simulation {simid}: "
                    f"{exc}"
                )

            completed += 1

            if completed % 100 == 0:

                print(
                    f"Completed "
                    f"{completed}/{len(sim_ids)} "
                    f"| converted: {counts['converted']} "
                    f"| existing: {counts['existing']}"
                )

    # --------------------------------------------------------
    # Conversion summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)

    print(
        f"Converted: {counts['converted']}"
    )

    print(
        f"Already existed: {counts['existing']}"
    )

    print(
        f"Missing: {counts['missing']}"
    )

    print(
        f"Errors: {counts['error']}"
    )

    if counts["error"] > 0:

        raise RuntimeError(
            "Some simulations failed during conversion."
        )

    # --------------------------------------------------------
    # Create split lookup
    # --------------------------------------------------------

    split_lookup = (
        splits
        .set_index("Simulation No")["split"]
        .to_dict()
    )

    # --------------------------------------------------------
    # Consolidate each split
    # --------------------------------------------------------

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        print()
        print("=" * 60)
        print(
            f"Creating {split_name} dataset"
        )
        print("=" * 60)

        split_sim_ids = [
            simid
            for simid in sim_ids
            if split_lookup[simid] == split_name
        ]

        frames = []

        for i, simid in enumerate(
            split_sim_ids,
            start=1
        ):

            path = (
                INDIVIDUAL_DIR
                / f"{simid:05d}.parquet"
            )

            if not path.exists():

                raise FileNotFoundError(
                    f"Missing converted file: {path}"
                )

            frames.append(
                pd.read_parquet(path)
            )

            if i % 250 == 0:

                print(
                    f"Loaded "
                    f"{i}/{len(split_sim_ids)}"
                )

        combined = pd.concat(
            frames,
            ignore_index=True
        )

        output_path = (
            PROCESSED_DIR
            / f"balance_{split_name}.parquet"
        )

        combined.to_parquet(
            output_path,
            index=False
        )

        print()
        print(
            f"Saved: {output_path}"
        )

        print(
            f"Rows: {len(combined):,}"
        )

        print(
            f"Simulations: "
            f"{combined['Simulation No'].nunique()}"
        )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()
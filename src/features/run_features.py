"""
Feature pipeline runner.

Loads cleaned data, engineers features, selects the best subset,
and saves the result ready for model training.

Usage:
    python src/features/run_features.py
"""

import time
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def main() -> None:
    t0 = time.time()

    # ------------------------------------------------------------------ #
    # Load                                                                 #
    # ------------------------------------------------------------------ #
    input_path = DATA_DIR / "cleaned.csv"
    print(f"Loading {input_path} ...")
    df = pd.read_csv(input_path)
    print(f"  Raw shape : {df.shape[0]:,} rows x {df.shape[1]} columns")

    t_load = time.time()
    print(f"  Load time : {t_load - t0:.2f}s")

    # ------------------------------------------------------------------ #
    # Feature engineering                                                  #
    # ------------------------------------------------------------------ #
    from features.engineer import create_features, select_features

    print("\nEngineering features ...")
    featured = create_features(df)
    t_eng = time.time()
    print(f"  Engineered shape : {featured.shape[0]:,} rows x {featured.shape[1]} columns")
    print(f"  Engineering time : {t_eng - t_load:.2f}s")

    # ------------------------------------------------------------------ #
    # Feature selection                                                    #
    # ------------------------------------------------------------------ #
    print("\nSelecting features ...")
    selected_names, final_df = select_features(featured)
    t_sel = time.time()
    print(f"  Selected shape   : {final_df.shape[0]:,} rows x {final_df.shape[1]} columns")
    print(f"  Selection time   : {t_sel - t_eng:.2f}s")

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*55}")
    print(f"  Before : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  After  : {final_df.shape[0]:,} rows x {final_df.shape[1]} columns")
    print(f"{'='*55}")
    print(f"\nKept numeric features ({len(selected_names)}):")
    for name in selected_names:
        print(f"  {name}")

    # ------------------------------------------------------------------ #
    # Save                                                                 #
    # ------------------------------------------------------------------ #
    output_path = DATA_DIR / "features.csv"
    final_df.to_csv(output_path, index=False)
    t_save = time.time()
    print(f"\nSaved to: {output_path}")

    print(f"\nTotal elapsed : {t_save - t0:.2f}s")


if __name__ == "__main__":
    main()

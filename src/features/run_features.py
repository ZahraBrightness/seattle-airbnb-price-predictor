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
    from features.engineer import (
        create_features, create_nlp_features, handle_nulls, select_features,
    )
    from features.calendar_features import create_calendar_features
    from features.review_features import create_review_features

    # Circular features derived from price — excluded before model training.
    CIRCULAR_COLS = ["peak_demand_score", "estimated_annual_revenue"]

    print("\nEngineering features ...")
    featured = create_features(df)
    t_eng = time.time()
    print(f"  Engineered shape : {featured.shape[0]:,} rows x {featured.shape[1]} columns")
    print(f"  Engineering time : {t_eng - t_load:.2f}s")

    # ------------------------------------------------------------------ #
    # NLP features                                                         #
    # ------------------------------------------------------------------ #
    print("\nExtracting NLP features ...")
    featured = create_nlp_features(featured)
    t_nlp = time.time()
    print(f"  Post-NLP shape   : {featured.shape[0]:,} rows x {featured.shape[1]} columns")
    print(f"  NLP time         : {t_nlp - t_eng:.2f}s")

    # ------------------------------------------------------------------ #
    # Calendar features                                                    #
    # ------------------------------------------------------------------ #
    print("\nEngineering calendar features ...")
    featured = create_calendar_features(featured)
    # Drop circular features before null handling and selection
    cols_to_drop = [c for c in CIRCULAR_COLS if c in featured.columns]
    if cols_to_drop:
        featured = featured.drop(columns=cols_to_drop)
        print(f"  Dropped circular columns: {cols_to_drop}")
    t_cal = time.time()
    print(f"  Post-calendar shape : {featured.shape[0]:,} rows x {featured.shape[1]} columns")
    print(f"  Calendar time       : {t_cal - t_nlp:.2f}s")

    # ------------------------------------------------------------------ #
    # Review sentiment features                                            #
    # ------------------------------------------------------------------ #
    print("\nEngineering review sentiment features ...")
    featured = create_review_features(featured)
    t_rev = time.time()
    print(f"  Post-review shape   : {featured.shape[0]:,} rows x {featured.shape[1]} columns")
    print(f"  Review time         : {t_rev - t_cal:.2f}s")

    # ------------------------------------------------------------------ #
    # Null handling                                                        #
    # ------------------------------------------------------------------ #
    print("\nHandling nulls ...")
    featured = handle_nulls(featured)
    t_nulls = time.time()
    print(f"  Post-null shape  : {featured.shape[0]:,} rows x {featured.shape[1]} columns")
    print(f"  Null handle time : {t_nulls - t_rev:.2f}s")

    # ------------------------------------------------------------------ #
    # Feature selection                                                    #
    # ------------------------------------------------------------------ #
    print("\nSelecting features ...")
    selected_names, final_df = select_features(featured)
    t_sel = time.time()
    print(f"  Selected shape   : {final_df.shape[0]:,} rows x {final_df.shape[1]} columns")
    print(f"  Selection time   : {t_sel - t_nulls:.2f}s")
    print(f"  Pipeline so far  : {t_sel - t0:.2f}s")

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

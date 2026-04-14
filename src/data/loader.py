import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_csv(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    df = pd.read_csv(path)
    return df


def print_shape(df: pd.DataFrame) -> None:
    rows, cols = df.shape
    print(f"Shape: {rows:,} rows x {cols} columns")


def print_schema(df: pd.DataFrame) -> None:
    print("\nColumns and dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"  {col:<30} {dtype}")


def print_summary_stats(df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        print("\nNo numeric columns found.")
        return
    stats = numeric.agg(["mean", "std", "min", "max"])
    print("\nSummary statistics (numeric columns):")
    print(stats.to_string())


def print_missing(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("\nNo missing values.")
        return
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"missing": missing, "pct": pct})
    print("\nMissing value counts:")
    print(report.to_string())


def inspect(filename: str) -> pd.DataFrame:
    df = load_csv(filename)
    print_shape(df)
    print_schema(df)
    print_summary_stats(df)
    print_missing(df)
    return df


if __name__ == "__main__":
    import sys

    filename = sys.argv[1] if len(sys.argv) > 1 else None
    if filename is None:
        csv_files = list(DATA_DIR.glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in {DATA_DIR}")
            sys.exit(1)
        filename = csv_files[0].name
        print(f"No filename given — using: {filename}\n")

    inspect(filename)

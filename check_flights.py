from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/flights/processed")
files = sorted(DATA_DIR.glob("*.parquet"))

print(f"Parquet files found: {len(files)}")

if not files:
    raise SystemExit("No processed Parquet files found.")

frames = [pd.read_parquet(file) for file in files]
flights = pd.concat(frames, ignore_index=True)

print(f"\nTotal flights: {len(flights):,}")
print(f"Date range: {flights['FlightDate'].min()} to {flights['FlightDate'].max()}")

print("\nColumns:")
print(flights.columns.tolist())

print("\nMissing values:")
print(flights[["Origin", "Dest", "DepDelay", "ArrDelay"]].isna().sum())

print("\nSample records:")
print(flights.head())

output = DATA_DIR / "flights_2024_2025.parquet"
flights.to_parquet(output, index=False)

print(f"\nConsolidated dataset saved to: {output}")

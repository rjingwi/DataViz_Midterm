import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# Paths
AIRPORT_FILE = Path("data/airports/airports.csv")
FLIGHT_DIR = Path("data/flights/processed")
OUTPUT_DIR = Path("data/flights/processed")

# 1. Clean airport data
airports = pd.read_csv(AIRPORT_FILE, low_memory=False)

airports = airports[
    (airports["iso_country"] == "US") &
    (airports["iata_code"].notna()) &
    (airports["latitude_deg"].notna()) &
    (airports["longitude_deg"].notna())
].copy()

airports = airports[
    airports["scheduled_service"].astype(str).str.lower() == "yes"
]

airports = airports[
    ["iata_code", "name", "municipality", "latitude_deg", "longitude_deg"]
]

airports = airports.drop_duplicates(subset="iata_code")

airports.to_csv("data/airports/us_airports.csv", index=False)

airport_codes = set(airports["iata_code"].astype(str))

print(f"U.S. airports retained: {len(airports):,}")

# 2. Filter flight records
output_file = OUTPUT_DIR / "domestic_flights_2024_2025.parquet"

if output_file.exists():
    output_file.unlink()

writer = None
total_flights = 0

for file in sorted(FLIGHT_DIR.glob("*.parquet")):
    if file.name in {
        "flights_2024_2025.parquet",
        "domestic_flights_2024_2025.parquet"
    }:
        continue

    print(f"Filtering {file.name}...")

    parquet_file = pq.ParquetFile(file)

    for batch in parquet_file.iter_batches(batch_size=250_000):
        df = batch.to_pandas()

        df = df[
            df["Origin"].isin(airport_codes) &
            df["Dest"].isin(airport_codes)
        ]

        if df.empty:
            continue

        table = pa.Table.from_pandas(df, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(
                output_file,
                table.schema,
                compression="snappy"
            )

        writer.write_table(table)
        total_flights += len(df)

    print(f"Completed: {file.name}")

if writer:
    writer.close()

print("\nDOMESTIC FLIGHT FILTERING COMPLETE")
print(f"Domestic flights retained: {total_flights:,}")
print(f"Airport lookup: data/airports/us_airports.csv")
print(f"Flight data: {output_file}")

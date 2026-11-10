import pandas as pd
import zipfile
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

INPUT_DIR = Path("data/flights/raw")
OUTPUT_DIR = Path("data/flights/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "FlightDate",
    "Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "DepTime",
    "DepDelay",
    "CRSArrTime",
    "ArrTime",
    "ArrDelay",
    "Cancelled",
    "CancellationCode",
    "Diverted",
    "Distance",
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay"
]

STRING_COLUMNS = [
    "Reporting_Airline", "Origin", "Dest", "CancellationCode"
]

NUMERIC_COLUMNS = [
    col for col in COLUMNS
    if col not in STRING_COLUMNS + ["FlightDate"]
]

zip_files = sorted(INPUT_DIR.glob("*.zip"))

print(f"Found {len(zip_files)} BTS files.")

for zip_path in zip_files:

    output_path = OUTPUT_DIR / f"{zip_path.stem}.parquet"

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Skipping: {zip_path.name}")
        continue

    writer = None
    rows_processed = 0

    try:
        print(f"\nProcessing: {zip_path.name}")

        with zipfile.ZipFile(zip_path) as archive:

            # Select the actual CSV, ignoring readme.html
            csv_files = [
                name for name in archive.namelist()
                if name.lower().endswith(".csv")
            ]

            if not csv_files:
                print("No CSV found. Skipping.")
                continue

            csv_file = csv_files[0]

            with archive.open(csv_file) as file:

                header = pd.read_csv(file, nrows=0)
                available = [
                    col for col in COLUMNS
                    if col in header.columns
                ]

            required = [
                "FlightDate",
                "Reporting_Airline",
                "Origin",
                "Dest"
            ]

            if not all(col in available for col in required):
                print("Missing required columns. Skipping.")
                continue

            with archive.open(csv_file) as file:

                reader = pd.read_csv(
                    file,
                    usecols=available,
                    chunksize=200_000,
                    low_memory=False
                )

                for chunk in reader:

                    for col in COLUMNS:
                        if col not in chunk.columns:
                            chunk[col] = pd.NA

                    chunk = chunk[COLUMNS]

                    chunk["FlightDate"] = pd.to_datetime(
                        chunk["FlightDate"],
                        errors="coerce"
                    )

                    for col in STRING_COLUMNS:
                        chunk[col] = chunk[col].astype("string")

                    for col in NUMERIC_COLUMNS:
                        chunk[col] = pd.to_numeric(
                            chunk[col],
                            errors="coerce"
                        ).astype("float64")

                    chunk["Year"] = chunk["FlightDate"].dt.year.astype("Int64")
                    chunk["Month"] = chunk["FlightDate"].dt.month.astype("Int64")
                    chunk["DayOfWeek"] = (
                        chunk["FlightDate"].dt.dayofweek.astype("Int64") + 1
                    )

                    table = pa.Table.from_pandas(
                        chunk,
                        preserve_index=False
                    )

                    if writer is None:
                        writer = pq.ParquetWriter(
                            output_path,
                            table.schema,
                            compression="snappy"
                        )

                    writer.write_table(table)
                    rows_processed += len(chunk)

        if writer:
            writer.close()
            writer = None

        print(f"SUCCESS: {rows_processed:,} flights")

    except Exception as error:

        if writer:
            writer.close()

        if output_path.exists():
            output_path.unlink()

        print(f"ERROR: {error}")

print("\nPROCESSING FINISHED!")

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

INPUT = Path("data/flights/processed/domestic_flights_2024_2025.parquet")
OUTPUT = Path("data/dashboard")

BATCH_SIZE = 250_000

DELAY_COLUMNS = [
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
]

REQUIRED_COLUMNS = [
    "FlightDate",
    "Reporting_Airline",
    "Origin",
    "Dest",
    "ArrDelay",
    "Cancelled",
] + DELAY_COLUMNS


def combine_partials(parts, keys, value_columns):
    if not parts:
        return pd.DataFrame(columns=keys + value_columns)

    combined = pd.concat(parts, ignore_index=True)

    return combined.groupby(
        keys, as_index=False
    )[value_columns].sum()


def main():

    OUTPUT.mkdir(parents=True, exist_ok=True)

    if not INPUT.exists():
        raise FileNotFoundError(f"Missing source dataset: {INPUT}")

    parquet = pq.ParquetFile(INPUT)

    available = set(parquet.schema.names)
    missing = set(REQUIRED_COLUMNS).difference(available)

    if missing:
        raise ValueError(
            f"Source Parquet is missing columns: {sorted(missing)}"
        )

    route_parts = []
    monthly_parts = []
    airline_parts = []
    cause_parts = []

    print(
        f"Processing {parquet.metadata.num_rows:,} flight records..."
    )

    for batch_number, batch in enumerate(
        parquet.iter_batches(
            batch_size=BATCH_SIZE,
            columns=REQUIRED_COLUMNS
        ),
        start=1,
    ):

        df = batch.to_pandas()

        df["FlightDate"] = pd.to_datetime(
            df["FlightDate"], errors="coerce"
        )

        df = df.dropna(
            subset=["FlightDate", "Origin", "Dest"]
        )

        if df.empty:
            continue

        df["Year"] = df["FlightDate"].dt.year.astype("int16")
        df["Month"] = df["FlightDate"].dt.month.astype("int8")

        df["Cancelled"] = pd.to_numeric(
            df["Cancelled"], errors="coerce"
        ).fillna(0)

        df["ArrDelay"] = pd.to_numeric(
            df["ArrDelay"], errors="coerce"
        )

        for col in DELAY_COLUMNS:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            ).fillna(0)

        # Route counts
        route_parts.append(
            df.groupby(
                ["Year", "Origin", "Dest"]
            ).size().reset_index(name="Flights")
        )

        # Arrival-delay helper columns
        valid_arr = df["ArrDelay"].notna()

        df["ValidArrivals"] = valid_arr.astype("int32")
        df["DelaySum"] = df["ArrDelay"].fillna(0)

        df["Late15"] = (
            df["ArrDelay"] >= 15
        ).fillna(False).astype("int32")

        # Monthly flight performance
        monthly_parts.append(
            df.groupby(
                ["Year", "Month"], as_index=False
            ).agg(
                Flights=("Origin", "size"),
                Cancelled=("Cancelled", "sum"),
                ValidArrivals=("ValidArrivals", "sum"),
                DelaySum=("DelaySum", "sum"),
                Late15=("Late15", "sum"),
            )
        )

        # Airline performance
        airline_parts.append(
            df.groupby(
                ["Year", "Reporting_Airline"],
                as_index=False
            ).agg(
                Flights=("Origin", "size"),
                Cancelled=("Cancelled", "sum"),
                ValidArrivals=("ValidArrivals", "sum"),
                DelaySum=("DelaySum", "sum"),
            )
        )

        # Delay causes
        cause_parts.append(
            df.groupby("Year", as_index=False)[
                DELAY_COLUMNS
            ].sum()
        )

        if batch_number % 10 == 0:
            print(f"Processed {batch_number} batches...")

    # Combine partial aggregations
    routes = combine_partials(
        route_parts,
        ["Year", "Origin", "Dest"],
        ["Flights"],
    )

    monthly = combine_partials(
        monthly_parts,
        ["Year", "Month"],
        [
            "Flights",
            "Cancelled",
            "ValidArrivals",
            "DelaySum",
            "Late15",
        ],
    )

    airlines = combine_partials(
        airline_parts,
        ["Year", "Reporting_Airline"],
        [
            "Flights",
            "Cancelled",
            "ValidArrivals",
            "DelaySum",
        ],
    )

    causes = combine_partials(
        cause_parts,
        ["Year"],
        DELAY_COLUMNS,
    )

    # Calculate monthly performance metrics
    monthly["AverageArrivalDelay"] = (
        monthly["DelaySum"]
        / monthly["ValidArrivals"].replace(0, float("nan"))
    )

    monthly["Late15Rate"] = (
        monthly["Late15"]
        / monthly["ValidArrivals"].replace(0, float("nan"))
        * 100
    )

    monthly["CancellationRate"] = (
        monthly["Cancelled"]
        / monthly["Flights"].replace(0, float("nan"))
        * 100
    )

    # Calculate airline performance metrics
    airlines["AverageArrivalDelay"] = (
        airlines["DelaySum"]
        / airlines["ValidArrivals"].replace(0, float("nan"))
    )

    airlines["CancellationRate"] = (
        airlines["Cancelled"]
        / airlines["Flights"].replace(0, float("nan"))
        * 100
    )

    # Convert delay causes into a chart-friendly format
    cause_names = {
        "CarrierDelay": "Carrier",
        "WeatherDelay": "Weather",
        "NASDelay": "National aviation system",
        "SecurityDelay": "Security",
        "LateAircraftDelay": "Late-arriving aircraft",
    }

    causes = causes.melt(
        id_vars="Year",
        var_name="Cause",
        value_name="DelayMinutes",
    )

    causes["Cause"] = causes["Cause"].map(cause_names)

    # Save dashboard datasets
    routes.to_csv(
        OUTPUT / "routes_by_year.csv", index=False
    )

    monthly.to_csv(
        OUTPUT / "monthly_performance.csv", index=False
    )

    airlines.to_csv(
        OUTPUT / "airline_performance.csv", index=False
    )

    causes.to_csv(
        OUTPUT / "delay_causes.csv", index=False
    )

    print("\nDashboard datasets created successfully!")

    for file in sorted(OUTPUT.glob("*.csv")):
        print(
            f"{file.name}: "
            f"{file.stat().st_size / 1024:.1f} KB"
        )

    print(f"\nRoutes: {len(routes):,}")
    print(f"Monthly records: {len(monthly):,}")
    print(f"Airline records: {len(airlines):,}")


if __name__ == "__main__":
    main()

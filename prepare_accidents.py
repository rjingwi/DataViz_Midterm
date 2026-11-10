import subprocess
import pandas as pd
import io
import re
from pathlib import Path

DB = "data/accidents/raw/avall.mdb"
OUTPUT = Path("data/accidents/processed")
OUTPUT.mkdir(parents=True, exist_ok=True)


def export_table(table):
    result = subprocess.run(
        ["mdb-export", DB, table],
        capture_output=True,
        text=True,
        check=True
    )
    return pd.read_csv(io.StringIO(result.stdout))


def clean_text(value):
    if pd.isna(value):
        return ""

    value = str(value).strip()

    if value.lower() in {"import", "nan", "none"}:
        return ""

    value = re.sub(r"\s+", " ", value)
    return value


print("Reading NTSB tables...")

narratives = export_table("narratives")
events = export_table("events")

# Keep the relevant narrative fields
narratives = narratives[
    ["ev_id", "narr_accp", "narr_accf", "narr_cause"]
].copy()

for column in ["narr_accp", "narr_accf", "narr_cause"]:
    narratives[column] = narratives[column].apply(clean_text)

# Combine narrative text for each accident.
# Multiple aircraft records can share the same event ID.
def combine_unique(series):
    values = list(dict.fromkeys(
        value for value in series if value
    ))
    return " ".join(values)


narratives = (
    narratives.groupby("ev_id")
    .agg({
        "narr_accp": combine_unique,
        "narr_accf": combine_unique,
        "narr_cause": combine_unique
    })
    .reset_index()
)

narratives["Narrative"] = (
    narratives["narr_accp"] + " " +
    narratives["narr_accf"]
).str.strip()

narratives["Narrative"] = narratives["Narrative"].str.replace(
    r"\s+", " ", regex=True
)

narratives["Probable_Cause"] = narratives["narr_cause"]

# Select accident details
event_columns = [
    "ev_id",
    "ev_date",
    "ev_year",
    "ev_type",
    "ev_city",
    "ev_state",
    "ev_country",
    "dec_latitude",
    "dec_longitude",
    "ev_highest_injury",
    "inj_tot_f",
    "inj_tot_m",
    "inj_tot_s",
    "inj_tot_t",
    "invest_agy"
]

events = events[event_columns].copy()

# Join accident details to narrative text
accidents = events.merge(
    narratives,
    on="ev_id",
    how="inner"
)

# Keep records with meaningful narrative text
accidents = accidents[
    accidents["Narrative"].str.len() >= 80
].copy()

# Convert dates and numeric fields
accidents["ev_date"] = pd.to_datetime(
    accidents["ev_date"],
    errors="coerce"
)

accidents["ev_year"] = pd.to_numeric(
    accidents["ev_year"],
    errors="coerce"
)

accidents["dec_latitude"] = pd.to_numeric(
    accidents["dec_latitude"],
    errors="coerce"
)

accidents["dec_longitude"] = pd.to_numeric(
    accidents["dec_longitude"],
    errors="coerce"
)

output_file = OUTPUT / "accidents_text.parquet"
accidents.to_parquet(output_file, index=False)

print("\nAccident text dataset created.")
print(f"Records: {len(accidents):,}")
print(f"Years: {accidents['ev_year'].min():.0f}–{accidents['ev_year'].max():.0f}")
print(f"Records with probable cause: {accidents['Probable_Cause'].ne('').sum():,}")
print(f"Records with location coordinates: {accidents['dec_latitude'].notna().sum():,}")
print(f"Saved to: {output_file}")

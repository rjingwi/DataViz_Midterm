import requests
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
YEARS = [2024, 2025]
BASE_URL = "https://transtats.bts.gov/PREZIP/"

OUTPUT_DIR = Path("data/flights/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Retry configuration
session = requests.Session()

retries = Retry(
    total=3,
    backoff_factor=2,
    status_forcelist=[500, 502, 503, 504]
)

session.mount("https://", HTTPAdapter(max_retries=retries))

# Download files
for year in YEARS:
    for month in range(1, 13):

        filename = (
            f"On_Time_Reporting_Carrier_On_Time_Performance_"
            f"1987_present_{year}_{month}.zip"
        )

        url = BASE_URL + filename
        filepath = OUTPUT_DIR / filename

        # Skip existing files
        if filepath.exists() and filepath.stat().st_size > 0:
            print(f"Already downloaded: {filename}")
            continue

        print(f"Downloading {year}-{month:02d}...")

        try:
            response = session.get(url, stream=True, timeout=120)
            response.raise_for_status()

            with open(filepath, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)

            print(f"Successfully downloaded: {filename}")

        except requests.RequestException as error:
            print(f"FAILED: {filename}")
            print(error)

            if filepath.exists():
                filepath.unlink()

print("\nBTS DOWNLOAD PROCESS COMPLETE!")

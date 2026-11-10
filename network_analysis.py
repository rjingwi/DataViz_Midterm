import pandas as pd
import networkx as nx
from pathlib import Path

FLIGHT_FILE = Path("data/flights/processed/domestic_flights_2024_2025.parquet")
OUTPUT_DIR = Path("data/flights/processed")

flights = pd.read_parquet(FLIGHT_FILE)

# Aggregate flights into airport routes
routes = (
    flights.groupby(["Origin", "Dest"])
    .size()
    .reset_index(name="Flights")
)

# Build directed network
G = nx.DiGraph()

for _, row in routes.iterrows():
    G.add_edge(
        row["Origin"],
        row["Dest"],
        weight=row["Flights"]
    )

# Calculate airport metrics
airport_metrics = pd.DataFrame({
    "Airport": list(G.nodes()),
    "Connections": [G.degree(n) for n in G.nodes()],
    "Total_Flights": [
        G.in_degree(n, weight="weight") +
        G.out_degree(n, weight="weight")
        for n in G.nodes()
    ]
})

airport_metrics = airport_metrics.sort_values(
    "Total_Flights",
    ascending=False
)

# Attach airport names and locations
airports = pd.read_csv("data/airports/us_airports.csv")

airport_metrics = airport_metrics.merge(
    airports,
    left_on="Airport",
    right_on="iata_code",
    how="left"
)

# Save outputs
routes.to_csv(OUTPUT_DIR / "routes.csv", index=False)
airport_metrics.to_csv(OUTPUT_DIR / "airport_metrics.csv", index=False)

print("\nTOP 15 AIRPORTS BY FLIGHT VOLUME")
print(
    airport_metrics[
        ["Airport", "name", "Total_Flights", "Connections"]
    ].head(15).to_string(index=False)
)

print("\nTOP 15 ROUTES")
print(
    routes.sort_values("Flights", ascending=False)
    .head(15)
    .to_string(index=False)
)

print(f"\nNetwork airports: {G.number_of_nodes():,}")
print(f"Directed routes: {G.number_of_edges():,}")

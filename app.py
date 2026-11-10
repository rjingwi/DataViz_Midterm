from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="U.S. Aviation Network & Performance Explorer",
    page_icon="✈️",
    layout="wide",
)

st.title("U.S. Aviation Network & Performance Explorer")

st.caption(
    "Exploring domestic flight connectivity, geographic patterns, "
    "operational performance, and aviation safety."
)


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

DATA = Path("data/dashboard")

ROUTES_PATH = DATA / "routes_by_year.csv"
MONTHLY_PATH = DATA / "monthly_performance.csv"
AIRLINE_PATH = DATA / "airline_performance.csv"
CAUSES_PATH = DATA / "delay_causes.csv"

AIRPORTS_PATH = Path("data/airports/us_airports.csv")
ACCIDENTS_PATH = Path(
    "data/accidents/processed/accidents_text.parquet"
)


# ---------------------------------------------------------
# Airline names
# ---------------------------------------------------------

AIRLINE_NAMES = {
    "9E": "Endeavor Air",
    "AA": "American Airlines",
    "AS": "Alaska Airlines",
    "B6": "JetBlue Airways",
    "DL": "Delta Air Lines",
    "F9": "Frontier Airlines",
    "G4": "Allegiant Air",
    "HA": "Hawaiian Airlines",
    "MQ": "Envoy Air",
    "NK": "Spirit Airlines",
    "OH": "PSA Airlines",
    "OO": "SkyWest Airlines",
    "QX": "Horizon Air",
    "UA": "United Airlines",
    "WN": "Southwest Airlines",
    "YV": "Mesa Air",
    "YX": "Republic Airways",
    "ZW": "Air Wisconsin",
}


# ---------------------------------------------------------
# Load dashboard datasets
# ---------------------------------------------------------

@st.cache_data
def load_dashboard_data():

    routes = pd.read_csv(ROUTES_PATH)
    monthly = pd.read_csv(MONTHLY_PATH)
    airlines = pd.read_csv(AIRLINE_PATH)
    causes = pd.read_csv(CAUSES_PATH)
    airports = pd.read_csv(AIRPORTS_PATH)

    airports["iata_code"] = (
        airports["iata_code"].astype(str).str.upper()
    )

    airports = airports.drop_duplicates("iata_code")

    return routes, monthly, airlines, causes, airports


@st.cache_data
def load_accidents():

    df = pd.read_parquet(ACCIDENTS_PATH)

    df["ev_year"] = pd.to_numeric(
        df["ev_year"], errors="coerce"
    )

    df["dec_latitude"] = pd.to_numeric(
        df["dec_latitude"], errors="coerce"
    )

    df["dec_longitude"] = pd.to_numeric(
        df["dec_longitude"], errors="coerce"
    )

    df["inj_tot_f"] = pd.to_numeric(
        df["inj_tot_f"], errors="coerce"
    )

    df["ev_date"] = pd.to_datetime(
        df["ev_date"], errors="coerce"
    )

    for col in ["Narrative", "Probable_Cause"]:
        df[col] = df[col].fillna("").astype(str)

    return df


# ---------------------------------------------------------
# Check required files
# ---------------------------------------------------------

required_files = [
    ROUTES_PATH,
    MONTHLY_PATH,
    AIRLINE_PATH,
    CAUSES_PATH,
    AIRPORTS_PATH,
]

missing_files = [
    str(path) for path in required_files
    if not path.exists()
]

if missing_files:

    st.error("Some dashboard datasets are missing:")

    for file in missing_files:
        st.write(f"- `{file}`")

    st.info(
        "Run `python prepare_dashboard_data.py` "
        "from your project directory."
    )

    st.stop()


try:

    routes, monthly, airlines, causes, airports = (
        load_dashboard_data()
    )

except Exception as exc:

    st.error(f"Could not load dashboard data: {exc}")
    st.stop()


# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------

st.sidebar.header("Explore the data")

available_years = sorted(
    routes["Year"].dropna().astype(int).unique().tolist()
)

selected_years = st.sidebar.multiselect(
    "Flight year",
    options=available_years,
    default=available_years,
)

if not selected_years:
    st.warning("Select at least one year.")
    st.stop()


# Filter aggregated datasets
routes = routes[
    routes["Year"].isin(selected_years)
].copy()

monthly = monthly[
    monthly["Year"].isin(selected_years)
].copy()

airlines = airlines[
    airlines["Year"].isin(selected_years)
].copy()

causes = causes[
    causes["Year"].isin(selected_years)
].copy()


# ---------------------------------------------------------
# Build airport metrics from route summaries
# ---------------------------------------------------------

departures = (
    routes.groupby("Origin")["Flights"]
    .sum()
    .reset_index(name="Departures")
    .rename(columns={"Origin": "Airport"})
)

arrivals = (
    routes.groupby("Dest")["Flights"]
    .sum()
    .reset_index(name="Arrivals")
    .rename(columns={"Dest": "Airport"})
)

connections = (
    routes.groupby("Origin")["Dest"]
    .nunique()
    .reset_index(name="Connections")
    .rename(columns={"Origin": "Airport"})
)

airport_metrics = departures.merge(
    arrivals,
    on="Airport",
    how="outer",
)

airport_metrics = airport_metrics.merge(
    connections,
    on="Airport",
    how="left",
)

airport_metrics[
    ["Departures", "Arrivals", "Connections"]
] = airport_metrics[
    ["Departures", "Arrivals", "Connections"]
].fillna(0)

airport_metrics["Total Traffic"] = (
    airport_metrics["Departures"]
    + airport_metrics["Arrivals"]
)

airport_metrics = airport_metrics.merge(
    airports,
    left_on="Airport",
    right_on="iata_code",
    how="left",
)


# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------

tab_overview, tab_network, tab_geo, tab_performance, tab_safety = (
    st.tabs(
        [
            "Overview",
            "Airport Networks",
            "Geographic Analysis",
            "Flight Performance",
            "Aviation Safety",
        ]
    )
)


# =========================================================
# TAB 1: OVERVIEW
# =========================================================

with tab_overview:

    st.header("Domestic aviation at a glance")

    total_flights = monthly["Flights"].sum()

    total_cancelled = monthly["Cancelled"].sum()

    total_airports = airport_metrics["Airport"].nunique()

    total_routes = len(routes.groupby(["Origin", "Dest"]))

    cancellation_rate = (
        total_cancelled / total_flights * 100
        if total_flights else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Domestic flights",
        f"{total_flights:,.0f}",
    )

    c2.metric(
        "Airports",
        f"{total_airports:,}",
    )

    c3.metric(
        "Directional routes",
        f"{total_routes:,}",
    )

    c4.metric(
        "Cancellation rate",
        f"{cancellation_rate:.2f}%",
    )

    st.subheader("Monthly flight volume")

    monthly["Date"] = pd.to_datetime(
        monthly["Year"].astype(str)
        + "-"
        + monthly["Month"].astype(str)
        + "-01"
    )

    monthly = monthly.sort_values("Date")

    fig = px.line(
        monthly,
        x="Date",
        y="Flights",
        markers=True,
        title="Monthly domestic flight volume",
        labels={
            "Date": "Month",
            "Flights": "Number of flights",
        },
    )

    fig.update_layout(hovermode="x unified")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Busiest airports")

    top_airports = airport_metrics.nlargest(
        10, "Total Traffic"
    ).sort_values("Total Traffic")

    fig = px.bar(
        top_airports,
        x="Total Traffic",
        y="Airport",
        orientation="h",
        title="Top 10 airports by total traffic",
        labels={
            "Total Traffic": "Arrivals + departures",
        },
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# TAB 2: AIRPORT NETWORKS
# =========================================================

with tab_network:

    st.header("Airport network structure")

    st.write(
        "Airports are represented as nodes, while origin-to-destination "
        "pairs represent directed connections."
    )

    left, right = st.columns(2)

    with left:

        st.subheader("Busiest airports")

        top_n = st.slider(
            "Number of airports",
            5,
            25,
            15,
            key="network_top",
        )

        top = airport_metrics.nlargest(
            top_n, "Departures"
        ).sort_values("Departures")

        fig = px.bar(
            top,
            x="Departures",
            y="Airport",
            orientation="h",
            title="Airports by departing flights",
            hover_data=["Arrivals", "Connections"],
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:

        st.subheader("Airport connectivity")

        top = airport_metrics.nlargest(
            top_n, "Connections"
        ).sort_values("Connections")

        fig = px.bar(
            top,
            x="Connections",
            y="Airport",
            orientation="h",
            title="Unique outbound destinations",
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Most frequently traveled routes")

    route_count = st.slider(
        "Routes to display",
        5,
        30,
        15,
        key="route_top",
    )

    top_routes = (
        routes.groupby(["Origin", "Dest"])["Flights"]
        .sum()
        .reset_index()
        .nlargest(route_count, "Flights")
    )

    top_routes["Route"] = (
        top_routes["Origin"]
        + " → "
        + top_routes["Dest"]
    )

    fig = px.bar(
        top_routes.sort_values("Flights"),
        x="Flights",
        y="Route",
        orientation="h",
        title="Top directional routes",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Connections represent unique outbound destination airports, "
        "not total flight volume."
    )


# =========================================================
# TAB 3: GEOGRAPHIC ANALYSIS
# =========================================================

with tab_geo:

    st.header("Geographic patterns")

    st.write(
        "Explore the geographic distribution of airports and "
        "the busiest domestic flight routes."
    )

    mapped = airport_metrics.dropna(
        subset=["latitude_deg", "longitude_deg"]
    )

    map_count = st.slider(
        "Airports to display",
        10,
        100,
        50,
    )

    map_airports = mapped.nlargest(
        map_count, "Departures"
    )

    fig = px.scatter_geo(
        map_airports,
        lat="latitude_deg",
        lon="longitude_deg",
        size="Departures",
        color="Departures",
        hover_name="Airport",
        hover_data=[
            "Departures",
            "Arrivals",
            "Connections",
        ],
        scope="usa",
        projection="albers usa",
        title="Airport locations and departure volume",
        size_max=24,
    )

    fig.update_layout(height=600)

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Busiest routes geographically")

    top_routes = (
        routes.groupby(["Origin", "Dest"])["Flights"]
        .sum()
        .reset_index()
        .nlargest(75, "Flights")
    )

    airport_coords = airports[
        ["iata_code", "latitude_deg", "longitude_deg"]
    ].drop_duplicates("iata_code")

    mapped_routes = top_routes.merge(
        airport_coords.rename(
            columns={
                "iata_code": "Origin",
                "latitude_deg": "origin_lat",
                "longitude_deg": "origin_lon",
            }
        ),
        on="Origin",
        how="left",
    )

    mapped_routes = mapped_routes.merge(
        airport_coords.rename(
            columns={
                "iata_code": "Dest",
                "latitude_deg": "dest_lat",
                "longitude_deg": "dest_lon",
            }
        ),
        on="Dest",
        how="left",
    )

    mapped_routes = mapped_routes.dropna(
        subset=[
            "origin_lat",
            "origin_lon",
            "dest_lat",
            "dest_lon",
        ]
    )

    fig = go.Figure()

    for row in mapped_routes.itertuples(index=False):

        fig.add_trace(
            go.Scattergeo(
                lon=[row.origin_lon, row.dest_lon],
                lat=[row.origin_lat, row.dest_lat],
                mode="lines",
                line={"width": 1},
                opacity=0.35,
                hoverinfo="text",
                text=(
                    f"{row.Origin} → {row.Dest}: "
                    f"{row.Flights:,.0f} flights"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Top 75 directional routes",
        geo={
            "scope": "usa",
            "projection_type": "albers usa",
            "showland": True,
        },
        height=600,
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# TAB 4: FLIGHT PERFORMANCE
# =========================================================

with tab_performance:

    st.header("Flight delays and cancellations")

    total_flights = monthly["Flights"].sum()
    cancelled = monthly["Cancelled"].sum()
    valid_arrivals = monthly["ValidArrivals"].sum()
    delay_sum = monthly["DelaySum"].sum()
    late_flights = monthly["Late15"].sum()

    avg_delay = (
        delay_sum / valid_arrivals
        if valid_arrivals else 0
    )

    late_rate = (
        late_flights / valid_arrivals * 100
        if valid_arrivals else 0
    )

    cancel_rate = (
        cancelled / total_flights * 100
        if total_flights else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Average arrival delay",
        f"{avg_delay:.1f} min",
    )

    c2.metric(
        "Arrivals 15+ min late",
        f"{late_rate:.1f}%",
    )

    c3.metric(
        "Cancelled flights",
        f"{cancelled:,.0f}",
    )

    c4.metric(
        "Cancellation rate",
        f"{cancel_rate:.2f}%",
    )

    st.subheader("Monthly average arrival delay")

    monthly_chart = monthly.copy()

    monthly_chart["Date"] = pd.to_datetime(
        monthly_chart["Year"].astype(str)
        + "-"
        + monthly_chart["Month"].astype(str)
        + "-01"
    )

    monthly_chart = monthly_chart.sort_values("Date")

    fig = px.line(
        monthly_chart,
        x="Date",
        y="AverageArrivalDelay",
        markers=True,
        title="Average arrival delay by month",
        labels={
            "Date": "Month",
            "AverageArrivalDelay": "Minutes",
        },
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Airline performance")

    airline_summary = (
        airlines.groupby("Reporting_Airline")
        .agg(
            Flights=("Flights", "sum"),
            Cancelled=("Cancelled", "sum"),
            ValidArrivals=("ValidArrivals", "sum"),
            DelaySum=("DelaySum", "sum"),
        )
        .reset_index()
    )

    airline_summary["CancellationRate"] = (
        airline_summary["Cancelled"]
        / airline_summary["Flights"]
        * 100
    )

    airline_summary["AverageArrivalDelay"] = (
        airline_summary["DelaySum"]
        / airline_summary["ValidArrivals"].replace(
            0, np.nan
        )
    )

    airline_summary["Airline"] = (
        airline_summary["Reporting_Airline"].map(
            lambda code: (
                f"{AIRLINE_NAMES.get(code, code)} ({code})"
            )
        )
    )

    fig = px.scatter(
        airline_summary,
        x="CancellationRate",
        y="AverageArrivalDelay",
        size="Flights",
        hover_name="Airline",
        hover_data=["Flights"],
        title="Airline delay vs. cancellation rate",
        labels={
            "CancellationRate": "Cancellation rate (%)",
            "AverageArrivalDelay": "Average arrival delay (minutes)",
        },
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Reported causes of arrival delay")

    cause_summary = (
        causes.groupby("Cause")["DelayMinutes"]
        .sum()
        .reset_index()
        .sort_values("DelayMinutes")
    )

    fig = px.bar(
        cause_summary,
        x="DelayMinutes",
        y="Cause",
        orientation="h",
        title="Total minutes attributed to delay causes",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Arrival-delay statistics exclude records with missing arrival "
        "delay. A flight is classified as late when its arrival delay "
        "is at least 15 minutes."
    )


# =========================================================
# TAB 5: AVIATION SAFETY
# =========================================================

with tab_safety:

    st.header("Aviation safety narratives")

    st.write(
        "Explore NTSB accident reports, accident locations, and "
        "patterns in probable-cause findings."
    )

    if not ACCIDENTS_PATH.exists():

        st.warning(
            "NTSB data is missing. Add the processed accident "
            "Parquet file to enable this section."
        )

    else:

        try:

            accidents = load_accidents()

        except Exception as exc:

            st.error(f"Could not load NTSB data: {exc}")
            accidents = pd.DataFrame()

        if not accidents.empty:

            safety = accidents.dropna(
                subset=["ev_year"]
            ).copy()

            safety["ev_year"] = (
                safety["ev_year"].astype(int)
            )

            years = sorted(
                safety["ev_year"].unique()
            )

            selected_safety_years = st.slider(
                "Accident years",
                min_value=int(min(years)),
                max_value=int(max(years)),
                value=(
                    int(min(years)),
                    int(max(years)),
                ),
                key="safety_years",
            )

            safety = safety[
                safety["ev_year"].between(
                    selected_safety_years[0],
                    selected_safety_years[1],
                )
            ]

            total_accidents = len(safety)

            with_cause = (
                safety["Probable_Cause"].str.strip().ne("").sum()
            )

            fatal_reports = (
                safety["inj_tot_f"].fillna(0).gt(0).sum()
            )

            s1, s2, s3 = st.columns(3)

            s1.metric(
                "Accident records",
                f"{total_accidents:,}",
            )

            s2.metric(
                "Reports with probable cause",
                f"{with_cause:,}",
            )

            s3.metric(
                "Reports with fatalities",
                f"{fatal_reports:,}",
            )

            st.subheader("Accident reports over time")

            annual = (
                safety.groupby("ev_year")
                .size()
                .reset_index(name="Accident Reports")
            )

            fig = px.line(
                annual,
                x="ev_year",
                y="Accident Reports",
                markers=True,
                title="NTSB accident reports by year",
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Accident locations")

            map_data = safety.dropna(
                subset=[
                    "dec_latitude",
                    "dec_longitude",
                ]
            ).copy()

            if "ev_country" in map_data.columns:

                country = (
                    map_data["ev_country"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.upper()
                )

                us_mask = country.isin(
                    [
                        "US",
                        "USA",
                        "UNITED STATES",
                        "UNITED STATES OF AMERICA",
                    ]
                )

                if us_mask.any():
                    map_data = map_data[us_mask]

            map_data = map_data[
                map_data["dec_latitude"].between(15, 75)
                & map_data["dec_longitude"].between(-180, -50)
            ]

            if not map_data.empty:

                fig = px.scatter_geo(
                    map_data,
                    lat="dec_latitude",
                    lon="dec_longitude",
                    color="ev_highest_injury",
                    hover_name="ev_city",
                    hover_data=[
                        "ev_year",
                        "ev_state",
                        "ev_type",
                        "inj_tot_f",
                    ],
                    scope="usa",
                    projection="albers usa",
                    title="NTSB accident locations in the United States",
                )

                fig.update_layout(height=550)

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            else:

                st.info(
                    "No U.S. accident coordinates are available "
                    "for this period."
                )

            st.subheader("Common probable-cause terms")

            cause_text = safety[
                "Probable_Cause"
            ].dropna().astype(str)

            cause_text = cause_text[
                cause_text.str.strip().ne("")
            ]

            if not cause_text.empty:

                try:

                    from sklearn.feature_extraction.text import (
                        CountVectorizer
                    )

                    vectorizer = CountVectorizer(
                        stop_words="english",
                        max_features=15,
                        ngram_range=(1, 2),
                        token_pattern=(
                            r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
                        ),
                    )

                    matrix = vectorizer.fit_transform(
                        cause_text
                    )

                    term_counts = np.asarray(
                        matrix.sum(axis=0)
                    ).ravel()

                    terms = pd.DataFrame(
                        {
                            "Term": vectorizer.get_feature_names_out(),
                            "Count": term_counts,
                        }
                    ).sort_values("Count")

                    fig = px.bar(
                        terms,
                        x="Count",
                        y="Term",
                        orientation="h",
                        title="Most frequent words and phrases",
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                except ValueError:

                    st.info(
                        "Not enough usable probable-cause text "
                        "to perform this analysis."
                    )

                except ImportError:

                    st.warning(
                        "Install scikit-learn to enable text analysis."
                    )

            st.subheader("Search accident narratives")

            search_term = st.text_input(
                "Search factual narratives and probable causes",
                placeholder="Example: weather, maintenance, pilot, engine",
            )

            if search_term.strip():

                match = (
                    safety["Narrative"].str.contains(
                        search_term,
                        case=False,
                        regex=False,
                    )
                    |
                    safety["Probable_Cause"].str.contains(
                        search_term,
                        case=False,
                        regex=False,
                    )
                )

                matches = safety[match]

                st.write(
                    f"Matching reports: **{len(matches):,}**"
                )

                if not matches.empty:

                    display_columns = [
                        col for col in [
                            "ev_date",
                            "ev_year",
                            "ev_city",
                            "ev_state",
                            "ev_type",
                            "Narrative",
                            "Probable_Cause",
                        ]
                        if col in matches.columns
                    ]

                    st.dataframe(
                        matches[display_columns].head(100),
                        use_container_width=True,
                        hide_index=True,
                    )

                else:

                    st.info("No reports matched that search.")

            st.warning(
                "Scope limitation: NTSB accident records include general "
                "aviation and other civil aviation activity, not just "
                "scheduled commercial airline flights. These records "
                "represent a different population from the BTS flight data."
            )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "Sources: Bureau of Transportation Statistics (BTS) On-Time "
    "Performance; OurAirports; National Transportation Safety Board "
    "(NTSB). Results are descriptive and do not establish causation."
)

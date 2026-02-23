"""
Module 1 - Historical Weather API
Fetches temperature, precipitation and wind speed data for Madrid, London
and Rio de Janeiro from the Open-Meteo Historical Weather API (2010-2020),
processes the data and plots the temporal evolution of each variable.
"""

import time
import logging
from typing import Optional

import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_URL = "https://archive-api.open-meteo.com/v1/archive"

COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}

VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]

START_DATE = "2010-01-01"
END_DATE = "2020-12-31"

# API call settings
MAX_RETRIES = 3
BACKOFF_SECONDS = 2.0

# Expected schema keys in the API response
REQUIRED_RESPONSE_KEYS = {"latitude", "longitude", "daily", "daily_units"}
REQUIRED_DAILY_KEYS = {"time"} | set(VARIABLES)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generic API caller
# ---------------------------------------------------------------------------


def call_api(
    url: str,
    params: dict,
    max_retries: int = MAX_RETRIES,
    backoff: float = BACKOFF_SECONDS,
) -> dict:
    """
    Generic helper to call an HTTP GET endpoint.

    Handles:
    - Non-200 status codes (raises RuntimeError).
    - Rate limiting / server errors (429, 5xx) with exponential back-off.
    - Network-level exceptions with retries.

    Parameters
    ----------
    url : str
        Full endpoint URL.
    params : dict
        Query parameters to send with the request.
    max_retries : int
        Maximum number of retry attempts.
    backoff : float
        Base seconds to wait between retries (doubles each attempt).

    Returns
    -------
    dict
        Parsed JSON response body.

    Raises
    ------
    RuntimeError
        If the request fails after all retries or returns a non-retryable error.
    """
    attempt = 0
    wait = backoff

    while attempt < max_retries:
        try:
            response = requests.get(url, params=params, timeout=30)

            # Rate-limited or server error: back off and retry
            if response.status_code in (429, 500, 502, 503, 504):
                logger.warning(
                    "Status %s received (attempt %d/%d). Retrying in %.1fs ...",
                    response.status_code,
                    attempt + 1,
                    max_retries,
                    wait,
                )
                time.sleep(wait)
                wait *= 2
                attempt += 1
                continue

            # Client error (e.g. bad params): fail immediately
            if not response.ok:
                error_body = response.json() if response.content else {}
                reason = error_body.get("reason", response.text)
                raise RuntimeError(
                    f"API returned HTTP {response.status_code}: {reason}"
                )

            return response.json()

        except requests.exceptions.RequestException as exc:
            logger.warning(
                "Network error (attempt %d/%d): %s. Retrying in %.1fs ...",
                attempt + 1,
                max_retries,
                exc,
                wait,
            )
            time.sleep(wait)
            wait *= 2
            attempt += 1

    raise RuntimeError(f"API call to {url} failed after {max_retries} attempts.")


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def validate_response(data: dict) -> None:
    """
    Validate that the API response contains the expected keys.

    Raises
    ------
    ValueError
        If required keys are missing, signalling a contract change in the API.
    """
    missing_top = REQUIRED_RESPONSE_KEYS - data.keys()
    if missing_top:
        raise ValueError(f"API response missing expected top-level keys: {missing_top}")

    missing_daily = REQUIRED_DAILY_KEYS - data["daily"].keys()
    if missing_daily:
        raise ValueError(
            f"API response 'daily' object missing expected keys: {missing_daily}"
        )


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------


def get_data_meteo_api(
    city: str,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    variables: Optional[list] = None,
) -> pd.DataFrame:
    """
    Fetch daily historical weather data from Open-Meteo for a given city.

    Parameters
    ----------
    city : str
        City name. Must be a key in COORDINATES.
    start_date : str
        Start date in ISO 8601 format (YYYY-MM-DD).
    end_date : str
        End date in ISO 8601 format (YYYY-MM-DD).
    variables : list, optional
        List of daily variable names to retrieve. Defaults to VARIABLES.

    Returns
    -------
    pd.DataFrame
        DataFrame with a DatetimeIndex and one column per requested variable,
        plus a 'city' column.

    Raises
    ------
    ValueError
        If the city is not found in COORDINATES or the response schema is invalid.
    RuntimeError
        If the API call fails.
    """
    if city not in COORDINATES:
        raise ValueError(
            f"City '{city}' not found. Available cities: {list(COORDINATES.keys())}"
        )

    if variables is None:
        variables = VARIABLES

    coords = COORDINATES[city]
    params = {
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "daily": ",".join(variables),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
    }

    logger.info("Fetching data for %s (%s → %s) ...", city, start_date, end_date)
    raw = call_api(API_URL, params)

    validate_response(raw)

    daily = raw["daily"]
    df = pd.DataFrame(daily)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    df["city"] = city

    logger.info("  → %d daily records received for %s.", len(df), city)
    return df


# ---------------------------------------------------------------------------
# Data processing
# ---------------------------------------------------------------------------


def fetch_all_cities(
    cities: Optional[list] = None,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    """
    Fetch and concatenate weather data for multiple cities.

    Parameters
    ----------
    cities : list, optional
        List of city names. Defaults to all keys in COORDINATES.
    start_date : str
        Start date in ISO 8601 format.
    end_date : str
        End date in ISO 8601 format.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame for all cities with a 'city' column.
    """
    if cities is None:
        cities = list(COORDINATES.keys())

    frames = [get_data_meteo_api(city, start_date, end_date) for city in cities]
    return pd.concat(frames)


def resample_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily data to monthly means/sums per city.

    Aggregation rules:
    - temperature_2m_mean    → monthly mean
    - precipitation_sum      → monthly sum
    - wind_speed_10m_max     → monthly mean of daily maxima

    Parameters
    ----------
    df : pd.DataFrame
        Combined daily DataFrame with a 'city' column.

    Returns
    -------
    pd.DataFrame
        Monthly aggregated DataFrame.
    """
    agg_rules = {
        "temperature_2m_mean": "mean",
        "precipitation_sum": "sum",
        "wind_speed_10m_max": "mean",
    }
    # Keep only known variable columns that actually exist in df
    agg_rules = {k: v for k, v in agg_rules.items() if k in df.columns}

    monthly = (
        df.groupby("city")
        .resample("ME")
        .agg(agg_rules)
        .reset_index(level=0)  # brings 'city' back as a column, keeps DatetimeIndex
    )
    return monthly


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

VARIABLE_LABELS = {
    "temperature_2m_mean": ("Mean Temperature (2 m)", "°C"),
    "precipitation_sum": ("Total Precipitation", "mm"),
    "wind_speed_10m_max": ("Max Wind Speed (10 m)", "km/h"),
}

CITY_COLORS = {
    "Madrid": "#e07b39",
    "London": "#3a86ff",
    "Rio": "#2dc653",
}


def plot_weather_variables(monthly_df: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot the monthly evolution of each weather variable for all cities.

    Creates a figure with one subplot per variable (3 rows × 1 column).
    Each city is shown as a distinct coloured line.

    Parameters
    ----------
    monthly_df : pd.DataFrame
        Monthly aggregated DataFrame with a 'city' column.
    save_path : str, optional
        If provided, saves the figure to this file path.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        "Historical Weather Evolution (2010–2020)\nMadrid · London · Rio de Janeiro",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    cities = monthly_df["city"].unique()

    for ax, variable in zip(axes, VARIABLES):
        label, unit = VARIABLE_LABELS[variable]

        for city in cities:
            city_df = monthly_df[monthly_df["city"] == city]
            ax.plot(
                city_df.index,
                city_df[variable],
                label=city,
                color=CITY_COLORS.get(city),
                linewidth=1.2,
                alpha=0.85,
            )

        ax.set_ylabel(f"{label}\n({unit})", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.tick_params(axis="x", rotation=30)

    axes[-1].set_xlabel("Date", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Figure saved to %s", save_path)

    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Entry point: fetch, process and plot historical weather data for
    Madrid, London and Rio de Janeiro (2010–2020).
    """
    # 1. Fetch daily data for all cities
    daily_df = fetch_all_cities()

    # 2. Resample to monthly resolution for cleaner visualisation
    monthly_df = resample_monthly(daily_df)

    # 3. Plot
    plot_weather_variables(monthly_df, save_path="weather_evolution_2010_2020.png")


if __name__ == "__main__":
    main()

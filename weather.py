"""Open-Meteo access and temperature trend analysis."""

from __future__ import annotations

import calendar
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Literal

import httpx
import pandas as pd
from scipy import stats

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
BASELINE_START = 1991
BASELINE_END = 2020

Dataset = Literal["era5_land", "era5"]


class WeatherApiError(RuntimeError):
    """A user-facing error raised for failed or malformed API responses."""


@dataclass(frozen=True)
class Location:
    name: str
    latitude: float
    longitude: float
    country: str = ""
    admin1: str = ""
    timezone: str = "auto"

    @property
    def label(self) -> str:
        parts = [self.name, self.admin1, self.country]
        return ", ".join(dict.fromkeys(part for part in parts if part))


@dataclass(frozen=True)
class TrendStatistics:
    slope_per_decade: float
    confidence_low: float
    confidence_high: float
    p_value: float
    r_squared: float
    intercept: float


def _get_json(
    url: str,
    params: dict[str, Any],
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    owns_client = client is None
    active_client = client or httpx.Client(timeout=60.0)
    try:
        response = active_client.get(url, params=params)
        payload = response.json()
        if isinstance(payload, dict) and payload.get("error"):
            raise WeatherApiError(
                str(payload.get("reason", "Open-Meteo rejected the request."))
            )
        response.raise_for_status()
    except WeatherApiError:
        raise
    except (httpx.HTTPError, ValueError) as exc:
        raise WeatherApiError(f"Open-Meteo request failed: {exc}") from exc
    finally:
        if owns_client:
            active_client.close()

    if not isinstance(payload, dict):
        raise WeatherApiError("Open-Meteo returned an unexpected response.")
    return payload


def search_locations(
    query: str,
    *,
    count: int = 10,
    client: httpx.Client | None = None,
) -> list[Location]:
    query = query.strip()
    if len(query) < 2:
        return []
    payload = _get_json(
        GEOCODING_URL,
        {"name": query, "count": count, "language": "en", "format": "json"},
        client,
    )
    locations: list[Location] = []
    for item in payload.get("results", []):
        try:
            locations.append(
                Location(
                    name=str(item["name"]),
                    latitude=float(item["latitude"]),
                    longitude=float(item["longitude"]),
                    country=str(item.get("country", "")),
                    admin1=str(item.get("admin1", "")),
                    timezone=str(item.get("timezone", "auto")),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return locations


def _fetch_daily_weather_range(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    dataset: Dataset,
    *,
    client: httpx.Client | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = _get_json(
        ARCHIVE_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min",
            "models": dataset,
            "timezone": "auto",
        },
        client,
    )
    daily = payload.get("daily")
    if not isinstance(daily, dict) or "time" not in daily:
        raise WeatherApiError("Open-Meteo response did not contain daily weather data.")

    columns = {
        "date": daily.get("time", []),
        "temperature_mean": daily.get("temperature_2m_mean", []),
        "temperature_max": daily.get("temperature_2m_max", []),
        "temperature_min": daily.get("temperature_2m_min", []),
    }
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1:
        raise WeatherApiError("Open-Meteo returned weather columns of different lengths.")

    frame = pd.DataFrame(columns)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("temperature_mean", "temperature_max", "temperature_min"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    metadata = {
        "latitude": payload.get("latitude"),
        "longitude": payload.get("longitude"),
        "elevation": payload.get("elevation"),
        "timezone": payload.get("timezone", "auto"),
        "units": payload.get("daily_units", {}),
    }
    return frame, metadata


def _year_chunks(start_date: date, end_date: date, years: int = 10) -> list[tuple[date, date]]:
    chunks: list[tuple[date, date]] = []
    current = start_date
    while current <= end_date:
        chunk_end = min(end_date, date(min(current.year + years - 1, end_date.year), 12, 31))
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return chunks


def fetch_daily_weather(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    dataset: Dataset,
    *,
    client: httpx.Client | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch daily data, splitting long histories to avoid archive timeouts."""
    if start_date > end_date:
        raise ValueError("Start date must not be after end date.")
    if dataset not in ("era5_land", "era5"):
        raise ValueError(f"Unsupported dataset: {dataset}")
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError("Coordinates are outside valid latitude/longitude bounds.")

    chunks = _year_chunks(start_date, end_date)

    def fetch_chunk(bounds: tuple[date, date]) -> tuple[pd.DataFrame, dict[str, Any]]:
        return _fetch_daily_weather_range(
            latitude,
            longitude,
            bounds[0],
            bounds[1],
            dataset,
            client=client,
        )

    # An injected client may have a stateful test transport; keep those requests sequential.
    if client is not None or len(chunks) == 1:
        responses = [fetch_chunk(bounds) for bounds in chunks]
    else:
        with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as executor:
            responses = list(executor.map(fetch_chunk, chunks))

    frames = [frame for frame, _ in responses]
    combined = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    return combined, responses[0][1]


def aggregate_yearly(
    daily: pd.DataFrame,
    month: int | None,
    temperature_column: str = "temperature_mean",
    aggregation: Literal["mean", "max", "min"] = "mean",
) -> pd.DataFrame:
    """Aggregate complete annual or selected-month temperatures by year."""
    allowed_columns = {"temperature_mean", "temperature_max", "temperature_min"}
    if temperature_column not in allowed_columns:
        raise ValueError(f"Unsupported temperature column: {temperature_column}")
    if aggregation not in {"mean", "max", "min"}:
        raise ValueError(f"Unsupported aggregation: {aggregation}")
    required = {"date", temperature_column}
    if not required.issubset(daily.columns):
        raise ValueError(f"Daily data must contain columns: {sorted(required)}")
    if month is not None and not 1 <= month <= 12:
        raise ValueError("Month must be between 1 and 12.")

    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["year"] = frame["date"].dt.year
    if month is not None:
        frame = frame[frame["date"].dt.month == month]

    records: list[dict[str, Any]] = []
    for year, group in frame.groupby("year"):
        expected = (
            calendar.monthrange(int(year), month)[1]
            if month is not None
            else 366 if calendar.isleap(int(year)) else 365
        )
        valid = group.dropna(subset=[temperature_column])
        if valid["date"].dt.normalize().nunique() != expected:
            continue
        records.append(
            {
                "year": int(year),
                "temperature": float(valid[temperature_column].agg(aggregation)),
                "days": expected,
            }
        )
    return pd.DataFrame(records, columns=["year", "temperature", "days"])


def add_anomalies(
    yearly: pd.DataFrame,
    baseline_start: int = BASELINE_START,
    baseline_end: int = BASELINE_END,
) -> tuple[pd.DataFrame, float]:
    baseline = yearly[yearly["year"].between(baseline_start, baseline_end)]
    expected_years = baseline_end - baseline_start + 1
    if len(baseline) != expected_years:
        raise ValueError(
            f"A complete {baseline_start}–{baseline_end} baseline is required."
        )
    normal = float(baseline["temperature"].mean())
    result = yearly.copy()
    result["anomaly"] = result["temperature"] - normal
    return result, normal


def calculate_trend(yearly: pd.DataFrame) -> TrendStatistics:
    clean = yearly.dropna(subset=["year", "temperature"])
    if len(clean) < 10:
        raise ValueError("At least 10 complete years are required for trend statistics.")
    regression = stats.linregress(clean["year"], clean["temperature"])
    critical_t = stats.t.ppf(0.975, df=len(clean) - 2)
    slope_per_decade = float(regression.slope * 10)
    margin_per_decade = float(critical_t * regression.stderr * 10)
    return TrendStatistics(
        slope_per_decade=slope_per_decade,
        confidence_low=slope_per_decade - margin_per_decade,
        confidence_high=slope_per_decade + margin_per_decade,
        p_value=float(regression.pvalue),
        r_squared=float(regression.rvalue**2),
        intercept=float(regression.intercept),
    )


def add_fitted_trend(yearly: pd.DataFrame, trend: TrendStatistics) -> pd.DataFrame:
    result = yearly.copy()
    slope_per_year = trend.slope_per_decade / 10
    result["fitted_temperature"] = slope_per_year * result["year"] + trend.intercept
    return result


def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, float_format="%.3f").encode("utf-8")

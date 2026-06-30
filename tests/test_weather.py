from __future__ import annotations

from datetime import date

import httpx
import pandas as pd
import pytest

from main import trend_chart
from weather import (
    WeatherApiError,
    add_anomalies,
    aggregate_yearly,
    calculate_trend,
    dataframe_to_csv_bytes,
    fetch_daily_weather,
    search_locations,
)


@pytest.mark.parametrize("temperature_label", ["Mean", "Maximum", "Minimum"])
def test_trend_chart_labels_observed_temperature_measure(temperature_label: str) -> None:
    frame = pd.DataFrame(
        {"year": [2000], "temperature": [10.0], "fitted_temperature": [10.0]}
    )

    figure = trend_chart(frame, "Annual", temperature_label)

    assert figure.data[0].name == f"Observed {temperature_label.lower()}"


def mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_search_locations_normalizes_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["name"] == "Warsaw"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "name": "Warsaw",
                        "latitude": 52.22977,
                        "longitude": 21.01178,
                        "country": "Poland",
                        "admin1": "Masovian Voivodeship",
                        "timezone": "Europe/Warsaw",
                    }
                ]
            },
        )

    with mock_client(handler) as client:
        results = search_locations("Warsaw", client=client)
    assert len(results) == 1
    assert results[0].label == "Warsaw, Masovian Voivodeship, Poland"
    assert results[0].timezone == "Europe/Warsaw"


def test_search_locations_handles_empty_and_short_queries() -> None:
    assert search_locations("x") == []

    with mock_client(lambda _: httpx.Response(200, json={})) as client:
        assert search_locations("missing", client=client) == []


def test_fetch_daily_weather_builds_query_and_normalizes_data() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["models"] == "era5_land"
        assert request.url.params["timezone"] == "auto"
        assert request.url.params["start_date"] == "1991-01-01"
        return httpx.Response(
            200,
            json={
                "latitude": 52.2,
                "longitude": 21.0,
                "timezone": "Europe/Warsaw",
                "elevation": 113,
                "daily_units": {"temperature_2m_mean": "°C"},
                "daily": {
                    "time": ["1991-01-01", "1991-01-02"],
                    "temperature_2m_mean": [1.1, 1.4],
                    "temperature_2m_max": [2.5, 2.7],
                    "temperature_2m_min": [-0.4, 0.2],
                },
            },
        )

    with mock_client(handler) as client:
        frame, metadata = fetch_daily_weather(
            52.2298,
            21.0118,
            date(1991, 1, 1),
            date(1991, 1, 2),
            "era5_land",
            client=client,
        )
    assert list(frame.columns) == [
        "date",
        "temperature_mean",
        "temperature_max",
        "temperature_min",
    ]
    assert frame["temperature_mean"].tolist() == [1.1, 1.4]
    assert metadata["timezone"] == "Europe/Warsaw"


def test_fetch_daily_weather_splits_long_ranges() -> None:
    requested_ranges = []

    def handler(request: httpx.Request) -> httpx.Response:
        start = request.url.params["start_date"]
        end = request.url.params["end_date"]
        requested_ranges.append((start, end))
        return httpx.Response(
            200,
            json={
                "daily": {
                    "time": [start],
                    "temperature_2m_mean": [1],
                    "temperature_2m_max": [2],
                    "temperature_2m_min": [0],
                }
            },
        )

    with mock_client(handler) as client:
        frame, _ = fetch_daily_weather(
            0,
            0,
            date(1991, 1, 1),
            date(2011, 12, 31),
            "era5",
            client=client,
        )

    assert requested_ranges == [
        ("1991-01-01", "2000-12-31"),
        ("2001-01-01", "2010-12-31"),
        ("2011-01-01", "2011-12-31"),
    ]
    assert len(frame) == 3


def test_api_errors_are_user_facing() -> None:
    with mock_client(
        lambda _: httpx.Response(400, json={"error": True, "reason": "bad dates"})
    ) as client:
        with pytest.raises(WeatherApiError, match="bad dates"):
            search_locations("Warsaw", client=client)


def make_daily(start: str, end: str, value_by_year=None) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    values = [
        (value_by_year or {}).get(timestamp.year, float(timestamp.year - 1990))
        for timestamp in dates
    ]
    return pd.DataFrame(
        {
            "date": dates,
            "temperature_mean": values,
            "temperature_max": [value + 5 for value in values],
            "temperature_min": [value - 5 for value in values],
        }
    )


def test_annual_aggregation_accepts_leap_year_and_rejects_incomplete_year() -> None:
    daily = make_daily("1999-01-01", "2000-12-31")
    daily = daily[daily["date"] != pd.Timestamp("1999-06-01")]
    yearly = aggregate_yearly(daily, None)
    assert yearly["year"].tolist() == [2000]
    assert yearly["days"].tolist() == [366]


def test_monthly_aggregation_rejects_incomplete_month() -> None:
    daily = make_daily("2000-01-01", "2001-01-31")
    daily = daily[daily["date"] != pd.Timestamp("2001-01-15")]
    yearly = aggregate_yearly(daily, 1)
    assert yearly["year"].tolist() == [2000]
    assert yearly["days"].tolist() == [31]


def test_aggregation_can_use_peak_daily_maximum_temperature() -> None:
    daily = make_daily("2000-01-01", "2000-01-31")
    daily.loc[daily["date"] == pd.Timestamp("2000-01-15"), "temperature_max"] = 32.7
    yearly = aggregate_yearly(daily, 1, "temperature_max", "max")
    assert yearly.iloc[0]["temperature"] == pytest.approx(32.7)


def test_aggregation_can_use_lowest_daily_minimum_temperature() -> None:
    daily = make_daily("2000-01-01", "2000-01-31")
    daily.loc[daily["date"] == pd.Timestamp("2000-01-15"), "temperature_min"] = -18.2
    yearly = aggregate_yearly(daily, 1, "temperature_min", "min")
    assert yearly.iloc[0]["temperature"] == pytest.approx(-18.2)


def test_anomaly_baseline_and_linear_trend() -> None:
    years = list(range(1991, 2021))
    yearly = pd.DataFrame(
        {
            "year": years,
            "temperature": [10 + 0.1 * (year - 1991) for year in years],
            "days": [365] * len(years),
        }
    )
    with_anomalies, normal = add_anomalies(yearly)
    trend = calculate_trend(with_anomalies)
    assert normal == pytest.approx(11.45)
    assert with_anomalies.iloc[0]["anomaly"] == pytest.approx(-1.45)
    assert trend.slope_per_decade == pytest.approx(1.0)
    assert trend.r_squared == pytest.approx(1.0)


def test_baseline_requires_all_thirty_years() -> None:
    yearly = pd.DataFrame({"year": range(1992, 2021), "temperature": 10.0})
    with pytest.raises(ValueError, match="complete 1991–2020 baseline"):
        add_anomalies(yearly)


def test_trend_requires_ten_years() -> None:
    yearly = pd.DataFrame({"year": range(2000, 2009), "temperature": range(9)})
    with pytest.raises(ValueError, match="At least 10"):
        calculate_trend(yearly)


def test_csv_export_is_utf8_and_uses_stable_precision() -> None:
    output = dataframe_to_csv_bytes(pd.DataFrame({"value": [1.23456]})).decode()
    assert output == "value\n1.235\n"

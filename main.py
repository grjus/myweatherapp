"""Streamlit entry point for the historical weather dashboard."""

from __future__ import annotations

import calendar
import importlib
import inspect
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import weather as weather_module

from weather import (
    BASELINE_END,
    BASELINE_START,
    CurrentWeather,
    Location,
    WeatherApiError,
    add_anomalies,
    add_fitted_trend,
    calculate_trend,
    dataframe_to_csv_bytes,
    fetch_current_weather,
    fetch_daily_weather,
    search_locations,
)


def aggregate_temperature(
    daily: pd.DataFrame,
    month: int | None,
    temperature_column: str,
    aggregation: str,
) -> pd.DataFrame:
    """Reload the analysis module when Streamlit retained its pre-change version."""
    function = weather_module.aggregate_yearly
    if "aggregation" not in inspect.signature(function).parameters:
        function = importlib.reload(weather_module).aggregate_yearly
    return function(daily, month, temperature_column, aggregation)  # type: ignore[arg-type]


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def cached_location_search(query: str) -> list[Location]:
    return search_locations(query)


@st.cache_data(ttl=10 * 60, max_entries=500, show_spinner=False)
def cached_current_weather(latitude: float, longitude: float) -> CurrentWeather:
    return fetch_current_weather(latitude, longitude)


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def cached_weather(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    dataset: str,
) -> tuple[pd.DataFrame, dict]:
    return fetch_daily_weather(latitude, longitude, start_date, end_date, dataset)  # type: ignore[arg-type]


def trend_chart(frame: pd.DataFrame, period_label: str, temperature_label: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame["year"],
            y=frame["temperature"],
            mode="lines+markers",
            name=f"Observed {temperature_label.lower()}",
            hovertemplate="%{x}: %{y:.2f} °C<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["year"],
            y=frame["fitted_temperature"],
            mode="lines",
            name="Linear trend",
            line={"dash": "dash", "color": "#d62728"},
            hovertemplate="%{x}: %{y:.2f} °C<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"{period_label} {temperature_label.lower()} temperature by year",
        xaxis_title="Year",
        yaxis_title="Temperature (°C)",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08},
    )
    return figure


def anomaly_chart(
    frame: pd.DataFrame, period_label: str, temperature_label: str
) -> go.Figure:
    colors = ["#d62728" if value >= 0 else "#1f77b4" for value in frame["anomaly"]]
    figure = go.Figure(
        go.Bar(
            x=frame["year"],
            y=frame["anomaly"],
            marker_color=colors,
            hovertemplate="%{x}: %{y:+.2f} °C<extra></extra>",
        )
    )
    figure.add_hline(y=0, line_color="black", line_width=1)
    figure.update_layout(
        title=(
            f"{period_label} {temperature_label.lower()} temperature anomalies "
            "from the 1991–2020 normal"
        ),
        xaxis_title="Year",
        yaxis_title="Temperature anomaly (°C)",
    )
    return figure


def daily_chart(frame: pd.DataFrame, year: int, month: int | None) -> go.Figure:
    selected = frame[frame["date"].dt.year == year]
    if month is not None:
        selected = selected[selected["date"].dt.month == month]
    figure = go.Figure()
    for column, label, color in (
        ("temperature_max", "Maximum", "#d62728"),
        ("temperature_mean", "Mean", "#2ca02c"),
        ("temperature_min", "Minimum", "#1f77b4"),
    ):
        figure.add_trace(
            go.Scatter(x=selected["date"], y=selected[column], name=label, line={"color": color})
        )
    figure.update_layout(
        title=f"Daily temperatures in {year}",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08},
    )
    return figure


def resolve_location() -> Location | None:
    st.sidebar.subheader("Location")
    manual = st.sidebar.toggle("Enter coordinates manually")
    if manual:
        latitude = st.sidebar.number_input("Latitude", -90.0, 90.0, 52.2298, format="%.4f")
        longitude = st.sidebar.number_input("Longitude", -180.0, 180.0, 21.0118, format="%.4f")
        return Location("Custom location", latitude, longitude)

    query = st.sidebar.text_input("City or postal code", value="Warsaw")
    try:
        results = cached_location_search(query) if len(query.strip()) >= 2 else []
    except WeatherApiError as exc:
        st.sidebar.error(str(exc))
        return None
    if not results:
        st.sidebar.warning("No matching location found.")
        return None
    return st.sidebar.selectbox("Matching location", results, format_func=lambda item: item.label)


def app() -> None:
    st.set_page_config(page_title="Historical Weather Trends", page_icon="🌡️", layout="wide")
    st.title("Historical Weather Trends")
    st.caption("Explore long-term local temperature patterns using consistent reanalysis data.")

    location = resolve_location()
    st.sidebar.subheader("Analysis")
    dataset_label = st.sidebar.selectbox(
        "Dataset",
        ("ERA5-Land (~11 km, 1950 onward)", "ERA5 (~25 km, 1940 onward)"),
    )
    dataset = "era5_land" if dataset_label.startswith("ERA5-Land") else "era5"
    minimum_year = 1950 if dataset == "era5_land" else 1940
    latest_full_year = date.today().year - 1
    start_year, end_year = st.sidebar.slider(
        "Year range",
        min_value=minimum_year,
        max_value=latest_full_year,
        value=(minimum_year, latest_full_year),
    )
    month_labels = ["Annual"] + list(calendar.month_name[1:])
    month_label = st.sidebar.selectbox("Period", month_labels)
    month = None if month_label == "Annual" else list(calendar.month_name).index(month_label)
    temperature_options = {
        "Mean (average daily mean)": ("temperature_mean", "mean"),
        "Maximum (highest daily maximum)": ("temperature_max", "max"),
        "Minimum (lowest daily minimum)": ("temperature_min", "min"),
    }
    temperature_label = st.sidebar.selectbox("Temperature measure", temperature_options)
    temperature_column, aggregation = temperature_options[temperature_label]
    short_temperature_label = temperature_label.split(" (")[0]

    if location is None:
        st.info("Choose a valid location to begin.")
        return

    st.subheader(location.label)
    try:
        current = cached_current_weather(location.latitude, location.longitude)
    except (WeatherApiError, ValueError) as exc:
        st.warning(f"Current conditions are unavailable: {exc}")
    else:
        st.caption(
            f"Current conditions · {current.description} · "
            f"Updated {current.observed_at} ({current.timezone})"
        )
        current_columns = st.columns(4)
        current_columns[0].metric("Temperature", f"{current.temperature:.1f} °C")
        current_columns[1].metric(
            "Feels like", f"{current.apparent_temperature:.1f} °C"
        )
        current_columns[2].metric("Humidity", f"{current.relative_humidity}%")
        current_columns[3].metric("Wind", f"{current.wind_speed:.1f} km/h")

    st.divider()
    st.subheader("Historical trends")
    fetch_start_year = min(start_year, BASELINE_START)
    fetch_end_year = max(end_year, BASELINE_END)
    try:
        with st.spinner("Loading historical weather data…"):
            daily, metadata = cached_weather(
                location.latitude,
                location.longitude,
                date(fetch_start_year, 1, 1),
                date(fetch_end_year, 12, 31),
                dataset,
            )
        all_yearly = aggregate_temperature(
            daily, month, temperature_column, aggregation
        )
        all_yearly, normal = add_anomalies(all_yearly)
        yearly = all_yearly[all_yearly["year"].between(start_year, end_year)].copy()
        trend = calculate_trend(yearly)
        yearly = add_fitted_trend(yearly, trend)
    except (WeatherApiError, ValueError) as exc:
        st.error(str(exc))
        return

    st.caption(
        f"Requested coordinates: {location.latitude:.4f}, {location.longitude:.4f} · "
        f"Grid point: {metadata.get('latitude')}, {metadata.get('longitude')} · "
        f"Timezone: {metadata.get('timezone')}"
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Trend", f"{trend.slope_per_decade:+.2f} °C/decade")
    metric_columns[1].metric(
        "95% confidence interval",
        f"{trend.confidence_low:+.2f} to {trend.confidence_high:+.2f}",
    )
    metric_columns[2].metric("R²", f"{trend.r_squared:.3f}")
    metric_columns[3].metric("p-value", f"{trend.p_value:.3g}")
    st.caption(
        f"1991–2020 {month_label.lower()} {short_temperature_label.lower()} "
        f"temperature normal: {normal:.2f} °C"
    )

    trend_tab, anomaly_tab, daily_tab, data_tab = st.tabs(
        ("Trend", "Anomalies", "Daily detail", "Data")
    )
    with trend_tab:
        st.plotly_chart(
            trend_chart(yearly, month_label, short_temperature_label), use_container_width=True
        )
    with anomaly_tab:
        st.plotly_chart(
            anomaly_chart(yearly, month_label, short_temperature_label), use_container_width=True
        )
    with daily_tab:
        detail_year = st.select_slider(
            "Detail year", options=list(range(start_year, end_year + 1)), value=end_year
        )
        st.plotly_chart(daily_chart(daily, detail_year, month), use_container_width=True)
    with data_tab:
        visible_daily = daily[
            daily["date"].dt.year.between(start_year, end_year)
            & (True if month is None else daily["date"].dt.month == month)
        ]
        st.download_button(
            "Download aggregated CSV",
            dataframe_to_csv_bytes(yearly),
            file_name=(
                f"{dataset}_{month_label.lower()}_"
                f"{short_temperature_label.lower()}_trend.csv"
            ),
            mime="text/csv",
        )
        st.download_button(
            "Download daily CSV",
            dataframe_to_csv_bytes(visible_daily),
            file_name=f"{dataset}_{month_label.lower()}_daily.csv",
            mime="text/csv",
        )
        st.dataframe(yearly, use_container_width=True, hide_index=True)

    with st.expander("Methodology and limitations"):
        st.markdown(
            """
            Mean temperature is the average of daily means. Maximum temperature is the single
            highest daily maximum in the selected month or year; minimum temperature is the single
            lowest daily minimum. Only complete years or months are included. Anomalies use the
            1991–2020 WMO climatological normal. The
            displayed trend is ordinary least-squares regression; its confidence interval and
            p-value describe statistical uncertainty, not physical causation.

            ERA5 and ERA5-Land are gridded reanalysis estimates, not readings from a particular
            weather station. A local trend can be consistent with broader climate warming, but this
            dashboard alone cannot attribute that trend to greenhouse-gas emissions. Local land use,
            natural variability, and the selected period can also affect results.
            """
        )

    st.divider()
    st.caption(
        "Weather data: Open-Meteo and Copernicus Climate Change Service ERA5/ERA5-Land (CC BY 4.0). "
        "Location search: GeoNames via Open-Meteo."
    )


if __name__ == "__main__":
    app()

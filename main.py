"""Streamlit entry point for the historical weather dashboard."""

from __future__ import annotations

import importlib
import inspect
from datetime import date
from typing import cast

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import weather as weather_module

from i18n import Language, SUPPORTED_LANGUAGES, month_name, t, weather_description
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
def cached_location_search(query: str, language: str) -> list[Location]:
    parameters = inspect.signature(search_locations).parameters
    if "language" in parameters:
        return search_locations(query, language=language)
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


def initialize_language() -> Language:
    """Initialize language once from the URL, then browser locale, then English."""
    selected = st.session_state.get("language")
    if selected in SUPPORTED_LANGUAGES:
        return cast(Language, selected)

    query_value = st.query_params.get("lang", "")
    if isinstance(query_value, list):
        query_value = query_value[-1] if query_value else ""
    query_language = str(query_value).upper()
    if query_language in SUPPORTED_LANGUAGES:
        language = cast(Language, query_language)
    else:
        browser_locale = (getattr(st.context, "locale", None) or "").lower()
        language = "PL" if browser_locale.startswith("pl") else "EN"
    st.session_state["language"] = language
    return language


def query_value(name: str) -> str | None:
    """Return the last scalar value for a query parameter."""
    value = st.query_params.get(name)
    if isinstance(value, list):
        return value[-1] if value else None
    return str(value) if value is not None else None


def query_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(query_value(name) or default)
    except ValueError:
        return default
    return value if minimum <= value <= maximum else default


def query_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(query_value(name) or default)
    except ValueError:
        return default
    return value if minimum <= value <= maximum else default


def mirror_options_to_url(
    language: Language,
    location: Location | None,
    dataset: str,
    start_year: int,
    end_year: int,
    month: int | None,
    temperature_measure: str,
) -> None:
    """Replace query parameters with the current shareable dashboard state."""
    parameters = {
        "lang": language.lower(),
        "mode": "manual" if st.session_state.manual_coordinates else "search",
        "dataset": dataset,
        "start": str(start_year),
        "end": str(end_year),
        "period": "annual" if month is None else str(month),
        "measure": temperature_measure,
    }
    if st.session_state.manual_coordinates:
        parameters["lat"] = f"{st.session_state.latitude:.4f}"
        parameters["lon"] = f"{st.session_state.longitude:.4f}"
    else:
        parameters["q"] = st.session_state.location_query
        if location is not None:
            parameters["loc"] = f"{location.latitude:.5f},{location.longitude:.5f}"
    st.query_params.from_dict(parameters)


def trend_chart(
    frame: pd.DataFrame,
    period_label: str,
    temperature_label: str,
    language: Language = "EN",
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame["year"],
            y=frame["temperature"],
            mode="lines+markers",
            name=t("observed", language, measure=temperature_label.lower()),
            hovertemplate="%{x}: %{y:.2f} °C<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["year"],
            y=frame["fitted_temperature"],
            mode="lines",
            name=t("linear_trend", language),
            line={"dash": "dash", "color": "#d62728"},
            hovertemplate="%{x}: %{y:.2f} °C<extra></extra>",
        )
    )
    figure.update_layout(
        title=t(
            "trend_chart_title",
            language,
            period=period_label,
            measure=temperature_label.lower(),
        ),
        xaxis_title=t("year", language),
        yaxis_title=t("temperature_axis", language),
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08},
    )
    return figure


def anomaly_chart(
    frame: pd.DataFrame,
    period_label: str,
    temperature_label: str,
    language: Language = "EN",
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
        title=t(
            "anomaly_chart_title",
            language,
            period=period_label,
            measure=temperature_label.lower(),
        ),
        xaxis_title=t("year", language),
        yaxis_title=t("anomaly_axis", language),
    )
    return figure


def daily_chart(
    frame: pd.DataFrame,
    year: int,
    month: int | None,
    language: Language = "EN",
) -> go.Figure:
    selected = frame[frame["date"].dt.year == year]
    if month is not None:
        selected = selected[selected["date"].dt.month == month]
    figure = go.Figure()
    for column, label, color in (
        ("temperature_max", t("series_max", language), "#d62728"),
        ("temperature_mean", t("series_mean", language), "#2ca02c"),
        ("temperature_min", t("series_min", language), "#1f77b4"),
    ):
        figure.add_trace(
            go.Scatter(x=selected["date"], y=selected[column], name=label, line={"color": color})
        )
    figure.update_layout(
        title=t("daily_chart_title", language, year=year),
        xaxis_title=t("date", language),
        yaxis_title=t("temperature_axis", language),
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08},
    )
    return figure


def resolve_location(language: Language) -> Location | None:
    st.sidebar.subheader(t("location", language))
    if "manual_coordinates" not in st.session_state:
        st.session_state.manual_coordinates = query_value("mode") == "manual"
    manual = st.sidebar.toggle(
        t("manual_coordinates", language), key="manual_coordinates"
    )
    if manual:
        if "latitude" not in st.session_state:
            st.session_state.latitude = query_float("lat", 52.2298, -90, 90)
        if "longitude" not in st.session_state:
            st.session_state.longitude = query_float("lon", 21.0118, -180, 180)
        latitude = st.sidebar.number_input(
            t("latitude", language),
            -90.0,
            90.0,
            format="%.4f",
            key="latitude",
        )
        longitude = st.sidebar.number_input(
            t("longitude", language),
            -180.0,
            180.0,
            format="%.4f",
            key="longitude",
        )
        return Location(t("custom_location", language), latitude, longitude)

    if "location_query" not in st.session_state:
        st.session_state.location_query = query_value("q") or t(
            "default_location_query", language
        )
    query = st.sidebar.text_input(
        t("city_search", language),
        key="location_query",
    )
    try:
        api_language = "pl" if language == "PL" else "en"
        results = (
            cached_location_search(query, api_language) if len(query.strip()) >= 2 else []
        )
    except WeatherApiError:
        st.sidebar.error(t("location_search_failed", language))
        return None
    if not results:
        st.sidebar.warning(t("no_location", language))
        return None
    if st.session_state.get("matching_location") not in results:
        selected_coordinates = query_value("loc")
        selected = results[0]
        if selected_coordinates:
            try:
                latitude_text, longitude_text = selected_coordinates.split(",", 1)
                latitude = float(latitude_text)
                longitude = float(longitude_text)
                selected = min(
                    results,
                    key=lambda item: abs(item.latitude - latitude)
                    + abs(item.longitude - longitude),
                )
            except ValueError:
                pass
        st.session_state.matching_location = selected
    return st.sidebar.selectbox(
        t("matching_location", language),
        results,
        format_func=lambda item: item.label,
        key="matching_location",
    )


def app() -> None:
    language = initialize_language()
    st.set_page_config(page_title=t("title", language), page_icon="🌡️", layout="wide")
    title_column, language_column = st.columns([5, 1], vertical_alignment="center")
    selected_language = language_column.segmented_control(
        t("language", language),
        SUPPORTED_LANGUAGES,
        key="language",
        width="stretch",
    )
    language = cast(Language, selected_language or language)
    title_column.title(t("title", language))
    st.caption(t("subtitle", language))

    location = resolve_location(language)
    st.sidebar.subheader(t("analysis", language))
    if "dataset" not in st.session_state:
        requested_dataset = query_value("dataset")
        st.session_state.dataset = (
            requested_dataset
            if requested_dataset in ("era5_land", "era5")
            else "era5_land"
        )
    dataset = st.sidebar.selectbox(
        t("dataset", language),
        ("era5_land", "era5"),
        format_func=lambda value: t(f"dataset_{value}", language),
        key="dataset",
    )
    minimum_year = 1950 if dataset == "era5_land" else 1940
    latest_full_year = date.today().year - 1
    if "year_range" not in st.session_state:
        start = query_int("start", minimum_year, minimum_year, latest_full_year)
        end = query_int("end", latest_full_year, minimum_year, latest_full_year)
        st.session_state.year_range = (min(start, end), max(start, end))
    else:
        start, end = st.session_state.year_range
        st.session_state.year_range = (
            max(minimum_year, min(start, latest_full_year)),
            max(minimum_year, min(end, latest_full_year)),
        )
    start_year, end_year = st.sidebar.slider(
        t("year_range", language),
        min_value=minimum_year,
        max_value=latest_full_year,
        key="year_range",
    )
    if "period" not in st.session_state:
        requested_period = query_value("period")
        st.session_state.period = (
            int(requested_period)
            if requested_period and requested_period.isdigit()
            and 1 <= int(requested_period) <= 12
            else None
        )
    month = st.sidebar.selectbox(
        t("period", language),
        [None, *range(1, 13)],
        format_func=lambda value: month_name(language, value),
        key="period",
    )
    month_label = month_name(language, month)
    temperature_options = {
        "mean": ("temperature_mean", "mean"),
        "max": ("temperature_max", "max"),
        "min": ("temperature_min", "min"),
    }
    if "temperature_measure" not in st.session_state:
        requested_measure = query_value("measure")
        st.session_state.temperature_measure = (
            requested_measure if requested_measure in temperature_options else "mean"
        )
    temperature_measure = st.sidebar.selectbox(
        t("temperature_measure", language),
        list(temperature_options),
        format_func=lambda value: t(f"temp_option_{value}", language),
        key="temperature_measure",
    )
    temperature_column, aggregation = temperature_options[temperature_measure]
    short_temperature_label = t(f"temp_{temperature_measure}", language)

    mirror_options_to_url(
        language,
        location,
        dataset,
        start_year,
        end_year,
        month,
        temperature_measure,
    )

    if location is None:
        st.info(t("choose_location", language))
        return

    st.subheader(location.label)
    try:
        current = cached_current_weather(location.latitude, location.longitude)
    except (WeatherApiError, ValueError):
        st.warning(t("current_unavailable", language))
    else:
        conditions = weather_description(language, current.weather_code)
        st.caption(
            f"{t('current_conditions', language)} · {conditions} · "
            f"{t('updated', language, time=current.observed_at, timezone=current.timezone)}"
        )
        current_columns = st.columns(4)
        current_columns[0].metric(
            t("temperature", language),
            t("temperature_value", language, value=current.temperature),
        )
        current_columns[1].metric(
            t("feels_like", language),
            t("temperature_value", language, value=current.apparent_temperature),
        )
        current_columns[2].metric(
            t("humidity", language),
            t("humidity_value", language, value=current.relative_humidity),
        )
        current_columns[3].metric(
            t("wind", language),
            t("wind_value", language, value=current.wind_speed),
        )

    st.divider()
    st.subheader(t("historical_trends", language))
    fetch_start_year = min(start_year, BASELINE_START)
    fetch_end_year = max(end_year, BASELINE_END)
    try:
        with st.spinner(t("loading_history", language)):
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
    except (WeatherApiError, ValueError):
        st.error(t("historical_unavailable", language))
        return

    st.caption(
        t(
            "coordinates_caption",
            language,
            latitude=location.latitude,
            longitude=location.longitude,
            grid_latitude=metadata.get("latitude"),
            grid_longitude=metadata.get("longitude"),
            timezone=metadata.get("timezone"),
        )
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        t("trend", language),
        t("trend_value", language, value=trend.slope_per_decade),
    )
    metric_columns[1].metric(
        t("confidence_interval", language),
        t(
            "confidence_value",
            language,
            low=trend.confidence_low,
            high=trend.confidence_high,
        ),
    )
    metric_columns[2].metric(t("r_squared", language), f"{trend.r_squared:.3f}")
    metric_columns[3].metric(t("p_value", language), f"{trend.p_value:.3g}")
    st.caption(
        t(
            "normal_caption",
            language,
            period=month_label.lower(),
            measure=short_temperature_label.lower(),
            normal=normal,
        )
    )

    trend_tab, anomaly_tab, daily_tab, data_tab = st.tabs(
        (
            t("tab_trend", language),
            t("tab_anomalies", language),
            t("tab_daily", language),
            t("tab_data", language),
        )
    )
    with trend_tab:
        st.plotly_chart(
            trend_chart(yearly, month_label, short_temperature_label, language),
            use_container_width=True,
        )
    with anomaly_tab:
        st.plotly_chart(
            anomaly_chart(yearly, month_label, short_temperature_label, language),
            use_container_width=True,
        )
    with daily_tab:
        detail_year = st.select_slider(
            t("detail_year", language),
            options=list(range(start_year, end_year + 1)),
            value=end_year,
            key="detail_year",
        )
        st.plotly_chart(
            daily_chart(daily, detail_year, month, language), use_container_width=True
        )
    with data_tab:
        visible_daily = daily[
            daily["date"].dt.year.between(start_year, end_year)
            & (True if month is None else daily["date"].dt.month == month)
        ]
        st.download_button(
            t("download_aggregated", language),
            dataframe_to_csv_bytes(yearly),
            file_name=(
                f"{dataset}_{month_label.lower()}_"
                f"{short_temperature_label.lower()}_trend.csv"
            ),
            mime="text/csv",
        )
        st.download_button(
            t("download_daily", language),
            dataframe_to_csv_bytes(visible_daily),
            file_name=f"{dataset}_{month_label.lower()}_daily.csv",
            mime="text/csv",
        )
        display_yearly = yearly.rename(
            columns={
                "year": t("year", language),
                "temperature": t("column_temperature", language),
                "days": t("column_days", language),
                "anomaly": t("column_anomaly", language),
                "fitted_temperature": t("column_fitted", language),
            }
        )
        st.dataframe(display_yearly, use_container_width=True, hide_index=True)

    with st.expander(t("methodology_title", language)):
        st.markdown(t("methodology", language))

    st.divider()
    st.caption(t("attribution", language))


if __name__ == "__main__":
    app()

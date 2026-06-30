# Historical Weather Trends

A local Streamlit dashboard for exploring historical temperature trends by location. It uses
[Open-Meteo](https://open-meteo.com/) historical reanalysis and geocoding APIs, which require no
API key for non-commercial use.

## Run

This project requires Python 3.12 and uses `uv` for dependency management.

```bash
uv sync --dev
uv run streamlit run main.py
```

Then open the local URL printed by Streamlit. Search for a city or postal code, or enter coordinates
directly. Choose ERA5-Land for finer-resolution land data since 1950, or ERA5 for global data since
1940. Select mean, peak maximum, or lowest minimum temperature and an annual or calendar-month
analysis. The dashboard also shows current temperature, apparent temperature, humidity, wind, and
weather conditions for the selected location.

## What the dashboard calculates

- Average daily mean, highest daily maximum, or lowest daily minimum for each complete period
- Temperature anomalies relative to the 1991–2020 WMO climatological normal
- Ordinary least-squares trend in °C per decade, with a 95% confidence interval, p-value, and R²
- Daily minimum, mean, and maximum temperature views
- Downloadable daily and aggregated CSV files

The app silently includes 1991–2020 data when necessary to calculate the fixed normal. Incomplete
years or months are excluded from trend calculations, and at least ten complete observations are
required. Long histories are fetched in parallel ten-year chunks and cached for 24 hours to avoid
archive timeouts and repeated API use.

## Interpretation and data source

ERA5 and ERA5-Land are gridded reanalysis products rather than measurements from an individual
weather station. They combine observations with weather models to provide spatially complete,
consistent records. ERA5-Land is the default because its grid is finer over land; ERA5 is available
for ocean locations and the additional 1940–1949 period.

A local temperature trend is useful evidence about warming or cooling at that location, but it does
not independently establish greenhouse-gas causation. Attribution requires broader observations and
physical climate analysis.

Data attribution: Open-Meteo and Copernicus Climate Change Service ERA5/ERA5-Land (CC BY 4.0).
Location search data is provided by GeoNames through Open-Meteo.

## Test

```bash
uv run pytest
```

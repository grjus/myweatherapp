# Repository Guidelines

## Project Structure & Module Organization

- `main.py` is the Streamlit entry point. It owns sidebar controls, caching, charts, and page layout.
- `weather.py` contains API access, data normalization, yearly aggregation, anomaly calculations, and trend analysis. Keep reusable logic here rather than in the UI layer.
- `tests/test_weather.py` contains the pytest suite, including mocked Open-Meteo responses and analysis tests.
- `pyproject.toml` defines Python 3.12 requirements, runtime dependencies, and pytest configuration. `uv.lock` pins resolved dependencies and should be committed.
- `README.md` documents user-facing setup, behavior, and methodology. Update it when commands or visible behavior change.

## Build, Test, and Development Commands

Use `uv` for environment and dependency management:

```bash
uv sync --dev                  # Create/update .venv and install dependencies
uv run streamlit run main.py  # Start the dashboard locally
uv run pytest                 # Run the complete test suite
uv run pytest tests/test_weather.py -q  # Run the main test module concisely
```

The project has no separate build step; Streamlit runs directly from source.

## Coding Style & Naming Conventions

Follow standard Python conventions: four-space indentation, type hints for public functions, concise docstrings where intent is not obvious, `snake_case` for functions and variables, and `PascalCase` for classes. Keep imports grouped as standard library, third-party packages, then local modules. Prefer small, testable functions in `weather.py`; keep Streamlit-specific calls in `main.py`.

No formatter or linter is currently configured. Match the existing style and avoid unrelated formatting changes.

## Testing Guidelines

Tests use pytest. Name files `test_*.py` and functions `test_<behavior>`. Mock HTTP calls with `httpx.MockTransport`; tests must not depend on live network access. Cover successful transformations, incomplete-data handling, API errors, and boundary conditions. Add a regression test with every bug fix. No coverage threshold is configured, but changed behavior should be directly exercised.

## Commit & Pull Request Guidelines

The repository currently has no Git history, so no local commit convention exists. Use short, imperative subjects such as `Fix temperature legend label`, and keep each commit focused.

Pull requests should explain the behavior change, list verification commands, and link relevant issues. Include screenshots for dashboard or chart changes. Call out changes to data interpretation, API usage, dependencies, or methodology explicitly.

## Security & Configuration

Do not commit `.env` files, virtual environments, caches, or `.streamlit/secrets.toml`. Open-Meteo currently requires no application API key; keep any future credentials in ignored local configuration.

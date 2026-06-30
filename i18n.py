"""Load localized UI text from external JSON catalogs."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast

Language = Literal["EN", "PL"]
SUPPORTED_LANGUAGES: tuple[Language, ...] = ("EN", "PL")
LOCALES_DIR = Path(__file__).with_name("locales")


@lru_cache(maxsize=len(SUPPORTED_LANGUAGES))
def _catalog(language: Language) -> dict[str, Any]:
    path = LOCALES_DIR / f"{language.lower()}.json"
    with path.open(encoding="utf-8") as catalog_file:
        return cast(dict[str, Any], json.load(catalog_file))


def t(key: str, language: Language, **values: Any) -> str:
    """Translate a dotted catalog key and interpolate named values."""
    value: Any = _catalog(language)
    try:
        for part in key.split("."):
            value = value[part]
    except (KeyError, TypeError) as exc:
        raise KeyError(f"Missing translation key {key!r} for {language}") from exc
    if not isinstance(value, str):
        raise TypeError(f"Translation key {key!r} for {language} is not text")
    return value.format(**values)


def month_name(language: Language, month: int | None) -> str:
    if month is None:
        return t("annual", language)
    months = _catalog(language)["months"]
    return cast(str, months[month - 1])


def weather_description(language: Language, code: int) -> str:
    weather = _catalog(language)["weather"]
    return cast(str, weather.get(str(code), weather["unknown"]))

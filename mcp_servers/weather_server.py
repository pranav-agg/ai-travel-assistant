"""MCP server: current conditions and forecast for the destination.

Backed by Open-Meteo (https://open-meteo.com) — keyless for non-commercial use.

* The model supplies concrete ISO dates; it never passes a day offset. It
  derives those dates from the date anchor injected into its system prompt.
  This server's job is to *validate* them, not to interpret relative phrases.
* Every returned day carries its own `date` and `weekday`. The assistant is
  instructed to copy those verbatim into itinerary headings rather than
  computing them, which is what stops fabricated dates appearing in output.

Run standalone:  python mcp_servers/weather_server.py
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    CITY_COORDS,
    DEMO_FORCE_TOOL_FAILURE,
    DESTINATION_TZ,
    HTTP_TIMEOUT_SECONDS,
    OPEN_METEO_URL,
)
from src.dates import (  # noqa: E402
    clamp_num_days,
    today_local,
    validate_window,
)

mcp = FastMCP("weather")

DAILY_VARIABLES = [
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "precipitation_probability_max",
]

# WMO 4677 weather codes as served by Open-Meteo.
WMO_CODES: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snowfall",
    73: "moderate snowfall",
    75: "heavy snowfall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def _describe(code: int | None) -> str:
    if code is None:
        return "unknown conditions"
    return WMO_CODES.get(int(code), f"unrecognised weather code {code}")


def _outdoor_suitability(precip_probability: float | None, code: int | None) -> str:
    """Derive an indoor/outdoor verdict so the LLM need not do meteorology.

    Singapore gets short, intense convective showers most afternoons, so
    probability alone overstates disruption; thunderstorms are the real
    blocker for outdoor plans.
    """
    if code is not None and int(code) in (95, 96, 99):
        return "poor"
    if precip_probability is None:
        return "mixed"
    if precip_probability >= 60:
        return "poor"
    if precip_probability >= 30:
        return "mixed"
    return "good"


def _resolve_city(city: str) -> tuple[float, float] | None:
    return CITY_COORDS.get(city.strip().lower())


def _error(message: str, *, recoverable: bool = True, **extra) -> dict:
    """Structured error. Never raises into the agent, never invents data."""
    return {"error": message, "tool": "get_weather_forecast",
            "recoverable": recoverable, **extra}


@mcp.tool()
def get_weather_forecast(
    city: str = "Singapore",
    start_date: str | None = None,
    num_days: int = 3,
) -> dict:
    """Get a daily weather forecast for a destination.

    Args:
        city: Destination name. Singapore and nearby cities are supported.
        start_date: First day to forecast, as ISO YYYY-MM-DD. Defaults to
            today in the destination's timezone. Must fall within the next
            16 days — derive it from the date anchor in your system prompt.
        num_days: How many consecutive days to return (1-16). Trimmed
            automatically if the window would run past the forecast horizon.

    Returns:
        A dict with `days`, each carrying its own `date` and `weekday`.
        Copy those into itinerary headings verbatim; do not compute dates.
    """
    if DEMO_FORCE_TOOL_FAILURE == "weather":
        return _error(
            "Weather service unavailable (DEMO_FORCE_TOOL_FAILURE is set). "
            "Answer using knowledge-base content only and say the forecast "
            "could not be retrieved."
        )

    coords = _resolve_city(city)
    if coords is None:
        return _error(
            f"No coordinates configured for '{city}'. "
            f"Supported: {', '.join(sorted(CITY_COORDS))}.",
            supported_cities=sorted(CITY_COORDS),
        )

    today = today_local(DESTINATION_TZ)

    if start_date is None:
        start = today
    else:
        try:
            start = date.fromisoformat(start_date.strip())
        except ValueError:
            return _error(
                f"start_date must be ISO YYYY-MM-DD, got '{start_date}'."
            )

    window_error = validate_window(start, num_days, today)
    if window_error is not None:
        return {**window_error.as_dict(), "tool": "get_weather_forecast",
                "today_local": today.isoformat()}

    effective_days = clamp_num_days(start, num_days, today)
    end = start + timedelta(days=effective_days - 1)

    latitude, longitude = coords
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": DESTINATION_TZ.key,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }

    try:
        with httpx.Client(timeout=HTTP_TIMEOUT_SECONDS) as client:
            response = client.get(OPEN_METEO_URL, params=params)
            response.raise_for_status()
            payload = response.json()
    except httpx.TimeoutException:
        return _error("Weather service timed out. No forecast retrieved.")
    except httpx.HTTPStatusError as exc:
        return _error(
            f"Weather service returned HTTP {exc.response.status_code}. "
            "No forecast retrieved."
        )
    except httpx.HTTPError as exc:
        return _error(f"Could not reach the weather service: {exc}")
    except ValueError:
        return _error("Weather service returned a malformed response.")

    daily = payload.get("daily") or {}
    dates = daily.get("time") or []
    if not dates:
        return _error("Weather service returned no daily data for that window.")

    def col(name: str) -> list:
        values = daily.get(name) or []
        return values + [None] * (len(dates) - len(values))

    codes = col("weather_code")
    temp_max = col("temperature_2m_max")
    temp_min = col("temperature_2m_min")
    precip_sum = col("precipitation_sum")
    precip_prob = col("precipitation_probability_max")

    days = []
    for i, iso in enumerate(dates):
        day = date.fromisoformat(iso)
        days.append(
            {
                "date": iso,
                "weekday": day.strftime("%A"),
                "condition": _describe(codes[i]),
                "temp_max_c": temp_max[i],
                "temp_min_c": temp_min[i],
                "precip_mm": precip_sum[i],
                "precip_probability_pct": precip_prob[i],
                "outdoor_suitability": _outdoor_suitability(precip_prob[i], codes[i]),
            }
        )

    label = (
        f"{start.strftime('%a %d %b')} - {end.strftime('%a %d %b %Y')}"
        if start != end
        else start.strftime("%a %d %b %Y")
    )

    return {
        "source": "Open-Meteo",
        "source_url": "https://open-meteo.com/",
        "retrieved_at": datetime.now(DESTINATION_TZ).isoformat(timespec="seconds"),
        "today_local": today.isoformat(),
        "location": city.title(),
        "timezone": DESTINATION_TZ.key,
        "requested_window": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "label": label,
            "trimmed_to_horizon": effective_days != num_days,
        },
        "days": days,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")

"""No-key weather helper for live assistant questions."""

from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import quote_plus


WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


@dataclass
class WeatherResult:
    location: str
    temperature_f: float | None
    apparent_temperature_f: float | None
    humidity_percent: float | None
    precipitation_in: float | None
    wind_speed_mph: float | None
    wind_gust_mph: float | None
    weather_code: int | None
    condition: str
    observation_time: str
    source_url: str


def looks_weather_question(question: str) -> bool:
    normalized = question.lower()
    terms = ("weather", "temperature", "forecast", "rain", "snow", "wind", "humid", "humidity")
    return any(term in normalized for term in terms)


def extract_weather_location(question: str) -> str | None:
    text = " ".join(question.strip().split())
    patterns = [
        r"\bin\s+(.+?)(?:\?|$)",
        r"\bfor\s+(.+?)(?:\?|$)",
        r"\bat\s+(.+?)(?:\?|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            location = match.group(1).strip(" .?!,")
            location = re.sub(r"\b(today|now|currently|right now)\b", "", location, flags=re.IGNORECASE)
            location = " ".join(location.split()).strip(" .?!,")
            if location:
                return location
    return None


def _first_geocode_result(location: str) -> dict:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Weather lookup requires requests. Run: pip install requests") from exc

    url = f"https://geocoding-api.open-meteo.com/v1/search?name={quote_plus(location)}&count=1&language=en&format=json"
    response = requests.get(url, timeout=12)
    response.raise_for_status()
    results = response.json().get("results") or []
    if not results:
        raise RuntimeError(f"Could not find weather coordinates for '{location}'.")
    return results[0]


def fetch_current_weather(location: str) -> WeatherResult:
    """Fetch current weather from Open-Meteo without an API key."""

    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Weather lookup requires requests. Run: pip install requests") from exc

    place = _first_geocode_result(location)
    latitude = place["latitude"]
    longitude = place["longitude"]
    place_name = ", ".join(
        part
        for part in [
            place.get("name"),
            place.get("admin1"),
            place.get("country_code"),
        ]
        if part
    )
    forecast_url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        "&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,wind_gusts_10m"
        "&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch&timezone=auto"
    )
    response = requests.get(forecast_url, timeout=12)
    response.raise_for_status()
    current = response.json().get("current") or {}
    weather_code = current.get("weather_code")
    condition = WEATHER_CODES.get(weather_code, f"Weather code {weather_code}") if weather_code is not None else "Unknown"
    return WeatherResult(
        location=place_name,
        temperature_f=current.get("temperature_2m"),
        apparent_temperature_f=current.get("apparent_temperature"),
        humidity_percent=current.get("relative_humidity_2m"),
        precipitation_in=current.get("precipitation"),
        wind_speed_mph=current.get("wind_speed_10m"),
        wind_gust_mph=current.get("wind_gusts_10m"),
        weather_code=weather_code,
        condition=condition,
        observation_time=current.get("time", "unknown"),
        source_url=forecast_url,
    )


def weather_result_to_context(result: WeatherResult) -> str:
    return "\n".join(
        [
            f"Location: {result.location}",
            f"Observation time: {result.observation_time}",
            f"Condition: {result.condition}",
            f"Temperature: {result.temperature_f} F",
            f"Feels like: {result.apparent_temperature_f} F",
            f"Humidity: {result.humidity_percent}%",
            f"Precipitation: {result.precipitation_in} inches",
            f"Wind speed: {result.wind_speed_mph} mph",
            f"Wind gusts: {result.wind_gust_mph} mph",
            "Source: Open-Meteo current forecast API",
        ]
    )

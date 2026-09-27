"""
Tool wrappers for SmartTrip AI agents.

Provides LangChain-compatible tool functions for:
- Weather lookup (Open-Meteo API)
- Tavily web search (events, advisories)
- Flight search (MCP mock data)
- Hotel search (MCP mock data)
- Currency conversion (Frankfurter API with fallback)
- RAG knowledge base search
"""

import json
import os
from pathlib import Path
from typing import Optional

import requests
from langchain_core.tools import tool

# ============================================================
# IMPORT RAG AND MCP MODULES
# ============================================================

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.knowledge_base import (
    search_visa_policy,
    search_destination_guide,
    search_travel_insurance,
    get_all_knowledge,
)


# ============================================================
# DATA DIRECTORY (for direct JSON access — MCP fallback)
# ============================================================

DATA_DIR = Path(__file__).resolve().parent.parent / "database"


# ============================================================
# TOOL 1: GET WEATHER FORECAST
# ============================================================

@tool
def get_weather_forecast(
    city: str,
    num_days: int = 7,
) -> dict:
    """
    Get weather forecast for a city using Open-Meteo API.
    Returns temperature, precipitation, and weather conditions.
    Args:
        city: Name of the city to get weather for.
        num_days: Number of forecast days (1-16). Default 7.
    """

    # Step 1: Geocode the city name to lat/lon
    geocode_url = "https://geocoding-api.open-meteo.com/v1/search"

    try:
        geo_resp = requests.get(
            geocode_url,
            params={"name": city, "count": 1, "language": "en"},
            timeout=10,
        )
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()

        if "results" not in geo_data or not geo_data["results"]:
            return {"error": f"Could not find location: {city}"}

        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]
        resolved_name = geo_data["results"][0].get("name", city)

    except requests.RequestException as e:
        return {"error": f"Geocoding failed: {str(e)}"}

    # Step 2: Fetch weather forecast
    weather_url = "https://api.open-meteo.com/v1/forecast"

    try:
        weather_resp = requests.get(
            weather_url,
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode",
                "timezone": "auto",
                "forecast_days": min(num_days, 16),
            },
            timeout=10,
        )
        weather_resp.raise_for_status()
        weather_data = weather_resp.json()

    except requests.RequestException as e:
        return {"error": f"Weather API failed: {str(e)}"}

    # Step 3: Parse and format the forecast
    daily = weather_data.get("daily", {})
    dates = daily.get("time", [])
    max_temps = daily.get("temperature_2m_max", [])
    min_temps = daily.get("temperature_2m_min", [])
    precip = daily.get("precipitation_sum", [])
    codes = daily.get("weathercode", [])

    # Weather code descriptions
    weather_descriptions = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy",
        3: "Overcast", 45: "Foggy", 48: "Rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
        80: "Slight rain showers", 81: "Moderate rain showers",
        82: "Violent rain showers", 95: "Thunderstorm",
        96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
    }

    forecast_days = []
    for i in range(len(dates)):
        forecast_days.append({
            "date": dates[i],
            "max_temp_c": max_temps[i] if i < len(max_temps) else None,
            "min_temp_c": min_temps[i] if i < len(min_temps) else None,
            "precipitation_mm": precip[i] if i < len(precip) else None,
            "condition": weather_descriptions.get(
                codes[i] if i < len(codes) else 0, "Unknown"
            ),
        })

    return {
        "city": resolved_name,
        "latitude": lat,
        "longitude": lon,
        "forecast": forecast_days,
        "source": "Open-Meteo API (free, no key required)",
    }


# ============================================================
# TOOL 2: SEARCH WEB (TAVILY)
# ============================================================

@tool
def search_web(
    query: str,
    max_results: int = 5,
) -> dict:
    """
    Search the web using Tavily API for local events, festivals,
    closures, and travel advisories.
    Args:
        query: Search query string.
        max_results: Maximum number of results to return. Default 5.
    """
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        return {
            "error": "TAVILY_API_KEY not set in environment variables.",
            "fallback": "Please set your Tavily API key in the .env file.",
        }

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
        )

        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:500],
            })

        return {
            "query": query,
            "results": results,
            "source": "Tavily Web Search",
        }

    except Exception as e:
        return {"error": f"Tavily search failed: {str(e)}"}


# ============================================================
# TOOL 3: SEARCH FLIGHTS (Direct JSON — no MCP needed at runtime)
# ============================================================

@tool
def search_flights(
    origin: str,
    destination: str,
    date: Optional[str] = None,
) -> dict:
    """
    Search available flights for a route and optional date.
    Uses mock flight data from the database.
    Args:
        origin: Departure city.
        destination: Arrival city.
        date: Optional travel date (YYYY-MM-DD). If not given, returns all dates.
    """
    data_path = DATA_DIR / "flights.json"

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            flights = json.load(f)
    except FileNotFoundError:
        return {"error": "Flight database not found."}

    results = []
    for flight in flights:
        origin_match = flight["origin"].lower() == origin.lower()
        dest_match = flight["destination"].lower() == destination.lower()
        date_match = (date is None) or (flight["date"] == date)

        if origin_match and dest_match and date_match:
            results.append(flight)

    if not results:
        return {
            "message": f"No flights found from {origin} to {destination}"
                       + (f" on {date}" if date else ""),
            "suggestion": "Try different dates or nearby airports.",
            "flights": [],
        }

    return {
        "origin": origin,
        "destination": destination,
        "date": date,
        "flights": results,
        "note": "⚠️ Prices are indicative and may vary. Please check airline websites for confirmed fares.",
    }


# ============================================================
# TOOL 4: SEARCH HOTELS
# ============================================================

@tool
def search_hotels(
    city: str,
    max_price_per_night: Optional[float] = None,
) -> dict:
    """
    Search available hotels in a city with optional price filter.
    Args:
        city: City to search hotels in.
        max_price_per_night: Optional max price per night in INR.
    """
    data_path = DATA_DIR / "hotels.json"

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            hotels = json.load(f)
    except FileNotFoundError:
        return {"error": "Hotel database not found."}

    results = []
    for hotel in hotels:
        if hotel["city"].lower() == city.lower():
            if max_price_per_night is None or hotel["price_per_night_inr"] <= max_price_per_night:
                results.append(hotel)

    # Sort by price
    results.sort(key=lambda x: x["price_per_night_inr"])

    if not results:
        return {
            "message": f"No hotels found in {city}"
                       + (f" under ₹{max_price_per_night}/night" if max_price_per_night else ""),
            "hotels": [],
        }

    return {
        "city": city,
        "hotels": results,
        "note": "⚠️ Prices are indicative. Please verify on booking platforms for confirmed rates.",
    }


# ============================================================
# TOOL 5: CONVERT CURRENCY
# ============================================================

@tool
def convert_currency(
    amount: float,
    from_currency: str,
    to_currency: str,
) -> dict:
    """
    Convert an amount from one currency to another using
    Frankfurter API with fallback to indicative rates.
    Args:
        amount: Amount to convert.
        from_currency: Source currency code (e.g., INR, USD).
        to_currency: Target currency code (e.g., SGD, EUR).
    """
    rates = {
        "INR": {"INR": 1.0, "USD": 0.0118, "EUR": 0.0100, "SGD": 0.0158,
                "THB": 0.41, "JPY": 1.76, "AED": 0.0434, "MYR": 0.053,
                "LKR": 3.54, "MVR": 0.182},
        "USD": {"INR": 84.7, "USD": 1.0, "EUR": 0.85, "SGD": 1.34,
                "THB": 34.5, "JPY": 149.0, "AED": 3.67},
        "SGD": {"INR": 63.3, "USD": 0.75, "EUR": 0.63, "SGD": 1.0},
        "EUR": {"INR": 100.0, "USD": 1.18, "EUR": 1.0, "SGD": 1.58},
        "THB": {"INR": 2.44, "USD": 0.029, "THB": 1.0},
        "JPY": {"INR": 0.57, "USD": 0.0067, "JPY": 1.0},
        "AED": {"INR": 23.06, "USD": 0.27, "AED": 1.0},
    }

    from_curr = from_currency.upper()
    to_curr = to_currency.upper()

    try:
        response = requests.get(
            "https://api.frankfurter.dev/v1/latest",
            params={"base": from_curr, "symbols": to_curr},
            timeout=5,
        )
        response.raise_for_status()
        rate_data = response.json()
        exchange_rate = rate_data["rates"][to_curr]
        rate_source = "Frankfurter API (live)"
        rate_date = rate_data.get("date")
    except (requests.RequestException, KeyError, ValueError):
        # Fallback to indicative rates
        if from_curr in rates and to_curr in rates.get(from_curr, {}):
            exchange_rate = rates[from_curr][to_curr]
        else:
            return {"error": f"Cannot convert {from_curr} to {to_curr}"}
        rate_source = "Indicative fallback rates"
        rate_date = None

    converted = round(amount * exchange_rate, 2)

    return {
        "from": from_curr,
        "to": to_curr,
        "original_amount": amount,
        "exchange_rate": exchange_rate,
        "converted_amount": converted,
        "rate_source": rate_source,
        "rate_date": rate_date,
    }


# ============================================================
# TOOL 6: RAG KNOWLEDGE SEARCH
# ============================================================

@tool
def lookup_visa_policy(country: str) -> dict:
    """
    Look up visa requirements for a country (for Indian passport holders).
    Args:
        country: Country name to check visa requirements for.
    """
    policy = search_visa_policy(country)

    if policy is None:
        return {
            "country": country,
            "message": f"No visa policy data found for {country} in our knowledge base.",
            "advice": "Please check the official embassy website for the latest visa requirements.",
        }

    return {
        "country": policy["country"],
        "visa_required": policy["visa_required"],
        "visa_type": policy["visa_type"],
        "duration_allowed": policy["duration_allowed"],
        "processing_time": policy["processing_time"],
        "fee_inr": policy["fee_inr"],
        "documents_required": policy["documents_required"],
        "notes": policy["notes"],
        "official_source": policy["official_source"],
        "last_updated": policy["last_updated"],
        "disclaimer": "⚠️ Visa rules change frequently. Always verify with the official embassy or consulate before travelling.",
    }


@tool
def lookup_destination_guide(city: str) -> dict:
    """
    Look up destination guide for a city including attractions,
    transport, food costs, and travel tips.
    Args:
        city: City name to look up.
    """
    guide = search_destination_guide(city)

    if guide is None:
        return {
            "city": city,
            "message": f"No destination guide found for {city} in our knowledge base.",
            "advice": "I'll use web search to find information about this destination.",
        }

    return guide


@tool
def lookup_travel_insurance(coverage_type: str = "Comprehensive") -> dict:
    """
    Look up travel insurance plans by coverage type.
    Args:
        coverage_type: Type of coverage (Basic, Comprehensive, Premium).
    """
    plans = search_travel_insurance(coverage_type)

    return {
        "coverage_type": coverage_type,
        "plans": plans,
        "disclaimer": "⚠️ Insurance plans shown are illustrative. Compare actual plans from insurers before purchasing.",
    }


# ============================================================
# TOOL LIST FOR AGENTS
# ============================================================

ALL_TOOLS = [
    get_weather_forecast,
    search_web,
    search_flights,
    search_hotels,
    convert_currency,
    lookup_visa_policy,
    lookup_destination_guide,
    lookup_travel_insurance,
]

PLANNER_TOOLS = [
    lookup_visa_policy,
    lookup_destination_guide,
    search_web,
]

ITINERARY_TOOLS = [
    lookup_destination_guide,
    search_web,
    get_weather_forecast,
]

BUDGET_TOOLS = [
    search_flights,
    search_hotels,
    convert_currency,
]

WEATHER_TOOLS = [
    get_weather_forecast,
]

SUMMARY_TOOLS = [
    convert_currency,
    lookup_travel_insurance,
]

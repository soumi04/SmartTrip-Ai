import json
from pathlib import Path

import requests
from mcp.server.fastmcp import FastMCP


# ============================================================
# CREATE MCP SERVER
# ============================================================

mcp = FastMCP("SmartTrip Travel Server")


# ============================================================
# TOOL 1: SEARCH FLIGHTS
# ============================================================

@mcp.tool()
def search_flights(
    origin: str,
    destination: str,
    date: str
) -> list:
    """
    Search available flights for a route and travel date.
    """

    data_path = (
        Path(__file__).resolve().parent.parent
        / "database"
        / "flights.json"
    )

    with open(data_path, "r", encoding="utf-8") as file:
        flights = json.load(file)

    results = []

    for flight in flights:

        if (
            flight["origin"].lower() == origin.lower()
            and
            flight["destination"].lower() == destination.lower()
            and
            flight["date"] == date
        ):
            results.append(flight)

    return results


# ============================================================
# TOOL 2: SEARCH HOTELS
# ============================================================

@mcp.tool()
def search_hotels(city: str) -> list:
    """
    Search available hotels in a city.
    """

    data_path = (
        Path(__file__).resolve().parent.parent
        / "database"
        / "hotels.json"
    )

    with open(data_path, "r", encoding="utf-8") as file:
        hotels = json.load(file)

    results = []

    for hotel in hotels:

        if hotel["city"].lower() == city.lower():
            results.append(hotel)

    return results


# ============================================================
# TOOL 3: CONVERT CURRENCY
# ============================================================

@mcp.tool()
def convert_currency(
    amount: float,
    from_currency: str,
    to_currency: str
) -> dict:
    """
    Convert an amount from one currency to another.
    """

    # Keep indicative rates available when Frankfurter cannot be reached.
    rates = {

        "INR": {
            "INR": 1.0,
            "USD": 0.0118,
            "EUR": 0.0100,
            "SGD": 0.0158,
            "THB": 0.41,
            "AED": 0.0434,
        },

        "USD": {
            "INR": 84.7,
            "USD": 1.0,
            "EUR": 0.85,
            "SGD": 1.34
        },

        "EUR": {
            "INR": 100.0,
            "USD": 1.18,
            "EUR": 1.0,
            "SGD": 1.58
        },

        "SGD": {
            "INR": 63.3,
            "USD": 0.75,
            "EUR": 0.63,
            "SGD": 1.0
        }
    }

    from_currency = from_currency.upper()
    to_currency = to_currency.upper()

    # Check source currency
    if from_currency not in rates:
        return {
            "error": f"Unsupported currency: {from_currency}"
        }

    # Check target currency
    if to_currency not in rates[from_currency]:
        return {
            "error": f"Unsupported currency: {to_currency}"
        }

    try:
        response = requests.get(
            "https://api.frankfurter.dev/v1/latest",
            params={"base": from_currency, "symbols": to_currency},
            timeout=5,
        )
        response.raise_for_status()
        rate_data = response.json()
        exchange_rate = rate_data["rates"][to_currency]
        rate_source = "Frankfurter"
        rate_date = rate_data["date"]
    except (requests.RequestException, KeyError, ValueError):
        exchange_rate = rates[from_currency][to_currency]
        rate_source = "indicative fallback"
        rate_date = None

    converted_amount = amount * exchange_rate

    return {
        "from": from_currency,
        "to": to_currency,
        "amount": amount,
        "exchange_rate": exchange_rate,
        "converted_amount": round(converted_amount, 2),
        "rate_source": rate_source,
        "rate_date": rate_date,
    }


# ============================================================
# START MCP SERVER
# ============================================================

if __name__ == "__main__":
    mcp.run()
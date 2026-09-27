"""
Utility helpers for SmartTrip AI.
"""

import json
from datetime import datetime


def format_currency(amount: float, currency: str = "INR") -> str:
    """Format a number as currency string."""
    symbols = {
        "INR": "₹",
        "USD": "$",
        "EUR": "€",
        "SGD": "S$",
        "THB": "฿",
        "JPY": "¥",
        "AED": "د.إ",
    }
    symbol = symbols.get(currency.upper(), currency)
    return f"{symbol}{amount:,.0f}"


def parse_date(date_str: str) -> str:
    """Try to parse a date string into YYYY-MM-DD format."""
    formats = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str


def safe_json_loads(text: str) -> dict:
    """Safely parse JSON from text that might contain extra content."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return {}

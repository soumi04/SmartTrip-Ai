"""Structured trip-planning workflow used by the Streamlit planner."""

import asyncio
import json
import math
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, TypedDict

import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel
from tavily import TavilyClient

from rag.retrieval import retrieve_travel_knowledge

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
LOCAL_COST_PER_PERSON_PER_DAY_INR = 3000
MAX_REPLANS = 1


class TripState(TypedDict, total=False):
    trip: dict[str, Any]
    execution_plan: list[str]
    travel_knowledge: dict[str, Any]
    events: list[dict[str, Any]]
    weather: list[dict[str, Any]]
    flight_options: list[dict[str, Any]]
    hotel_options: list[dict[str, Any]]
    selected_flight: dict[str, Any] | None
    selected_hotel: dict[str, Any] | None
    exchange_rate: float
    rate_source: str
    rate_date: str | None
    itinerary: list[dict[str, Any]]
    costs: dict[str, Any]
    warnings: list[str]
    advisories: list[Any]
    replans: list[str]
    budget_feedback: str
    result: dict[str, Any]


class ItineraryDay(BaseModel):
    date: str
    theme: str
    morning: str
    afternoon: str
    evening: str
    weather_note: str = ""


class ItineraryDraft(BaseModel):
    days: list[ItineraryDay]


def _append_warning(warnings: list[str], message: str) -> list[str]:
    updated = list(warnings)
    if message not in updated:
        updated.append(message)
    return updated


def _planner_node(state: TripState) -> dict[str, Any]:
    trip = state["trip"]
    plan = [
        "Retrieve destination guidance, visa sources, and date-specific conditions.",
        "Search outbound and return flights and destination hotels through MCP.",
        "Build an interest- and weather-aware day-by-day itinerary.",
        "Convert the estimate to the requested currency and check the budget.",
    ]
    if state.get("replans"):
        plan.append("Retry once with the lowest-priced hotel option.")
    return {
        "execution_plan": plan,
        "advisories": [
            f"Planning {trip['destination']} from {trip['start_date']} through {trip['end_date']}."
        ],
    }


def _weather_condition(code: int) -> str:
    if code == 0:
        return "Clear"
    if code in (1, 2, 3):
        return "Mainly clear to overcast"
    if code in (45, 48):
        return "Fog"
    if code in (51, 53, 55, 56, 57):
        return "Drizzle"
    if code in (61, 63, 65, 66, 67, 80, 81, 82):
        return "Rain"
    if code in (71, 73, 75, 77, 85, 86):
        return "Snow"
    if code in (95, 96, 99):
        return "Thunderstorm"
    return "Variable conditions"


def _get_weather(destination: str, start: date, end: date) -> tuple[list[dict[str, Any]], str | None]:
    today = date.today()
    forecast_end = min(end, today + timedelta(days=16))
    forecast_start = max(start, today)
    if forecast_start > forecast_end:
        return [], "Weather forecast is not available this far ahead; check closer to departure."

    try:
        location_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": destination, "count": 1, "language": "en", "format": "json"},
            timeout=8,
        )
        location_response.raise_for_status()
        locations = location_response.json().get("results", [])
        if not locations:
            return [], f"Open-Meteo could not locate {destination}; weather is unavailable."

        location = locations[0]
        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "start_date": forecast_start.isoformat(),
                "end_date": forecast_end.isoformat(),
                "timezone": "auto",
            },
            timeout=8,
        )
        weather_response.raise_for_status()
        daily = weather_response.json().get("daily", {})
        forecasts = []
        dates = daily.get("time", [])
        rain_probabilities = daily.get("precipitation_probability_max") or [None] * len(dates)
        max_temperatures = daily.get("temperature_2m_max") or [None] * len(dates)
        min_temperatures = daily.get("temperature_2m_min") or [None] * len(dates)
        codes = daily.get("weather_code") or [None] * len(dates)
        for index, forecast_date in enumerate(dates):
            code = codes[index] if index < len(codes) else None
            forecasts.append(
                {
                    "date": forecast_date,
                    "condition": _weather_condition(code) if code is not None else "Unknown",
                    "min_temp_c": min_temperatures[index] if index < len(min_temperatures) else None,
                    "max_temp_c": max_temperatures[index] if index < len(max_temperatures) else None,
                    "precipitation_probability": rain_probabilities[index] if index < len(rain_probabilities) else None,
                    "source": "Open-Meteo",
                }
            )
        warning = None
        if end > forecast_end:
            warning = f"Open-Meteo forecast is available only through {forecast_end.isoformat()} for these dates."
        return forecasts, warning
    except (requests.RequestException, KeyError, ValueError, IndexError) as error:
        return [], f"Open-Meteo weather lookup failed: {error.__class__.__name__}."


def _get_events(destination: str, start: date, end: date) -> tuple[list[dict[str, Any]], str | None]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return [], "Tavily is not configured; current events and closure checks are unavailable."
    try:
        response = TavilyClient(api_key=api_key).search(
            query=(
                f"events festivals public closures travel advisories in {destination} "
                f"between {start.isoformat()} and {end.isoformat()}"
            ),
            topic="news",
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            max_results=5,
            search_depth="basic",
        )
        events = [
            {
                "title": item.get("title", "Travel update"),
                "snippet": item.get("content", ""),
                "url": item.get("url", ""),
                "date": item.get("published_date"),
                "source": "Tavily web search",
            }
            for item in response.get("results", [])
        ]
        return events, None if events else "Tavily returned no event results for the selected dates."
    except Exception as error:
        return [], f"Tavily event search failed: {error.__class__.__name__}."


def _research_node(state: TripState) -> dict[str, Any]:
    trip = state["trip"]
    start = date.fromisoformat(trip["start_date"])
    end = date.fromisoformat(trip["end_date"])
    knowledge = retrieve_travel_knowledge(trip["destination"])
    weather, weather_warning = _get_weather(trip["destination"], start, end)
    events, events_warning = _get_events(trip["destination"], start, end)

    warnings = list(state.get("warnings", []))
    for warning in (weather_warning, events_warning):
        if warning:
            warnings = _append_warning(warnings, warning)
    return {
        "travel_knowledge": knowledge,
        "weather": weather,
        "events": events,
        "warnings": warnings,
    }


def _parse_tool_result(result: Any) -> Any:
    if getattr(result, "isError", False):
        raise RuntimeError("MCP tool returned an error.")
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured

    values = []
    for item in getattr(result, "content", []):
        text = getattr(item, "text", None)
        if text is None:
            continue
        try:
            values.append(json.loads(text))
        except json.JSONDecodeError:
            values.append(text)
    if not values:
        return []
    return values[0] if len(values) == 1 else values


async def _call_mcp_tools(trip: dict[str, Any]) -> dict[str, Any]:
    server_path = PROJECT_ROOT / "mcp_server" / "server.py"
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
        cwd=PROJECT_ROOT,
    )
    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as client:
            await client.initialize()
            calls = [
                client.call_tool(
                    "search_flights",
                    {"origin": trip["origin"], "destination": trip["destination"], "date": trip["start_date"]},
                ),
                client.call_tool(
                    "search_flights",
                    {"origin": trip["destination"], "destination": trip["origin"], "date": trip["end_date"]},
                ),
                client.call_tool("search_hotels", {"city": trip["destination"]}),
            ]
            if trip["currency"] == "INR":
                responses = await asyncio.gather(*calls)
                exchange_rate, rate_source, rate_date = 1.0, "INR base currency", None
            else:
                calls.append(
                    client.call_tool(
                        "convert_currency",
                        {"amount": 1.0, "from_currency": "INR", "to_currency": trip["currency"]},
                    )
                )
                responses = await asyncio.gather(*calls)
                currency_data = _parse_tool_result(responses[3])
                if not isinstance(currency_data, dict) or "exchange_rate" not in currency_data:
                    raise RuntimeError("Currency conversion did not return an exchange rate.")
                exchange_rate = float(currency_data["exchange_rate"])
                rate_source = currency_data.get("rate_source", "Unknown")
                rate_date = currency_data.get("rate_date")

    return {
        "outbound_flights": _parse_tool_result(responses[0]),
        "return_flights": _parse_tool_result(responses[1]),
        "hotels": _parse_tool_result(responses[2]),
        "exchange_rate": exchange_rate,
        "rate_source": rate_source,
        "rate_date": rate_date,
    }


async def _booking_node(state: TripState) -> dict[str, Any]:
    trip = state["trip"]
    data = await _call_mcp_tools(trip)
    outbound = data["outbound_flights"]
    returning = data["return_flights"]
    hotels = data["hotels"]
    outbound = outbound if isinstance(outbound, list) else []
    returning = returning if isinstance(returning, list) else []
    hotels = hotels if isinstance(hotels, list) else []

    traveler_count = int(trip["travelers"])
    flight_options = []
    for outbound_flight in outbound:
        for return_flight in returning:
            flight_options.append(
                {
                    "outbound": outbound_flight,
                    "return": return_flight,
                    "price_inr": (float(outbound_flight["price_inr"]) + float(return_flight["price_inr"])) * traveler_count,
                    "currency": "INR",
                    "indicative": True,
                }
            )
    flight_options.sort(key=lambda option: option["price_inr"])

    start = date.fromisoformat(trip["start_date"])
    end = date.fromisoformat(trip["end_date"])
    nights = (end - start).days
    rooms = math.ceil(traveler_count / 2)
    hotel_options = []
    for hotel in hotels:
        option = dict(hotel)
        option.update(
            {
                "nights": nights,
                "rooms": rooms,
                "total_price_inr": float(hotel["price_per_night_inr"]) * nights * rooms,
                "currency": "INR",
                "indicative": True,
            }
        )
        hotel_options.append(option)
    hotel_options.sort(key=lambda hotel: (-float(hotel.get("rating", 0)), hotel["total_price_inr"]))

    selected_hotel = state.get("selected_hotel")
    if selected_hotel is None and hotel_options:
        selected_hotel = hotel_options[0]

    warnings = list(state.get("warnings", []))
    if not outbound:
        warnings = _append_warning(warnings, "No outbound mock flight matched this route and date; airfare is excluded from the estimate.")
    if not returning:
        warnings = _append_warning(warnings, "No return mock flight matched this route and date; airfare is incomplete.")
    if not hotel_options:
        warnings = _append_warning(warnings, "No mock hotels matched this destination; lodging is excluded from the estimate.")

    return {
        "flight_options": flight_options,
        "hotel_options": hotel_options,
        "selected_flight": flight_options[0] if flight_options else None,
        "selected_hotel": selected_hotel,
        "exchange_rate": data["exchange_rate"],
        "rate_source": data["rate_source"],
        "rate_date": data["rate_date"],
        "warnings": warnings,
    }


def _weather_by_date(weather: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["date"]: item for item in weather if item.get("date")}


def _curated_itinerary(state: TripState) -> list[dict[str, Any]]:
    trip = state["trip"]
    start = date.fromisoformat(trip["start_date"])
    end = date.fromisoformat(trip["end_date"])
    knowledge = state.get("travel_knowledge", {})
    activities = list(knowledge.get("activities", []))
    interests = {str(value).casefold() for value in trip.get("interests", [])}
    if interests:
        preferred = [item for item in activities if item.get("interest", "").casefold() in interests]
        activities = preferred + [item for item in activities if item not in preferred]
    if not activities:
        activities = [
            {"name": "Explore a central neighbourhood", "setting": "outdoor", "interest": "General"},
            {"name": "Visit a local museum or gallery", "setting": "indoor", "interest": "Culture"},
            {"name": "Try a local market or food hall", "setting": "indoor", "interest": "Food"},
        ]

    weather = _weather_by_date(state.get("weather", []))
    itinerary = []
    trip_days = (end - start).days + 1
    for index in range(trip_days):
        day_date = start + timedelta(days=index)
        weather_item = weather.get(day_date.isoformat(), {})
        rain_chance = weather_item.get("precipitation_probability")
        rainy = isinstance(rain_chance, (int, float)) and rain_chance >= 60
        available = activities
        if rainy:
            indoors = [item for item in activities if item.get("setting") == "indoor"]
            if indoors:
                available = indoors

        first = available[(index * 2) % len(available)]
        second = available[(index * 2 + 1) % len(available)]
        if index == 0:
            morning = f"Depart {trip['origin']} for {trip['destination']}; keep plans light around arrival."
        elif index == trip_days - 1:
            morning = "Check out and leave time for the return journey."
        else:
            morning = f"Visit {first['name']}."

        if index == trip_days - 1:
            afternoon = f"Return to {trip['origin']} on {day_date.isoformat()}; confirm flight times."
        else:
            afternoon = f"Explore {second['name']}; check opening times before setting out."
        evening = "Choose a nearby dinner option and keep the evening flexible."
        if index == 0:
            evening = "Settle in, have dinner near your accommodation, and rest after travel."
        elif rainy:
            evening = "Choose an indoor dinner option and recheck local conditions."

        weather_note = weather_item.get("condition", "Forecast unavailable for this date.")
        if rain_chance is not None:
            weather_note += f"; precipitation chance {rain_chance}%."
        itinerary.append(
            {
                "date": day_date.isoformat(),
                "theme": first.get("interest", "Explore"),
                "morning": morning,
                "afternoon": afternoon,
                "evening": evening,
                "weather_note": weather_note,
            }
        )
    return itinerary


def _itinerary_node(state: TripState) -> dict[str, Any]:
    curated = _curated_itinerary(state)
    if not os.getenv("OPENAI_API_KEY"):
        return {"itinerary": curated}

    try:
        model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.2,
        ).with_structured_output(ItineraryDraft)
        prompt = {
            "trip": state["trip"],
            "curated_activity_options": state.get("travel_knowledge", {}).get("activities", []),
            "weather": state.get("weather", []),
            "dates_to_cover": [day["date"] for day in curated],
            "constraints": [
                "Return exactly one itinerary entry for every requested date.",
                "Use curated activity options; do not invent specific events or closures.",
                "Respect weather notes and prefer indoor activities on high-rain days.",
                "Do not make visa claims or include unverified prices.",
            ],
        }
        draft = model.invoke(
            "Create a concise, practical trip schedule. Keep travel days light and use only "
            "provided activity options and weather facts.\n"
            + json.dumps(prompt, ensure_ascii=True)
        )
        generated = [day.model_dump() for day in draft.days]
        if {day.get("date") for day in generated} != {day["date"] for day in curated}:
            raise ValueError("The model returned different itinerary dates.")
        return {"itinerary": generated}
    except Exception as error:
        warnings = _append_warning(
            state.get("warnings", []),
            f"AI itinerary generation was unavailable ({error.__class__.__name__}); curated suggestions are shown.",
        )
        return {"itinerary": curated, "warnings": warnings}


def _compute_costs(state: TripState) -> dict[str, Any]:
    trip = state["trip"]
    exchange_rate = float(state.get("exchange_rate", 1.0))
    traveler_count = int(trip["travelers"])
    start = date.fromisoformat(trip["start_date"])
    end = date.fromisoformat(trip["end_date"])
    nights = (end - start).days
    travel_days = nights + 1
    hotel = state.get("selected_hotel")
    flight = state.get("selected_flight")
    flight_inr = float(flight["price_inr"]) if flight else 0.0
    hotel_inr = float(hotel["total_price_inr"]) if hotel else 0.0
    local_inr = LOCAL_COST_PER_PERSON_PER_DAY_INR * traveler_count * travel_days
    total_inr = flight_inr + hotel_inr + local_inr
    complete = bool(
        flight and flight.get("outbound") and flight.get("return") and hotel
    )

    def convert(amount: float) -> float:
        return round(amount * exchange_rate, 2)

    total = convert(total_inr)
    budget = float(trip["budget"])
    return {
        "currency": trip["currency"],
        "flights": convert(flight_inr),
        "hotel": convert(hotel_inr),
        "local_estimate": convert(local_inr),
        "total": total,
        "budget": budget,
        "within_budget": total <= budget if complete else None,
        "complete_estimate": complete,
        "rate_source": state.get("rate_source", "Unknown"),
        "rate_date": state.get("rate_date"),
        "nights": nights,
        "hotel_rooms": math.ceil(traveler_count / 2),
        "travelers": traveler_count,
        "exclusions": ["Travel insurance, meals beyond the local estimate, and unlisted fees are not priced."],
        "indicative": True,
    }


def _budget_node(state: TripState) -> dict[str, Any]:
    costs = _compute_costs(state)
    hotels = state.get("hotel_options", [])
    selected = state.get("selected_hotel")
    replans = list(state.get("replans", []))
    if costs["within_budget"] is False and len(replans) < MAX_REPLANS and hotels:
        cheapest = min(hotels, key=lambda hotel: hotel["total_price_inr"])
        current_price = selected.get("total_price_inr", float("inf")) if selected else float("inf")
        if cheapest["total_price_inr"] < current_price:
            replans.append(
                f"Budget exceeded at {costs['currency']} {costs['total']:,.2f}; retrying with {cheapest['name']}, the lowest-priced mock hotel."
            )
            return {"selected_hotel": cheapest, "replans": replans, "budget_feedback": "replan"}

    if costs["within_budget"] is None:
        warnings = _append_warning(
            state.get("warnings", []),
            "Budget status is not guaranteed because a complete round-trip flight and lodging estimate is unavailable.",
        )
        return {"costs": costs, "budget_feedback": "complete", "warnings": warnings}
    return {"costs": costs, "budget_feedback": "complete"}


def _route_after_budget(state: TripState) -> str:
    return "planner" if state.get("budget_feedback") == "replan" else "summary"


def _summary_node(state: TripState) -> dict[str, Any]:
    trip = state["trip"]
    knowledge = state.get("travel_knowledge", {})
    visa_note = knowledge.get("visa_note") or (
        "Visa eligibility was not determined. Verify requirements with the official embassy or immigration website for your passport nationality."
    )
    warnings = list(state.get("warnings", []))
    warnings = _append_warning(warnings, "Flight and hotel prices are mock, indicative estimates and are not live bookings.")
    warnings = _append_warning(warnings, "Verify visa rules with the official embassy or immigration authority before booking.")
    if not knowledge.get("guide"):
        warnings = _append_warning(warnings, "No curated destination guide is available for this destination yet.")

    result = {
        "destination": trip["destination"],
        "itinerary": state.get("itinerary", []),
        "flight_options": state.get("flight_options", []),
        "hotel_options": state.get("hotel_options", []),
        "selected_flight": state.get("selected_flight"),
        "selected_hotel": state.get("selected_hotel"),
        "costs": state.get("costs", {}),
        "visa_notes": {"summary": visa_note, "sources": knowledge.get("visa_sources", [])},
        "events": state.get("events", []),
        "weather": state.get("weather", []),
        "advisories": list(state.get("advisories", [])) + list(knowledge.get("guide_sources", [])),
        "warnings": warnings,
        "replans": state.get("replans", []),
    }
    insurance = knowledge.get("insurance", {})
    if insurance.get("note"):
        result["warnings"].append(insurance["note"])
    if insurance.get("source"):
        result["advisories"].append(insurance["source"])
    return {"result": result}


def _build_graph():
    graph = StateGraph(TripState)
    graph.add_node("planner", _planner_node)
    graph.add_node("research", _research_node)
    graph.add_node("booking", _booking_node)
    graph.add_node("itinerary", _itinerary_node)
    graph.add_node("budget", _budget_node)
    graph.add_node("summary", _summary_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "research")
    graph.add_edge("research", "booking")
    graph.add_edge("booking", "itinerary")
    graph.add_edge("itinerary", "budget")
    graph.add_conditional_edges("budget", _route_after_budget, {"planner": "planner", "summary": "summary"})
    graph.add_edge("summary", END)
    return graph.compile()


TRIP_GRAPH = _build_graph()


async def plan_trip(trip: dict[str, Any]) -> dict[str, Any]:
    """Validate the submitted trip and return the planner result dictionary."""
    required = ("origin", "destination", "start_date", "end_date", "budget", "currency", "nationality", "travelers")
    missing = [field for field in required if trip.get(field) in (None, "")]
    if missing:
        raise ValueError("Missing required trip details: " + ", ".join(missing))

    start = date.fromisoformat(str(trip["start_date"]))
    end = date.fromisoformat(str(trip["end_date"]))
    if end <= start:
        raise ValueError("Return date must be after departure date.")
    if (end - start).days > 30:
        raise ValueError("Trips longer than 30 nights are not supported by this demo planner.")
    if float(trip["budget"]) <= 0:
        raise ValueError("Budget must be greater than zero.")
    if int(trip["travelers"]) < 1:
        raise ValueError("At least one traveller is required.")

    normalized = dict(trip)
    normalized.update(
        {
            "origin": str(trip["origin"]).strip(),
            "destination": str(trip["destination"]).strip(),
            "currency": str(trip["currency"]).upper(),
            "nationality": str(trip["nationality"]).strip(),
            "travelers": int(trip["travelers"]),
            "budget": float(trip["budget"]),
            "interests": list(trip.get("interests", [])),
        }
    )
    if normalized["currency"] not in {"INR", "USD", "EUR", "SGD", "THB", "AED"}:
        raise ValueError("Supported currencies are INR, USD, EUR, SGD, THB, and AED.")

    final_state = await TRIP_GRAPH.ainvoke(
        {"trip": normalized, "warnings": [], "advisories": [], "replans": []}
    )
    return final_state["result"]

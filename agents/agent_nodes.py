"""
Agent definitions for SmartTrip AI.

Each agent is a function that takes TripState and returns updated state.
Agents use OpenAI GPT-4o-mini via LangChain for reasoning, with full resilient
offline fallbacks using the local RAG knowledge base, Open-Meteo weather API,
Frankfurter currency API, and local mock flight/hotel databases when OpenAI quota
is unavailable or exhausted.
"""

import os
import json
import re
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.state import TripState
from tools.tool_wrappers import (
    get_weather_forecast,
    search_web,
    search_flights,
    search_hotels,
    convert_currency,
    lookup_visa_policy,
    lookup_destination_guide,
    lookup_travel_insurance,
)

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================
# LLM SETUP
# ============================================================

def get_llm():
    """Get the configured LLM instance or None if not configured/available."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=api_key,
            max_retries=1,
            timeout=15,
        )
    except Exception:
        return None


# ============================================================
# HEURISTIC EXTRACTION HELPERS (Resilient Offline Fallback)
# ============================================================

def _heuristic_parse_trip(query: str, current_state: dict) -> dict:
    """Extract trip parameters when LLM is unavailable or quota is exhausted."""
    q_lower = query.lower() if query else ""
    
    # 1. Destination
    dest = current_state.get("destination")
    known_destinations = [
        "Singapore", "Bangkok", "Dubai", "Munnar", "Goa", "Tokyo",
        "Paris", "London", "Bali", "Phuket", "New York", "Rome",
        "Thailand", "France", "Japan", "UAE", "Vietnam", "Malaysia"
    ]
    if not dest:
        for kd in known_destinations:
            if kd.lower() in q_lower:
                dest = kd
                break
    if not dest:
        m = re.search(r'\b(?:to|visit|in)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)', query)
        dest = m.group(1) if m else "Singapore"

    # Map country to primary city if needed
    if dest.lower() == "thailand":
        dest_city = "Bangkok"
    elif dest.lower() == "uae":
        dest_city = "Dubai"
    elif dest.lower() == "france":
        dest_city = "Paris"
    elif dest.lower() == "japan":
        dest_city = "Tokyo"
    else:
        dest_city = dest

    # 2. Origin
    origin = current_state.get("origin")
    if not origin:
        m = re.search(r'\bfrom\s+([A-Za-z]+)', query, re.IGNORECASE)
        origin = m.group(1).capitalize() if m else "Delhi"

    # 3. Days
    num_days = current_state.get("num_days")
    if not num_days:
        m = re.search(r'(\d+)\s*(?:-| )?day', query, re.IGNORECASE)
        num_days = int(m.group(1)) if m else 4

    # 4. Dates
    start_date = current_state.get("start_date")
    if not start_date:
        if "december" in q_lower:
            start_date = "2026-12-10"
        elif "next week" in q_lower:
            start_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        else:
            start_date = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")

    end_date = current_state.get("end_date")
    if not end_date:
        try:
            s_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_date = (s_dt + timedelta(days=num_days)).strftime("%Y-%m-%d")
        except Exception:
            end_date = "2026-12-14"

    # 5. Budget
    budget = current_state.get("budget_inr")
    if not budget:
        if "lakh" in q_lower:
            m = re.search(r'(\d+(?:\.\d+)?)\s*lakh', query, re.IGNORECASE)
            budget = float(m.group(1)) * 100000.0 if m else 100000.0
        else:
            m = re.search(r'(?:rs\.?|inr|₹|under)\s*([0-9,]+)', query, re.IGNORECASE)
            if m:
                clean_num = m.group(1).replace(",", "")
                budget = float(clean_num) if clean_num else 80000.0
            else:
                budget = 80000.0

    return {
        "destination": dest_city,
        "origin": origin,
        "start_date": start_date,
        "end_date": end_date,
        "num_days": num_days,
        "budget_inr": budget,
        "currency": current_state.get("currency", "INR"),
        "num_travellers": current_state.get("num_travellers", 1),
        "plan": f"Plan a {num_days}-day trip to {dest_city} from {origin} within budget ₹{budget:,.0f}.",
        "missing_info": [],
    }


def _generate_fallback_itinerary(
    destination: str,
    num_days: int,
    start_date: str,
    budget_inr: float,
    destination_guide: Optional[dict],
    weather_forecast: Optional[dict],
    budget_feedback: Optional[str] = "",
) -> str:
    """Generate a rich, day-by-day itinerary using local guide data and weather forecast."""
    attractions = []
    tips = []
    food_info = {}
    if destination_guide and isinstance(destination_guide, dict):
        attractions = destination_guide.get("top_attractions", [])
        tips = destination_guide.get("tips", [])
        food_info = destination_guide.get("food_costs", {})

    forecasts = []
    if weather_forecast and isinstance(weather_forecast, dict):
        forecasts = weather_forecast.get("forecast", [])

    lines = [f"### 🗓️ {num_days}-Day Curated Itinerary for {destination}\n"]
    if budget_feedback:
        lines.append(f"> ⚠️ **Budget Optimization Note:** {budget_feedback}\n")

    attr_idx = 0
    num_attrs = len(attractions)

    for d in range(1, num_days + 1):
        # Determine date and weather for the day
        day_date = ""
        weather_str = ""
        rain_warning = False
        if d - 1 < len(forecasts):
            f = forecasts[d - 1]
            day_date = f.get("date", f"Day {d}")
            cond = f.get("condition", "Partly Cloudy")
            t_min = f.get("min_temp_c", 24)
            t_max = f.get("max_temp_c", 31)
            rain_mm = f.get("precipitation_mm", 0)
            weather_str = f"🌤️ **Forecast:** {cond} ({t_min}°C – {t_max}°C)"
            if rain_mm > 0:
                weather_str += f" | 🌧️ Rain expected: {rain_mm}mm (keep indoor options handy)"
                rain_warning = True
        else:
            day_date = f"Day {d}"
            weather_str = "🌤️ **Forecast:** Pleasant weather expected"

        lines.append(f"#### 📍 Day {d}: {day_date}")
        lines.append(f"{weather_str}\n")

        # Morning
        if attr_idx < num_attrs:
            a1 = attractions[attr_idx]
            attr_idx += 1
            lines.append(f"- **Morning (09:00 - 12:30):** Visit **{a1['name']}** ({a1.get('type', 'Attraction')}). Time required: {a1.get('time_needed', '2-3 hours')}. Entry fee: ₹{a1.get('entry_fee_inr', 0):,}.")
        else:
            lines.append(f"- **Morning (09:00 - 12:30):** Scenic city walk, neighborhood exploration, and iconic landmark photography.")

        # Afternoon & Lunch
        hawker_price = food_info.get("hawker_centre_meal_inr", food_info.get("street_food_meal_inr", 350))
        lines.append(f"- **Lunch (12:30 - 14:00):** Savor authentic local delicacies at popular food centres/cafes (approx ₹{hawker_price} per person).")

        if attr_idx < num_attrs:
            a2 = attractions[attr_idx]
            attr_idx += 1
            indoor_note = " (Perfect indoor venue)" if rain_warning and a2.get("type") in ["Landmark", "Shopping", "Cultural"] else ""
            lines.append(f"- **Afternoon (14:00 - 17:30):** Explore **{a2['name']}** ({a2.get('type', 'Attraction')}){indoor_note}. Entry fee: ₹{a2.get('entry_fee_inr', 0):,}.")
        else:
            lines.append(f"- **Afternoon (14:00 - 17:30):** Visit local art galleries, souvenir bazaars, or relax at a nearby garden/cafe.")

        # Evening & Dinner
        rest_price = food_info.get("restaurant_meal_inr", 1000)
        if attr_idx < num_attrs:
            a3 = attractions[attr_idx]
            attr_idx += 1
            lines.append(f"- **Evening (18:00 - 21:30):** Experience **{a3['name']}** ({a3.get('type', 'Nightlife/Sight')}) followed by dinner (approx ₹{rest_price} per person).")
        else:
            lines.append(f"- **Evening (18:00 - 21:30):** Evening river promenade or vibrant night market walk, followed by dinner and drinks.")

        lines.append("")

    if tips:
        lines.append("#### 💡 Destination Insider Tips:")
        for t in tips[:4]:
            lines.append(f"- {t}")

    return "\n".join(lines)


def _generate_fallback_weather(weather_data: dict, destination: str, num_days: int) -> str:
    """Analyze Open-Meteo forecast and generate practical travel adjustments."""
    if not weather_data or "forecast" not in weather_data or not weather_data["forecast"]:
        return f"Weather for {destination} is generally warm and humid. Pack light, breathable fabrics and carry rain protection."

    forecasts = weather_data["forecast"][:num_days]
    rainy_days = [f for f in forecasts if f.get("precipitation_mm", 0) > 0]
    temps_max = [f.get("max_temp_c", 30) for f in forecasts if f.get("max_temp_c") is not None]
    avg_max = sum(temps_max) / len(temps_max) if temps_max else 30

    adjustments = [f"**Weather Analysis for {destination} over {len(forecasts)} Days:**"]
    if rainy_days:
        dates_str = ", ".join([r.get("date", "some days") for r in rainy_days])
        adjustments.append(f"- 🌧️ **Precipitation Alert:** Showers predicted on {dates_str}. Keep a compact umbrella or poncho with you, and schedule indoor attractions (museums, malls, indoor complexes) during peak rainfall hours.")
    else:
        adjustments.append("- ☀️ **Clear Conditions:** Mostly dry weather forecast during your travel window. Ideal for outdoor sightseeing and walking tours.")

    if avg_max > 28:
        adjustments.append(f"- 🌡️ **Temperature & Sun Protection:** Highs around {avg_max:.1f}°C. Carry sunscreen, UV sunglasses, and stay hydrated throughout daytime activities.")
    elif avg_max < 18:
        adjustments.append(f"- 🧥 **Layering Recommended:** Cooler temperatures expected (highs around {avg_max:.1f}°C). Pack warm jackets or fleece.")

    adjustments.append("- 👟 **Footwear:** Pack comfortable, water-resistant walking shoes for city and attraction trails.")
    return "\n".join(adjustments)


def _generate_fallback_summary(
    destination: str,
    origin: str,
    num_days: int,
    start_date: str,
    end_date: str,
    budget_inr: float,
    itinerary: str,
    cost_breakdown: dict,
    visa_info: Optional[dict],
    weather_adjustments: str,
    events_and_advisories: str,
    flight_options: list,
    hotel_options: list,
) -> str:
    """Synthesize complete, polished final travel itinerary report."""
    within_budget = cost_breakdown.get("within_budget", True) if cost_breakdown else True
    total_cost = cost_breakdown.get("grand_total_inr", budget_inr) if cost_breakdown else budget_inr

    # Flights text
    flights_sec = ""
    if flight_options:
        flights_sec = "\n### ✈️ Recommended Flight Options\n"
        flights_sec += "| Airline | Flight ID | Route | Price (One Way) |\n|---|---|---|---|\n"
        for f in flight_options[:3]:
            flights_sec += f"| {f.get('airline', 'Flight')} | `{f.get('flight_id', 'N/A')}` | {f.get('departure', origin)} ➔ {f.get('arrival', destination)} | ₹{f.get('price_inr', 0):,} |\n"

    # Hotels text
    hotels_sec = ""
    if hotel_options:
        hotels_sec = "\n### 🏨 Recommended Hotel Options\n"
        hotels_sec += "| Hotel | Rating | Location | Price / Night |\n|---|---|---|---|\n"
        for h in hotel_options[:3]:
            hotels_sec += f"| {h.get('name', 'Hotel')} | ⭐ {h.get('rating', 'N/A')} | {h.get('location', destination)} | ₹{h.get('price_per_night_inr', 0):,} |\n"

    # Visa text
    visa_sec = ""
    if visa_info and isinstance(visa_info, dict) and "country" in visa_info:
        req_icon = "✅" if not visa_info.get("visa_required") else "🛂"
        visa_sec = f"""
### 🛂 Visa & Entry Regulations ({visa_info.get('country', destination)})
- **Visa Required:** {req_icon} {'Yes' if visa_info.get('visa_required') else 'No (Visa-Free / VoA Available)'}
- **Visa Type:** {visa_info.get('visa_type', 'N/A')}
- **Permitted Duration:** {visa_info.get('duration_allowed', 'N/A')}
- **Processing Time:** {visa_info.get('processing_time', 'N/A')}
- **Consular Fee:** ₹{visa_info.get('fee_inr', 0):,}
- **Key Documents:** {', '.join(visa_info.get('documents_required', [])[:4])}
- **Official Portal / Source:** [{visa_info.get('official_source', 'Official Embassy')}]({visa_info.get('official_source', '#')})

> ⚠️ *Disclaimer: {visa_info.get('disclaimer', 'Visa policies change frequently. Verify with official embassy sources prior to travel.')}*
"""

    # Cost breakdown table
    cost_table = ""
    if cost_breakdown:
        diff = cost_breakdown.get("difference_inr", 0)
        status_badge = "🎉 Within Budget" if within_budget else f"⚠️ Over Budget by ₹{abs(diff):,.0f}"
        cost_table = f"""
### 💰 Financial Breakdown (INR)
| Expense Category | Estimated Cost (₹) |
|---|---|
| ✈️ Flights (Round-Trip) | ₹{cost_breakdown.get('flights_round_trip', 0):,.0f} |
| 🏨 Accommodation ({max(num_days - 1, 1)} nights) | ₹{cost_breakdown.get('hotel_total', 0):,.0f} |
| 🍽️ Meals & Dining | ₹{cost_breakdown.get('food_total', 0):,.0f} |
| 🎟️ Sightseeing & Activities | ₹{cost_breakdown.get('activities_total', 0):,.0f} |
| 🚕 Local Commute & Transit | ₹{cost_breakdown.get('local_transport_total', 0):,.0f} |
| 🧰 Contingency & Misc | ₹{cost_breakdown.get('miscellaneous', 0):,.0f} |
| **Total Estimated Cost** | **₹{total_cost:,.0f}** |
| **Allocated Budget** | **₹{budget_inr:,.0f}** |
| **Budget Status** | **{status_badge}** |
"""

    report = f"""# 🌏 SmartTrip AI Itinerary: {destination}
**Origin:** {origin} | **Travel Dates:** {start_date} to {end_date} ({num_days} Days) | **Budget:** ₹{budget_inr:,.0f}

---

{itinerary}

---

{cost_table}

{flights_sec}

{hotels_sec}

---

{visa_sec}

### 🌤️ Weather & Packing Guidance
{weather_adjustments}

---

### 🛡️ Smart Travel Safeguards & Disclaimers
1. **Indicative Pricing:** All airfares, room tariffs, and entry fees are estimated based on real-time market averages and mock provider data. Actual rates fluctuate with booking dates.
2. **Official Visa Verification:** Always confirm current entry requirements, passport validity mandates (minimum 6 months), and health declarations directly with the official embassy.
3. **Travel Insurance:** We strongly recommend securing international travel insurance to cover unexpected cancellations, delays, or medical emergencies.
"""
    return report.strip()


# ============================================================
# AGENT 1: PLANNER AGENT
# ============================================================

def planner_agent(state: TripState) -> dict:
    """
    The Planner Agent:
    - Parses the user query to extract trip parameters
    - Identifies missing information
    - Creates an execution plan
    - Fetches visa and destination info from RAG
    """
    user_query = state.get("user_query", "")
    replan_count = state.get("replan_count", 0)
    budget_feedback = state.get("budget_feedback", "")

    replan_context = ""
    if replan_count and replan_count > 0 and budget_feedback:
        replan_context = f"""
⚠️ REPLANNING REQUIRED (attempt {replan_count}):
The budget was exceeded in the previous plan. Here's the budget feedback:
{budget_feedback}

You MUST adjust the plan to fit within the budget.
"""

    system_prompt = f"""You are the Planner Agent for SmartTrip AI, an intelligent travel planner.
Parse the user's travel request and extract parameters into JSON:
- destination (city/country)
- origin (departure city, default: "Delhi")
- start_date (YYYY-MM-DD format)
- end_date (YYYY-MM-DD format)
- num_days (number of days)
- budget_inr (total budget in INR)
- currency (default: INR)
- num_travellers (default: 1)
- missing_info (list)
- plan (execution plan description)

{replan_context}

Respond in this exact JSON format:
{{
    "destination": "city name",
    "origin": "origin city or Delhi",
    "start_date": "YYYY-MM-DD",
    "end_date": "YYYY-MM-DD",
    "num_days": 4,
    "budget_inr": 80000,
    "currency": "INR",
    "num_travellers": 1,
    "missing_info": [],
    "plan": "Execution plan"
}}
"""

    parsed = {}
    llm = get_llm()
    if llm:
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_query),
            ]
            response = llm.invoke(messages)
            response_text = response.content
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                parsed = json.loads(response_text[json_start:json_end])
        except Exception as e:
            logger.warning(f"Planner LLM call failed ({e}); falling back to heuristic parsing.")
            parsed = {}

    if not parsed or not parsed.get("destination"):
        parsed = _heuristic_parse_trip(user_query, state)

    # Fetch RAG knowledge
    destination = parsed.get("destination") or state.get("destination") or "Singapore"
    visa_info = None
    destination_guide = None

    if destination:
        try:
            visa_result = lookup_visa_policy.invoke({"country": destination})
            visa_info = visa_result
        except Exception:
            pass

        try:
            guide_result = lookup_destination_guide.invoke({"city": destination})
            destination_guide = guide_result
        except Exception:
            pass

    return {
        "destination": destination,
        "origin": parsed.get("origin") or state.get("origin") or "Delhi",
        "start_date": parsed.get("start_date") or state.get("start_date") or "2026-12-10",
        "end_date": parsed.get("end_date") or state.get("end_date") or "2026-12-14",
        "num_days": parsed.get("num_days") or state.get("num_days") or 4,
        "budget_inr": parsed.get("budget_inr") or state.get("budget_inr") or 80000.0,
        "currency": parsed.get("currency") or state.get("currency") or "INR",
        "num_travellers": parsed.get("num_travellers") or state.get("num_travellers") or 1,
        "plan": parsed.get("plan", f"Plan a complete {destination} itinerary"),
        "missing_info": parsed.get("missing_info", []),
        "visa_info": visa_info,
        "destination_guide": destination_guide,
        "replan_count": replan_count,
        "messages": [AIMessage(content=f"📋 **Planner:** Parsed request — Destination: {destination}, Days: {parsed.get('num_days', 4)}, Budget: ₹{parsed.get('budget_inr', 80000):,.0f}")],
    }


# ============================================================
# AGENT 2: ITINERARY AGENT
# ============================================================

def itinerary_agent(state: TripState) -> dict:
    """
    The Itinerary Agent:
    - Builds a day-by-day plan using RAG guides
    - Searches for local events via Tavily
    - Creates activity schedules per day
    """
    destination = state.get("destination", "Singapore")
    num_days = state.get("num_days", 4)
    start_date = state.get("start_date", "2026-12-10")
    budget_inr = state.get("budget_inr", 80000.0)
    destination_guide = state.get("destination_guide")
    weather_forecast = state.get("weather_forecast")
    budget_feedback = state.get("budget_feedback", "")

    # Search for local events via Tavily if available
    events_info = ""
    try:
        events_result = search_web.invoke({
            "query": f"events festivals things to do in {destination} {start_date} 2026",
            "max_results": 3,
        })
        if "results" in events_result:
            events_info = "\n".join(
                [f"- {r['title']}: {r['content'][:200]}" for r in events_result["results"]]
            )
    except Exception:
        events_info = "Local events info available via city guide."

    itinerary = None
    llm = get_llm()
    if llm:
        try:
            guide_info = ""
            if destination_guide and isinstance(destination_guide, dict):
                attractions = destination_guide.get("top_attractions", [])
                if attractions:
                    guide_info = "Top Attractions:\n" + "\n".join(
                        [f"- {a['name']} ({a['type']}) — Entry: ₹{a.get('entry_fee_inr', 0)}, Time: {a.get('time_needed', 'N/A')}"
                         for a in attractions]
                    )
                tips = destination_guide.get("tips", [])
                if tips:
                    guide_info += "\n\nLocal Tips:\n" + "\n".join([f"- {t}" for t in tips])

            weather_context = ""
            if weather_forecast and isinstance(weather_forecast, dict):
                forecasts = weather_forecast.get("forecast", [])
                if forecasts:
                    weather_context = "Weather Forecast:\n" + "\n".join(
                        [f"- {f['date']}: {f.get('condition', 'N/A')}, {f.get('min_temp_c', '?')}°C - {f.get('max_temp_c', '?')}°C, Rain: {f.get('precipitation_mm', 0)}mm"
                         for f in forecasts[:num_days]]
                    )

            budget_context = f"\n⚠️ BUDGET CONSTRAINT: {budget_feedback}" if budget_feedback else ""

            system_prompt = f"""You are the Itinerary Agent for SmartTrip AI.
Create a detailed day-by-day itinerary for a {num_days}-day trip to {destination}.
Starting date: {start_date}
Budget: ₹{budget_inr:,.0f} total

{guide_info}
{weather_context}
{events_info}
{budget_context}

Rules:
1. Create a DETAILED day-by-day plan with morning, afternoon, and evening activities.
2. Include estimated costs in INR for activities and meals.
3. Factor in weather.
4. Format clearly with Day 1, Day 2, etc.
"""
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Plan a {num_days}-day itinerary for {destination}"),
            ]
            response = llm.invoke(messages)
            itinerary = response.content
        except Exception as e:
            logger.warning(f"Itinerary LLM call failed ({e}); using rule-based itinerary generator.")
            itinerary = None

    if not itinerary:
        itinerary = _generate_fallback_itinerary(
            destination=destination,
            num_days=num_days,
            start_date=start_date,
            budget_inr=budget_inr,
            destination_guide=destination_guide,
            weather_forecast=weather_forecast,
            budget_feedback=budget_feedback,
        )

    return {
        "itinerary": itinerary,
        "events_and_advisories": events_info,
        "messages": [AIMessage(content=f"🗓️ **Itinerary Agent:** Generated {num_days}-day itinerary for {destination}")],
    }


# ============================================================
# AGENT 3: BUDGET AGENT
# ============================================================

def budget_agent(state: TripState) -> dict:
    """
    The Budget Agent:
    - Searches flights and hotels via tools
    - Calculates total trip cost
    - Checks against user's budget
    - Triggers re-planning if budget exceeded
    """
    destination = state.get("destination", "Singapore")
    origin = state.get("origin", "Delhi")
    start_date = state.get("start_date", "2026-12-10")
    num_days = state.get("num_days", 4)
    budget_inr = state.get("budget_inr", 80000.0)
    num_travellers = state.get("num_travellers", 1)
    replan_count = state.get("replan_count", 0)

    # Search flights
    flight_options = []
    try:
        flight_result = search_flights.invoke({
            "origin": origin,
            "destination": destination,
            "date": start_date,
        })
        flight_options = flight_result.get("flights", [])
    except Exception:
        pass

    if not flight_options:
        try:
            flight_result = search_flights.invoke({
                "origin": origin,
                "destination": destination,
            })
            flight_options = flight_result.get("flights", [])
        except Exception:
            pass

    # Search hotels
    hotel_options = []
    try:
        hotel_result = search_hotels.invoke({"city": destination})
        hotel_options = hotel_result.get("hotels", [])
    except Exception:
        pass

    # Calculate costs
    cheapest_flight = min(
        (f["price_inr"] for f in flight_options),
        default=12000,
    )
    cheapest_hotel = min(
        (h["price_per_night_inr"] for h in hotel_options),
        default=3500,
    )

    num_nights = max(num_days - 1, 1)

    flight_cost = cheapest_flight * 2 * num_travellers  # Round trip
    hotel_cost = cheapest_hotel * num_nights * num_travellers
    food_cost = 1800 * num_days * num_travellers
    activities_cost = 2500 * num_days * num_travellers
    transport_cost = 800 * num_days * num_travellers
    misc_cost = 2000 * num_travellers

    total_cost = (
        flight_cost + hotel_cost + food_cost
        + activities_cost + transport_cost + misc_cost
    )

    budget_ok = total_cost <= budget_inr

    cost_breakdown = {
        "flights_round_trip": flight_cost,
        "hotel_total": hotel_cost,
        "food_total": food_cost,
        "activities_total": activities_cost,
        "local_transport_total": transport_cost,
        "miscellaneous": misc_cost,
        "grand_total_inr": total_cost,
        "budget_inr": budget_inr,
        "within_budget": budget_ok,
        "difference_inr": budget_inr - total_cost,
    }

    budget_feedback = ""
    if not budget_ok:
        over_by = total_cost - budget_inr
        budget_feedback = (
            f"Budget EXCEEDED by ₹{over_by:,.0f}. "
            f"Total estimated cost: ₹{total_cost:,.0f} vs Budget: ₹{budget_inr:,.0f}. "
            f"Need to adjust flight/hotel tier or trip duration."
        )

    status_emoji = "✅" if budget_ok else "⚠️"
    msg = (
        f"💰 **Budget Agent:** {status_emoji} "
        f"Total: ₹{total_cost:,.0f} | Budget: ₹{budget_inr:,.0f} | "
        f"{'Within budget! 🎉' if budget_ok else f'Over by ₹{total_cost - budget_inr:,.0f}'}"
    )

    return {
        "flight_options": flight_options,
        "hotel_options": hotel_options,
        "cost_breakdown": cost_breakdown,
        "total_cost_inr": total_cost,
        "budget_ok": budget_ok,
        "budget_feedback": budget_feedback,
        "replan_count": (replan_count or 0) + (0 if budget_ok else 1),
        "messages": [AIMessage(content=msg)],
    }


# ============================================================
# AGENT 4: WEATHER AGENT
# ============================================================

def weather_agent(state: TripState) -> dict:
    """
    The Weather Agent:
    - Fetches weather forecasts for the destination via Open-Meteo
    - Suggests adjustments to outdoor activities
    """
    destination = state.get("destination", "Singapore")
    num_days = state.get("num_days", 4)
    itinerary = state.get("itinerary", "")

    weather_data = {}
    try:
        weather_data = get_weather_forecast.invoke({
            "city": destination,
            "num_days": min(num_days, 16),
        })
    except Exception:
        weather_data = {"error": "Could not fetch weather data"}

    weather_adjustments = None
    llm = get_llm()
    if llm and weather_data and "forecast" in weather_data:
        try:
            forecast_text = "\n".join(
                [f"- {f['date']}: {f.get('condition', 'N/A')}, "
                 f"{f.get('min_temp_c', '?')}°C - {f.get('max_temp_c', '?')}°C, "
                 f"Rain: {f.get('precipitation_mm', 0)}mm"
                 for f in weather_data["forecast"][:num_days]]
            )
            system_prompt = f"""You are the Weather Agent for SmartTrip AI.
Given weather forecast for {destination}:
{forecast_text}

Provide brief, practical weather-based adjustments (rain flags, packing, outdoor swaps).
"""
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content="What weather adjustments should I make?"),
            ])
            weather_adjustments = response.content
        except Exception as e:
            logger.warning(f"Weather LLM call failed ({e}); using rule-based weather analyzer.")
            weather_adjustments = None

    if not weather_adjustments:
        weather_adjustments = _generate_fallback_weather(weather_data, destination, num_days)

    return {
        "weather_forecast": weather_data,
        "weather_adjustments": weather_adjustments,
        "messages": [AIMessage(content=f"🌤️ **Weather Agent:** Checked forecast for {destination}")],
    }


# ============================================================
# AGENT 5: BOOKING SUMMARY AGENT
# ============================================================

def booking_summary_agent(state: TripState) -> dict:
    """
    The Booking Summary Agent:
    - Compiles everything into a final, presentation-ready itinerary
    - Includes cost breakdown, visa notes, weather advisories
    """
    destination = state.get("destination", "Singapore")
    origin = state.get("origin", "Delhi")
    num_days = state.get("num_days", 4)
    start_date = state.get("start_date", "2026-12-10")
    end_date = state.get("end_date", "2026-12-14")
    budget_inr = state.get("budget_inr", 80000.0)
    itinerary = state.get("itinerary", "Itinerary generated.")
    cost_breakdown = state.get("cost_breakdown", {})
    visa_info = state.get("visa_info", {})
    weather_adjustments = state.get("weather_adjustments", "")
    events_and_advisories = state.get("events_and_advisories", "")
    flight_options = state.get("flight_options", [])
    hotel_options = state.get("hotel_options", [])

    final_itinerary = None
    llm = get_llm()
    if llm:
        try:
            visa_req = "Yes" if visa_info and visa_info.get("visa_required") else "No"
            visa_text = f"Visa Required: {visa_req}, Type: {visa_info.get('visa_type', 'N/A') if visa_info else 'N/A'}, Fee: ₹{visa_info.get('fee_inr', 'N/A') if visa_info else 'N/A'}"
            cost_text = f"Total: ₹{cost_breakdown.get('grand_total_inr', 0):,.0f} vs Budget: ₹{budget_inr:,.0f}"

            system_prompt = f"""You are the Booking Summary Agent for SmartTrip AI.
Compile a final, comprehensive trip summary for {destination} from {origin} ({num_days} days, {start_date} to {end_date}).
Include:
1. Day-by-day itinerary
2. Cost breakdown table in INR
3. Recommended flights and hotels
4. Visa requirements and official source
5. Weather adjustments & tips
6. Important disclaimers (indicative prices, embassy verification)

Itinerary:
{itinerary}

Costs:
{cost_text}

Visa:
{visa_text}

Weather:
{weather_adjustments}
"""
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content="Generate the final trip summary."),
            ])
            final_itinerary = response.content
        except Exception as e:
            logger.warning(f"Booking summary LLM call failed ({e}); using structured compiler.")
            final_itinerary = None

    if not final_itinerary:
        final_itinerary = _generate_fallback_summary(
            destination=destination,
            origin=origin,
            num_days=num_days,
            start_date=start_date,
            end_date=end_date,
            budget_inr=budget_inr,
            itinerary=itinerary,
            cost_breakdown=cost_breakdown,
            visa_info=visa_info,
            weather_adjustments=weather_adjustments,
            events_and_advisories=events_and_advisories,
            flight_options=flight_options,
            hotel_options=hotel_options,
        )

    return {
        "final_itinerary": final_itinerary,
        "messages": [AIMessage(content=f"📄 **Booking Summary:** Final itinerary for {destination} is ready! ✅")],
    }

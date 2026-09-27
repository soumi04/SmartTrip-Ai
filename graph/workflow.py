"""
LangGraph workflow for SmartTrip AI.

Implements the planner-executor pattern with conditional re-planning:
  Planner → Weather → Itinerary → Budget → [re-plan if over budget] → Booking Summary

The Budget Agent can trigger a conditional edge back to the Planner
if the budget is exceeded (up to 2 re-plan attempts).
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from langgraph.graph import StateGraph, END

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.state import TripState
from agents.agent_nodes import (
    planner_agent,
    itinerary_agent,
    budget_agent,
    weather_agent,
    booking_summary_agent,
)


# ============================================================
# CONDITIONAL EDGE: CHECK BUDGET
# ============================================================

def should_replan(state: TripState) -> str:
    """
    Conditional edge after Budget Agent:
    - If budget is OK → proceed to Booking Summary
    - If budget exceeded and replan_count < 2 → go back to Planner
    - If budget exceeded but already replanned twice → proceed anyway
    """
    budget_ok = state.get("budget_ok", True)
    replan_count = state.get("replan_count", 0)

    if budget_ok:
        return "booking_summary"
    elif replan_count is not None and replan_count < 2:
        return "planner"
    else:
        return "booking_summary"


# ============================================================
# BUILD THE GRAPH
# ============================================================

def build_trip_graph() -> StateGraph:
    """
    Build and compile the LangGraph workflow.

    Flow:
        START → Planner → Weather → Itinerary → Budget
                ↑                                  ↓
                └──── (if over budget) ←───────────┘
                                                   ↓
                                          Booking Summary → END
    """
    graph = StateGraph(TripState)

    graph.add_node("planner", planner_agent)
    graph.add_node("weather", weather_agent)
    graph.add_node("itinerary", itinerary_agent)
    graph.add_node("budget", budget_agent)
    graph.add_node("booking_summary", booking_summary_agent)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "weather")
    graph.add_edge("weather", "itinerary")
    graph.add_edge("itinerary", "budget")

    graph.add_conditional_edges(
        "budget",
        should_replan,
        {
            "booking_summary": "booking_summary",
            "planner": "planner",
        },
    )

    graph.add_edge("booking_summary", END)

    return graph.compile()


# ============================================================
# FORMAT RESULT HELPER (For both Streamlit Form and Chat views)
# ============================================================

def format_trip_output(final_state: dict) -> dict:
    """Format raw LangGraph output state into structured dictionary for Streamlit renderers."""
    cost_breakdown = final_state.get("cost_breakdown") or {}
    destination = final_state.get("destination", "Singapore")
    origin = final_state.get("origin", "Delhi")
    num_days = final_state.get("num_days", 4)
    start_date = final_state.get("start_date", "2026-12-10")
    currency = final_state.get("currency", "INR")
    budget_val = final_state.get("budget_inr") or cost_breakdown.get("budget_inr", 80000.0)
    total_cost = cost_breakdown.get("grand_total_inr", budget_val)

    flights_cost = cost_breakdown.get("flights_round_trip", 24000.0)
    hotel_cost = cost_breakdown.get("hotel_total", 14000.0)
    local_est = (
        cost_breakdown.get("food_total", 0.0)
        + cost_breakdown.get("activities_total", 0.0)
        + cost_breakdown.get("local_transport_total", 0.0)
    )

    costs = {
        "flights": flights_cost,
        "hotel": hotel_cost,
        "local_estimate": local_est,
        "total": total_cost,
        "budget": budget_val,
        "within_budget": cost_breakdown.get("within_budget", total_cost <= budget_val),
        "currency": currency,
        "rate_source": "Frankfurter API (live)",
        "rate_date": "Current",
    }

    destination_guide = final_state.get("destination_guide") or {}
    weather_forecast = final_state.get("weather_forecast") or {}
    attractions = destination_guide.get("top_attractions", []) if isinstance(destination_guide, dict) else []
    forecasts = weather_forecast.get("forecast", []) if isinstance(weather_forecast, dict) else []

    try:
        s_date = datetime.strptime(start_date, "%Y-%m-%d")
    except Exception:
        s_date = datetime.now()

    itinerary_days = []
    attr_idx = 0
    num_attrs = len(attractions)

    for d in range(1, num_days + 1):
        day_date = (s_date + timedelta(days=d - 1)).strftime("%b %d, %Y")
        weather_note = ""
        if d - 1 < len(forecasts):
            f = forecasts[d - 1]
            weather_note = f"{f.get('condition', 'Pleasant')}, {f.get('min_temp_c', 24)}°C - {f.get('max_temp_c', 31)}°C"
            if f.get("precipitation_mm", 0) > 0:
                weather_note += f" (Rain: {f.get('precipitation_mm')}mm - carry umbrella)"

        theme = "City Highlights"
        m_name = "Morning Sightseeing"
        a_name = "Afternoon Highlights"
        e_name = "Evening Dining & Leisure"

        if attr_idx < num_attrs:
            a = attractions[attr_idx]
            attr_idx += 1
            theme = a.get("name", "City Exploration")
            m_name = f"Visit {a['name']} ({a.get('type', 'Attraction')}). Time needed: {a.get('time_needed', '2-3 hrs')}. Entry fee: ₹{a.get('entry_fee_inr', 0):,}."

        if attr_idx < num_attrs:
            a = attractions[attr_idx]
            attr_idx += 1
            a_name = f"Explore {a['name']} ({a.get('type', 'Attraction')}). Time needed: {a.get('time_needed', '2-3 hrs')}. Entry fee: ₹{a.get('entry_fee_inr', 0):,}."

        if attr_idx < num_attrs:
            a = attractions[attr_idx]
            attr_idx += 1
            e_name = f"Head to {a['name']} ({a.get('type', 'Sight')}) followed by local dinner."

        itinerary_days.append({
            "date": day_date,
            "theme": theme,
            "morning": m_name,
            "afternoon": a_name,
            "evening": e_name,
            "weather_note": weather_note,
        })

    flight_options = final_state.get("flight_options", [])
    hotel_options = final_state.get("hotel_options", [])
    selected_flight = flight_options[0] if flight_options else None
    selected_hotel = hotel_options[0] if hotel_options else None

    visa_info = final_state.get("visa_info") or {}
    visa_notes = {
        "summary": f"Visa Required: {'Yes' if visa_info.get('visa_required') else 'No (Visa-Free / VoA Available)'}. Type: {visa_info.get('visa_type', 'N/A')}. Permitted stay: {visa_info.get('duration_allowed', 'N/A')}. Processing: {visa_info.get('processing_time', 'N/A')}. Fee: ₹{visa_info.get('fee_inr', 0):,}.",
        "sources": [{"Official Source": visa_info.get("official_source", "https://www.mfa.gov.sg")}],
    }

    events_items = [
        {"name": a.get("name"), "type": a.get("type"), "fee": f"₹{a.get('entry_fee_inr', 0):,}"}
        for a in attractions[:4]
    ]
    weather_items = [
        {"date": f.get("date"), "temp": f"{f.get('min_temp_c', 24)}°C - {f.get('max_temp_c', 31)}°C", "condition": f.get("condition", "Clear")}
        for f in forecasts[:num_days]
    ]

    warnings = []
    if cost_breakdown.get("budget_feedback"):
        warnings.append(cost_breakdown["budget_feedback"])

    replans = []
    if final_state.get("replan_count", 0) > 0:
        replans.append(f"Plan refined across {final_state['replan_count']} budget review cycles to fit within ₹{budget_val:,.0f}.")

    return {
        "final_itinerary": final_state.get("final_itinerary", ""),
        "costs": costs,
        "itinerary": itinerary_days,
        "selected_flight": selected_flight,
        "flight_options": flight_options,
        "selected_hotel": selected_hotel,
        "hotel_options": hotel_options,
        "visa_notes": visa_notes,
        "events": events_items,
        "weather": weather_items,
        "advisories": [f"Always verify passport validity and entry regulations before departing for {destination}."],
        "warnings": warnings,
        "replans": replans,
    }


# ============================================================
# RUN / PLAN TRIP FUNCTIONS
# ============================================================

def run_trip_planner(user_query: str) -> dict:
    """
    Run the complete trip planning workflow for a free-form text query.

    Args:
        user_query: The user's travel planning request.

    Returns:
        The final state containing the complete itinerary and all metadata.
    """
    graph = build_trip_graph()

    initial_state = {
        "messages": [],
        "user_query": user_query,
        "destination": None,
        "origin": None,
        "start_date": None,
        "end_date": None,
        "num_days": None,
        "budget_inr": None,
        "currency": "INR",
        "num_travellers": 1,
        "plan": None,
        "missing_info": None,
        "visa_info": None,
        "destination_guide": None,
        "itinerary": None,
        "flight_options": None,
        "hotel_options": None,
        "cost_breakdown": None,
        "total_cost_inr": None,
        "budget_ok": None,
        "budget_feedback": None,
        "weather_forecast": None,
        "weather_adjustments": None,
        "events_and_advisories": None,
        "final_itinerary": None,
        "replan_count": 0,
    }

    final_state = graph.invoke(initial_state)
    return format_trip_output(final_state)


async def plan_trip(trip_input: Any) -> dict:
    """
    Async interface expected by Streamlit form submission.
    Accepts either a dictionary of trip parameters or a query string.
    """
    graph = build_trip_graph()

    if isinstance(trip_input, dict):
        destination = trip_input.get("destination", "Singapore").strip()
        origin = trip_input.get("origin", "Delhi").strip()
        start_date = trip_input.get("start_date", "2026-12-10")
        end_date = trip_input.get("end_date", "2026-12-14")
        budget = float(trip_input.get("budget", 80000.0))
        currency = trip_input.get("currency", "INR")
        nationality = trip_input.get("nationality", "Indian")
        travelers = int(trip_input.get("travelers", 1))

        try:
            s_dt = datetime.strptime(start_date, "%Y-%m-%d")
            e_dt = datetime.strptime(end_date, "%Y-%m-%d")
            num_days = max((e_dt - s_dt).days, 1)
        except Exception:
            num_days = 4

        query = f"Plan a {num_days}-day trip to {destination} from {origin} starting {start_date} under {currency} {budget:,.0f} for {travelers} traveler(s)."

        initial_state = {
            "messages": [],
            "user_query": query,
            "destination": destination,
            "origin": origin,
            "start_date": start_date,
            "end_date": end_date,
            "num_days": num_days,
            "budget_inr": budget,
            "currency": currency,
            "num_travellers": travelers,
            "plan": None,
            "missing_info": None,
            "visa_info": None,
            "destination_guide": None,
            "itinerary": None,
            "flight_options": None,
            "hotel_options": None,
            "cost_breakdown": None,
            "total_cost_inr": None,
            "budget_ok": None,
            "budget_feedback": None,
            "weather_forecast": None,
            "weather_adjustments": None,
            "events_and_advisories": None,
            "final_itinerary": None,
            "replan_count": 0,
        }
    else:
        query = str(trip_input)
        initial_state = {
            "messages": [],
            "user_query": query,
            "destination": None,
            "origin": None,
            "start_date": None,
            "end_date": None,
            "num_days": None,
            "budget_inr": None,
            "currency": "INR",
            "num_travellers": 1,
            "plan": None,
            "missing_info": None,
            "visa_info": None,
            "destination_guide": None,
            "itinerary": None,
            "flight_options": None,
            "hotel_options": None,
            "cost_breakdown": None,
            "total_cost_inr": None,
            "budget_ok": None,
            "budget_feedback": None,
            "weather_forecast": None,
            "weather_adjustments": None,
            "events_and_advisories": None,
            "final_itinerary": None,
            "replan_count": 0,
        }

    final_state = graph.invoke(initial_state)
    return format_trip_output(final_state)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    test_query = "Plan a 4-day trip to Singapore in December under Rs 80,000 from Chennai"
    print(f"\n🚀 Running SmartTrip AI for: {test_query}\n")
    res = run_trip_planner(test_query)
    print("Execution complete! Costs:", res.get("costs"))
    print("Days in itinerary:", len(res.get("itinerary", [])))

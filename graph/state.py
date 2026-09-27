"""
Shared state definition for the SmartTrip AI LangGraph workflow.

Uses TypedDict with Annotated fields for proper LangGraph state management.
"""

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


# ============================================================
# TRIP STATE — Shared across all agents
# ============================================================

class TripState(TypedDict):
    """
    The shared state passed between all agents in the graph.
    """

    # --- User Input ---
    messages: Annotated[list, add_messages]
    user_query: str

    # --- Parsed Trip Parameters ---
    destination: Optional[str]
    origin: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    num_days: Optional[int]
    budget_inr: Optional[float]
    currency: Optional[str]
    num_travellers: Optional[int]

    # --- Planner Output ---
    plan: Optional[str]
    missing_info: Optional[list[str]]

    # --- RAG Knowledge ---
    visa_info: Optional[dict]
    destination_guide: Optional[dict]

    # --- Itinerary Agent Output ---
    itinerary: Optional[str]

    # --- Budget Agent Output ---
    flight_options: Optional[list[dict]]
    hotel_options: Optional[list[dict]]
    cost_breakdown: Optional[dict]
    total_cost_inr: Optional[float]
    budget_ok: Optional[bool]
    budget_feedback: Optional[str]

    # --- Weather Agent Output ---
    weather_forecast: Optional[dict]
    weather_adjustments: Optional[str]

    # --- Web Search Results ---
    events_and_advisories: Optional[str]

    # --- Final Output ---
    final_itinerary: Optional[str]
    replan_count: Optional[int]

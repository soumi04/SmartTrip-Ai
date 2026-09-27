"""
RAG Knowledge Base for SmartTrip AI.

Loads visa policies, destination guides, and travel insurance data
from JSON files and provides semantic search capabilities.
"""

import json
from pathlib import Path
from typing import Optional


# ============================================================
# DATA DIRECTORY
# ============================================================

DATA_DIR = Path(__file__).resolve().parent.parent / "database"


# ============================================================
# LOAD RAG DOCUMENTS
# ============================================================

def _load_json(filename: str) -> list:
    """Load a JSON file from the database directory."""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_visa_policies() -> list[dict]:
    """Load visa policy documents."""
    return _load_json("visa_policies.json")


def load_destination_guides() -> list[dict]:
    """Load destination guide documents."""
    return _load_json("destination_guides.json")


def load_travel_insurance() -> list[dict]:
    """Load travel insurance documents."""
    return _load_json("travel_insurance.json")


# ============================================================
# SEARCH FUNCTIONS
# ============================================================

def search_visa_policy(country: str) -> Optional[dict]:
    """
    Search visa policies for a specific country.
    Returns the matching visa policy or None.
    """
    policies = load_visa_policies()
    country_lower = country.lower().strip()

    for policy in policies:
        if policy["country"].lower() == country_lower:
            return policy

    # Fuzzy match: check if the country name is contained
    for policy in policies:
        if (
            country_lower in policy["country"].lower()
            or policy["country"].lower() in country_lower
        ):
            return policy

    return None


def search_destination_guide(city: str) -> Optional[dict]:
    """
    Search destination guides for a specific city.
    Returns the matching guide or None.
    """
    guides = load_destination_guides()
    city_lower = city.lower().strip()

    for guide in guides:
        if guide["city"].lower() == city_lower:
            return guide

    # Fuzzy match
    for guide in guides:
        if (
            city_lower in guide["city"].lower()
            or guide["city"].lower() in city_lower
        ):
            return guide

    return None


def search_travel_insurance(
    coverage_type: str = "Comprehensive",
) -> list[dict]:
    """
    Search travel insurance plans by coverage type.
    Returns matching insurance plans.
    """
    plans = load_travel_insurance()
    coverage_lower = coverage_type.lower().strip()

    results = []
    for plan in plans:
        if plan["coverage_type"].lower() == coverage_lower:
            results.append(plan)

    # If no exact match, return all plans
    if not results:
        return plans

    return results


def get_all_knowledge(destination: str) -> dict:
    """
    Retrieve all RAG knowledge for a destination.
    Returns a dict with visa_policy, destination_guide, and insurance_plans.
    """
    visa = search_visa_policy(destination)
    guide = search_destination_guide(destination)
    insurance = load_travel_insurance()

    return {
        "visa_policy": visa,
        "destination_guide": guide,
        "insurance_plans": insurance,
    }

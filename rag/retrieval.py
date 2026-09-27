"""Retrieve curated travel guidance with source metadata."""

import json
from pathlib import Path
from typing import Any

KNOWLEDGE_PATH = Path(__file__).with_name("travel_knowledge.json")


def retrieve_travel_knowledge(destination: str) -> dict[str, Any]:
    """Return the matching destination guide and visa caveat."""
    with KNOWLEDGE_PATH.open("r", encoding="utf-8") as knowledge_file:
        knowledge = json.load(knowledge_file)

    destination_record = next(
        (
            record
            for record in knowledge.get("destinations", [])
            if record["destination"].casefold() == destination.strip().casefold()
        ),
        None,
    )

    if destination_record is None:
        return {
            "guide": [],
            "visa_note": (
                "Visa eligibility was not determined. Check the destination's official "
                "embassy or immigration website for rules matching your passport and trip."
            ),
            "visa_sources": [],
            "guide_sources": [],
            "insurance": knowledge.get("insurance", {}),
        }

    return {
        "guide": destination_record.get("guide", []),
        "visa_note": destination_record.get("visa_note", ""),
        "visa_sources": [destination_record["visa_source"]]
        if destination_record.get("visa_source")
        else [],
        "guide_sources": [destination_record["guide_source"]]
        if destination_record.get("guide_source")
        else [],
        "insurance": knowledge.get("insurance", {}),
    }

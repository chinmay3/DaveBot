from __future__ import annotations

from typing import Literal


QueryClass = Literal["history", "menu", "food", "cooking", "regional", "general"]


def classify_query(question: str) -> QueryClass:
    q = question.lower()
    if any(
        token in q
        for token in ["combo", "meal", "slider", "sliders", "tender", "tenders", "shake", "shakes", "#1", "#2", "#3", "#4"]
    ):
        return "menu"
    if any(
        token in q
        for token in ["reaper", "spice level", "heat level", "hot level", "extra hot", "medium", "mild", "no spice", "spiciest"]
    ):
        return "regional"
    if any(token in q for token in ["found", "founded", "founder", "history", "start", "valuation", "roark"]):
        return "history"
    if any(token in q for token in ["vegan", "menu", "cauliflower", "eat", "ingredient", "allergy"]):
        return "food"
    if any(token in q for token in ["cook", "fried", "marinate", "spice level", "breading", "technique"]):
        return "cooking"
    if any(token in q for token in ["nashville", "tennessee", "regional", "hot chicken"]):
        return "regional"
    return "general"


def topic_filter_for_class(query_class: QueryClass) -> str | None:
    mapping = {
        "history": "restaurant-history",
        "menu": "restaurant-history",
        "food": None,
        "cooking": "cooking-techniques",
        "regional": "nashville-hot-chicken",
    }
    return mapping.get(query_class)

import re


_STATS_KEYWORDS = {
    "mean", "average", "distribution", "count", "max", "min",
    "trend", "compare", "summary", "describe", "plot", "histogram"
}

_FORECAST_KEYWORDS = {
    "forecast", "predict", "future", "next", "projection", "estimate"
}

_EXPLAIN_KEYWORDS = {
    "why", "explain", "reason", "interpret", "insight", "impact"
}


def route_query(query: str) -> str:
    if not query or not isinstance(query, str):
        return "llm"

    q = query.lower()
    tokens = set(re.findall(r"\b\w+\b", q))

    if tokens & _FORECAST_KEYWORDS:
        return "forecast"

    if tokens & _STATS_KEYWORDS:
        return "stats"

    if tokens & _EXPLAIN_KEYWORDS:
        return "llm"

    return "llm"

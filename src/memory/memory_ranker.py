import re
from datetime import datetime


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]+", (text or "").lower()))


def rank_items(query: str, items: list[dict], limit: int = 8) -> list[dict]:
    query_tokens = _tokens(query)
    ranked = []

    for item in items:
        content = item.get("content") or item.get("message") or ""
        item_tokens = _tokens(content)
        overlap = len(query_tokens & item_tokens)
        importance = int(item.get("importance") or 1)
        recency = 0.0

        created_at = item.get("created_at")
        if created_at:
            try:
                age_days = max((datetime.utcnow() - datetime.fromisoformat(created_at)).days, 0)
                recency = max(0.0, 1.0 - min(age_days, 30) / 30)
            except Exception:
                recency = 0.0

        score = overlap * 2.0 + importance * 1.2 + recency
        ranked.append({**item, "score": score})

    ranked.sort(key=lambda row: row.get("score", 0), reverse=True)
    return ranked[:limit]


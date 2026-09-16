from .memory_ranker import rank_items
from .memory_store import list_memories, recent_chats


def retrieve_memory_items(user_id: str, query: str, limit: int = 8) -> list[dict]:
    memories = list_memories(user_id, limit=200)
    chats = []
    for chat in recent_chats(user_id, limit=30):
        chats.append({
            "id": f"chat-{chat['id']}",
            "user_id": user_id,
            "memory_type": "short_term",
            "content": f"User: {chat['message']}\nEVA: {chat['response']}",
            "importance": 2,
            "created_at": chat["created_at"],
        })

    return rank_items(query, memories + chats, limit=limit)


def retrieve_memory(user_id: str, query: str, limit: int = 8) -> str:
    items = retrieve_memory_items(user_id, query, limit=limit)
    if not items:
        return ""

    lines = []
    for item in items:
        memory_type = item.get("memory_type", "memory")
        content = (item.get("content") or "").strip()
        if content:
            lines.append(f"- [{memory_type}] {content[:700]}")
    return "\n".join(lines)


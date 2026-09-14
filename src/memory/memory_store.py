import os
import re
import sqlite3
from pathlib import Path
from datetime import datetime


DB_PATH = Path(os.getenv("EVA_MEMORY_DB", Path(__file__).resolve().parents[2] / "memory.db"))

SENSITIVE_MEMORY_PATTERN = re.compile(
    r"(password|passcode|api[_\s-]?key|secret|private\s+key|access\s+token|bearer\s+|bank\s+account|credit\s+card|ssn|cvv)",
    re.IGNORECASE,
)


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_memory_db():
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            role TEXT DEFAULT 'user',
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            title TEXT,
            message TEXT NOT NULL,
            response TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            memory_type TEXT NOT NULL,
            content TEXT NOT NULL,
            importance INTEGER DEFAULT 1,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chats_user_created ON chats(user_id, created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_user_type ON memories(user_id, memory_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_user_created ON memories(user_id, created_at)")
    conn.commit()
    conn.close()


def upsert_user(user_id: str, name: str = "", email: str = "", role: str = "user"):
    init_memory_db()
    value = email or user_id
    conn = _connect()
    conn.execute("""
        INSERT INTO users (name, email, role, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(email) DO UPDATE SET
            name = COALESCE(NULLIF(excluded.name, ''), users.name),
            role = COALESCE(NULLIF(excluded.role, ''), users.role)
    """, (name or user_id, value, role or "user", datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


def save_chat(user_id: str, message: str, response: str, title: str = "Chat"):
    init_memory_db()
    conn = _connect()
    conn.execute("""
        INSERT INTO chats (user_id, title, message, response, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, title, message, response, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


def save_memory(user_id: str, content: str, memory_type: str = "long_term", importance: int = 1):
    init_memory_db()
    text = (content or "").strip()
    if not text or SENSITIVE_MEMORY_PATTERN.search(text):
        return None

    conn = _connect()

    existing_memory = conn.execute("""
        SELECT id FROM memories
        WHERE user_id = ? AND memory_type = ? AND content = ?
    """, (user_id, memory_type, text)).fetchone()
    if existing_memory:
        conn.close()
        return existing_memory["id"]

    cur = conn.execute("""
        INSERT INTO memories (user_id, memory_type, content, importance, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, memory_type, text, max(1, min(int(importance or 1), 5)), datetime.utcnow().isoformat()))
    conn.commit()
    memory_id = cur.lastrowid
    conn.close()
    return memory_id


def update_memory(user_id: str, memory_id: int, content: str = None, memory_type: str = None, importance: int = None) -> bool:
    init_memory_db()
    fields = []
    params = []

    if content is not None:
        text = content.strip()
        if not text or SENSITIVE_MEMORY_PATTERN.search(text):
            return False
        fields.append("content = ?")
        params.append(text)
    if memory_type:
        fields.append("memory_type = ?")
        params.append(memory_type)
    if importance is not None:
        fields.append("importance = ?")
        params.append(max(1, min(int(importance or 1), 5)))

    if not fields:
        return False

    params.extend([memory_id, user_id])
    conn = _connect()
    cur = conn.execute(
        f"UPDATE memories SET {', '.join(fields)} WHERE id = ? AND user_id = ?",
        params,
    )
    conn.commit()
    updated = cur.rowcount > 0
    conn.close()
    return updated


def list_memories(user_id: str, query: str = "", memory_type: str = "", limit: int = 100):
    init_memory_db()
    clauses = ["user_id = ?"]
    params = [user_id]
    if memory_type:
        clauses.append("memory_type = ?")
        params.append(memory_type)
    if query:
        clauses.append("content LIKE ?")
        params.append(f"%{query}%")

    sql = f"""
        SELECT id, user_id, memory_type, content, importance, created_at
        FROM memories
        WHERE {' AND '.join(clauses)}
        ORDER BY importance DESC, created_at DESC
        LIMIT ?
    """
    params.append(limit)
    conn = _connect()
    rows = [dict(row) for row in conn.execute(sql, params).fetchall()]
    conn.close()
    return rows


def recent_chats(user_id: str, limit: int = 10):
    init_memory_db()
    conn = _connect()
    rows = [dict(row) for row in conn.execute("""
        SELECT id, user_id, title, message, response, created_at
        FROM chats
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
    """, (user_id, limit)).fetchall()]
    conn.close()
    return rows


def delete_memory(user_id: str, memory_id: int) -> bool:
    init_memory_db()
    conn = _connect()
    cur = conn.execute("DELETE FROM memories WHERE id = ? AND user_id = ?", (memory_id, user_id))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


def forget_all_memories(user_id: str) -> dict:
    init_memory_db()
    conn = _connect()
    memory_cur = conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
    chat_cur = conn.execute("DELETE FROM chats WHERE user_id = ?", (user_id,))
    conn.commit()
    result = {"memories_deleted": memory_cur.rowcount, "chats_deleted": chat_cur.rowcount}
    conn.close()
    return result


def export_memories(user_id: str) -> dict:
    return {
        "user_id": user_id,
        "memories": list_memories(user_id, limit=1000),
        "recent_chats": recent_chats(user_id, limit=100),
        "exported_at": datetime.utcnow().isoformat(),
    }

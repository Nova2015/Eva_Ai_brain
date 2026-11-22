import sqlite3

def q(db, sql):
    try:
        c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        r = c.execute(sql).fetchall()
        c.close()
        return r
    except Exception as e:
        return f"ERR {type(e).__name__}: {e}"

print("feedback by rating/thumb:",
      q("memory.db", "SELECT rating, thumb, count(*) FROM feedback GROUP BY rating, thumb"))
print("chat_history rows:",
      q("memory.db", "SELECT count(*) FROM chat_history"))
print("candidates by status:",
      q("learning_loop.db", "SELECT status, count(*) FROM training_candidates GROUP BY status"))
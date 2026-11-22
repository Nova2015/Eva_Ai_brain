import sqlite3, sys

def counts(db):
    try:
        c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        tables = [r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        print(f"\n== {db} ==")
        for t in tables:
            n = c.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
            print(f"  {t}: {n}")
        c.close()
    except Exception as e:
        print(f"\n== {db} == ERR {type(e).__name__}: {e}")

for db in sys.argv[1:]:
    counts(db)
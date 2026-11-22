with open('app.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

old = '''    token = ws.headers.get("X-Agent-Token", "")
    if token != AGENT_TOKEN:
        await ws.close(code=4001); return'''

new = '''    # Accept token from header OR query param
    token = ws.headers.get("X-Agent-Token", "") or ws.query_params.get("token", "")
    if token != AGENT_TOKEN:
        await ws.close(code=4001); return'''

found = old in content
if found:
    content = content.replace(old, new)
    print("✅ Token check fixed")
else:
    print("⚠️  Not found — already fixed or different code")

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Done! Restart uvicorn.")

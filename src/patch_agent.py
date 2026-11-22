"""
Run from E:\\code\\Rag project\\src:
    python patch_agent.py
"""

with open('app.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

agent_code = '''
# ── Local Agent WebSocket ─────────────────────────────────
import uuid
from fastapi import WebSocket, WebSocketDisconnect

AGENT_TOKEN    = os.getenv("EVA_AGENT_TOKEN", "eva_agent_secret_2025")
_agent_ws      = None   # connected local agent
_pending_tasks = {}     # task_id → asyncio.Future

@app.websocket("/agent/ws")
async def agent_websocket(ws: WebSocket):
    global _agent_ws
    token = ws.headers.get("X-Agent-Token", "")
    if token != AGENT_TOKEN:
        await ws.close(code=4001); return
    await ws.accept()
    _agent_ws = ws
    logger.info("Local agent connected")
    try:
        async for raw in ws.iter_text():
            data = json.loads(raw)
            if data.get("type") == "result":
                task_id = data.get("task_id")
                fut = _pending_tasks.pop(task_id, None)
                if fut and not fut.done():
                    fut.set_result(data.get("result", {}))
    except WebSocketDisconnect:
        _agent_ws = None
        logger.info("Local agent disconnected")

async def send_to_agent(action: str, params: dict, timeout: int = 15) -> dict:
    if not _agent_ws:
        return {"status": "error", "message": "Local agent not connected. Run eva_local_agent.py on your machine."}
    task_id = str(uuid.uuid4())
    fut = asyncio.get_event_loop().create_future()
    _pending_tasks[task_id] = fut
    await _agent_ws.send_text(json.dumps({"type": "task", "task_id": task_id, "action": action, "params": params}))
    try:
        return await asyncio.wait_for(fut, timeout=timeout)
    except asyncio.TimeoutError:
        _pending_tasks.pop(task_id, None)
        return {"status": "error", "message": "Agent timed out"}

class AgentTask(BaseModel):
    action: str
    params: dict = {}

@app.post("/agent/execute")
@limiter.limit("30/minute")
async def agent_execute(request: Request, task: AgentTask, current_user: str = Depends(get_current_user)):
    """Send a task to the local Windows agent"""
    result = await send_to_agent(task.action, task.params)
    return result

@app.get("/agent/status")
async def agent_status():
    return {"connected": _agent_ws is not None, "pending_tasks": len(_pending_tasks)}

# ─────────────────────────────────────────────────────────
'''

# Add before the last line of app.py
old = '''# ── API Keys Update Endpoint'''
new = agent_code + '''# ── API Keys Update Endpoint'''

found = old in content
if found:
    content = content.replace(old, new)
    print("✅ Agent WebSocket endpoint added")
else:
    # Append at end
    content += agent_code
    print("✅ Agent endpoint appended to end of file")

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Done! Restart uvicorn and run eva_local_agent.py")

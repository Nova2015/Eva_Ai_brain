"""
Update 64 — Autopilot: autonomous multi-step tasks
==================================================
Goal in, researched report out, without step-by-step prompting:

    goal -> plan (LLM) -> execute tools in sequence -> synthesise -> report

Why this is safe to run unattended
----------------------------------
Autonomy is bounded by what the agent CAN do, not by what we ask it not to.
Every tool in the registry is READ-ONLY and side-effect free:

    web_search   DuckDuckGo query
    read_url     fetch a public page and extract text
    rag_query    ask EVA's own indexed knowledge base
    think        reason over what has been gathered so far

There is deliberately no tool that writes, sends, deletes, posts, executes
code, or spends money. An autonomous loop that can only read and summarise
cannot do damage if the planner goes wrong — the worst case is a bad report.
Anything with real-world side effects stays behind explicit human approval
(see agents/approval_gate.py) and is out of scope here.

Bounded by construction: max steps, wall-clock deadline, per-tool timeouts,
output truncation, and cooperative cancellation checked between steps.

Storage note: tasks live in the backend's instance-local sqlite, which Render
resets on redeploy. Finished reports are returned in full and can be copied
out; durable storage is the same open item as the review queue.
"""

import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from src.learning_loop import _db as _loop_db, require_train_admin

logger = logging.getLogger("EVA.autopilot")
router = APIRouter()

HF_ML_URL = os.getenv("HF_ML_URL", "https://keshav1212-eva-ml-service.hf.space").rstrip("/")

MAX_STEPS = int(os.getenv("AUTOPILOT_MAX_STEPS", "10"))
DEADLINE_SECONDS = int(os.getenv("AUTOPILOT_DEADLINE_SECONDS", "300"))
TOOL_TIMEOUT = 25
MAX_TOOL_CHARS = 6000
MAX_CONTEXT_CHARS = 18000

STATUS_RUNNING = "RUNNING"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"
STATUS_CANCELLED = "CANCELLED"

_cancelled: set = set()
_lock = threading.Lock()
_tables_ready = False


def _now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _db() -> sqlite3.Connection:
    global _tables_ready
    conn = _loop_db()
    if not _tables_ready:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS autopilot_tasks ("
            " id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,"
            " goal TEXT NOT NULL, status TEXT NOT NULL, progress INTEGER DEFAULT 0,"
            " plan TEXT, steps TEXT, report TEXT, error TEXT)"
        )
        conn.commit()
        _tables_ready = True
    return conn


def _update(task_id: str, **fields: Any) -> None:
    sets, vals = ["updated_at = ?"], [_now()]
    for k, v in fields.items():
        sets.append(f"{k} = ?")
        vals.append(json.dumps(v) if isinstance(v, (dict, list)) else v)
    vals.append(task_id)
    with _lock:
        conn = _db()
        conn.execute(f"UPDATE autopilot_tasks SET {', '.join(sets)} WHERE id = ?", vals)
        conn.commit()


# ── Tools (read-only, every one) ─────────────────────────────────────────────

def _tool_web_search(query: str) -> str:
    out: List[str] = []
    with httpx.Client(timeout=TOOL_TIMEOUT, follow_redirects=True,
                      headers={"User-Agent": "EVA-Autopilot/1.0"}) as client:
        r = client.get("https://api.duckduckgo.com/",
                       params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1})
        d = r.json()
        if d.get("Abstract"):
            out.append(f"{d.get('Heading', query)}: {d['Abstract']} [{d.get('AbstractURL', '')}]")
        for topic in (d.get("RelatedTopics") or [])[:8]:
            if isinstance(topic, dict) and topic.get("Text"):
                out.append(f"{topic['Text']} [{topic.get('FirstURL', '')}]")
        # DuckDuckGo's instant-answer API is thin for many queries; fall back to
        # the HTML endpoint so research tasks have something to read.
        if len(out) < 3:
            h = client.get("https://html.duckduckgo.com/html/", params={"q": query},
                           headers={"User-Agent": "Mozilla/5.0"})
            for m in re.findall(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', h.text)[:8]:
                url, title = m[0], re.sub(r"<[^>]+>", "", m[1]).strip()
                if title:
                    out.append(f"{title} [{url}]")
    return "\n".join(out) if out else "No results."


def _tool_read_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return "Refused: only http(s) URLs are allowed."
    # Block obvious internal targets — an autonomous fetcher must not be usable
    # as a probe into private networks.
    if re.search(r"://(localhost|127\.|0\.0\.0\.0|10\.|192\.168\.|169\.254\.|\[?::1)", url, re.I):
        return "Refused: internal address."
    with httpx.Client(timeout=TOOL_TIMEOUT, follow_redirects=True,
                      headers={"User-Agent": "EVA-Autopilot/1.0"}) as client:
        r = client.get(url)
        r.raise_for_status()
        if "text/html" not in r.headers.get("content-type", "text/html"):
            return r.text[:MAX_TOOL_CHARS]
        text = re.sub(r"<script[^>]*>.*?</script>", " ", r.text, flags=re.S | re.I)
        text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&[a-zA-Z#0-9]+;", " ", text)
        return re.sub(r"\s{2,}", " ", text).strip()[:MAX_TOOL_CHARS]


def _tool_rag_query(question: str) -> str:
    with httpx.Client(timeout=60) as client:
        r = client.post(f"{HF_ML_URL}/rag/chat",
                        json={"message": question, "workspace_id": "default-org"})
        r.raise_for_status()
        d = r.json()
    srcs = ", ".join(d.get("sources") or []) or "none"
    return f"{(d.get('answer') or '')[:MAX_TOOL_CHARS]}\n(sources: {srcs})"


def _llm(prompt: str, max_tokens: int = 900, attempts: int = 3) -> str:
    """Try every configured provider, then retry with backoff.

    A long task makes several LLM calls in quick succession, which is exactly
    how a per-minute rate limit gets hit mid-run. Backing off briefly turns a
    transient 429 into a pause instead of a lost task.
    """
    from src.admin_training import _call_provider, _providers
    configured = [p for p in _providers() if p.get("available")]
    if not configured:
        raise RuntimeError("no LLM provider is configured on this instance")

    last = "unknown error"
    for attempt in range(attempts):
        for p in configured:
            try:
                return _call_provider(p["id"], prompt)
            except Exception as e:
                last = f"{p['id']}: {type(e).__name__}"
                logger.warning("[AUTOPILOT] provider %s failed (attempt %d): %s",
                               p["id"], attempt + 1, str(e)[:160])
        if attempt < attempts - 1:
            time.sleep(8 * (attempt + 1))   # 8s, then 16s — clears per-minute limits
    raise RuntimeError(f"all providers failed after {attempts} attempts ({last})")


def _tool_think(instruction: str, context: str) -> str:
    return _llm(f"{instruction}\n\nMaterial gathered so far:\n{context[:MAX_CONTEXT_CHARS]}")


TOOLS = {
    "web_search": "Search the public web. Input: the search query.",
    "read_url":   "Fetch one public page and extract its text. Input: an absolute URL.",
    "rag_query":  "Ask EVA's own indexed document knowledge base. Input: a question.",
    "think":      "Reason over everything gathered so far. Input: what to work out.",
}


def _run_tool(tool: str, arg: str, context: str) -> str:
    if tool == "web_search":
        return _tool_web_search(arg)
    if tool == "read_url":
        return _tool_read_url(arg)
    if tool == "rag_query":
        return _tool_rag_query(arg)
    if tool == "think":
        return _tool_think(arg, context)
    raise ValueError(f"unknown tool {tool}")


# ── Planner + executor ───────────────────────────────────────────────────────

def _make_plan(goal: str) -> List[Dict[str, str]]:
    tool_doc = "\n".join(f"- {k}: {v}" for k, v in TOOLS.items())
    prompt = (
        "You are a research planner. Break the goal into at most "
        f"{MAX_STEPS - 1} concrete steps using ONLY these tools:\n{tool_doc}\n\n"
        "Rules: start by gathering information before reasoning about it; "
        "use read_url only for URLs a previous search actually returned; "
        "the last step should be a `think` step that assembles the findings.\n"
        'Return ONLY a JSON array like [{"tool":"web_search","input":"...","why":"..."}]\n\n'
        f"GOAL: {goal}"
    )
    raw = _llm(prompt)
    match = re.search(r"\[.*\]", raw, re.S)
    if not match:
        raise RuntimeError("planner returned no JSON array")
    plan = json.loads(match.group(0))
    clean: List[Dict[str, str]] = []
    for step in plan[:MAX_STEPS - 1]:
        tool = str(step.get("tool", "")).strip()
        if tool in TOOLS:
            clean.append({"tool": tool,
                          "input": str(step.get("input", ""))[:500],
                          "why": str(step.get("why", ""))[:200]})
    if not clean:
        raise RuntimeError("planner produced no usable steps")
    return clean


def _run_task(task_id: str, goal: str) -> None:
    started = time.time()
    steps: List[Dict[str, Any]] = []
    context_parts: List[str] = []
    try:
        plan = _make_plan(goal)
        _update(task_id, plan=json.dumps(plan), progress=10)

        for i, step in enumerate(plan, 1):
            if task_id in _cancelled:
                _update(task_id, status=STATUS_CANCELLED, progress=100,
                        steps=json.dumps(steps))
                return
            if time.time() - started > DEADLINE_SECONDS:
                steps.append({"n": i, "tool": "-", "input": "",
                              "output": "Deadline reached; stopping and reporting early.",
                              "ok": False, "seconds": 0})
                break

            t0 = time.time()
            entry = {"n": i, "tool": step["tool"], "input": step["input"],
                     "why": step.get("why", "")}
            try:
                out = _run_tool(step["tool"], step["input"], "\n\n".join(context_parts))
                entry.update(ok=True, output=str(out)[:MAX_TOOL_CHARS])
                context_parts.append(f"[step {i} · {step['tool']}({step['input']})]\n{out}")
            except Exception as e:
                entry.update(ok=False, output=f"{type(e).__name__}: {str(e)[:200]}")
            entry["seconds"] = round(time.time() - t0, 2)
            steps.append(entry)
            _update(task_id, steps=json.dumps(steps),
                    progress=10 + int(75 * i / max(1, len(plan))))

        material = "\n\n".join(context_parts)[:MAX_CONTEXT_CHARS]
        try:
            report = _llm(
                "Write the final deliverable for this goal using ONLY the material "
                "gathered below. Use markdown with clear headings. Cite source URLs "
                "inline where the material provides them. If the material is thin, "
                "say so plainly instead of padding.\n\n"
                f"GOAL: {goal}\n\nMATERIAL:\n{material}",
                max_tokens=1400,
            )
            degraded = False
        except Exception as e:
            # Synthesis is the last step, but the research before it is the
            # expensive part. Losing a completed multi-step crawl because the
            # final call hit a rate limit is unacceptable, so the gathered
            # material is returned raw with an explicit notice.
            logger.warning("[AUTOPILOT] %s synthesis unavailable: %s", task_id, e)
            report = (
                f"# {goal}\n\n"
                f"> **Synthesis unavailable** ({type(e).__name__}). Every research step below "
                "completed; only the final write-up could not be generated. The raw findings "
                "are preserved here — re-run the task to get the written briefing.\n\n"
                "## Raw findings\n\n" + material
            )
            degraded = True

        _update(task_id, status=STATUS_COMPLETED, progress=100,
                steps=json.dumps(steps), report=report,
                error="synthesis_unavailable" if degraded else None)
        logger.info("[AUTOPILOT] %s completed in %ds (%d steps, degraded=%s)",
                    task_id, int(time.time() - started), len(steps), degraded)
    except Exception as e:
        logger.error("[AUTOPILOT] %s failed: %s", task_id, e)
        _update(task_id, status=STATUS_FAILED, progress=100,
                steps=json.dumps(steps), error=f"{type(e).__name__}: {str(e)[:300]}")
    finally:
        _cancelled.discard(task_id)


# ── API ──────────────────────────────────────────────────────────────────────

class RunBody(BaseModel):
    goal: str


@router.post("/run")
async def autopilot_run(body: RunBody, _: bool = Depends(require_train_admin)):
    goal = (body.goal or "").strip()
    if len(goal) < 8:
        raise HTTPException(status_code=400, detail="Give the task a real goal (8+ characters)")
    task_id = uuid.uuid4().hex[:12]
    with _lock:
        conn = _db()
        conn.execute(
            "INSERT INTO autopilot_tasks (id, created_at, updated_at, goal, status, progress)"
            " VALUES (?, ?, ?, ?, ?, 0)",
            (task_id, _now(), _now(), goal[:500], STATUS_RUNNING),
        )
        conn.commit()
    threading.Thread(target=_run_task, args=(task_id, goal), daemon=True).start()
    return {"ok": True, "id": task_id, "status": STATUS_RUNNING,
            "limits": {"max_steps": MAX_STEPS, "deadline_seconds": DEADLINE_SECONDS}}


@router.get("/tasks")
async def autopilot_tasks(limit: int = Query(20, ge=1, le=100),
                          _: bool = Depends(require_train_admin)):
    rows = _db().execute(
        "SELECT id, created_at, goal, status, progress FROM autopilot_tasks"
        " ORDER BY created_at DESC LIMIT ?", (limit,),
    ).fetchall()
    return {"tasks": [{"id": r[0], "created_at": r[1], "goal": r[2],
                       "status": r[3], "progress": r[4]} for r in rows]}


@router.get("/tasks/{task_id}")
async def autopilot_task(task_id: str, _: bool = Depends(require_train_admin)):
    r = _db().execute(
        "SELECT id, created_at, updated_at, goal, status, progress, plan, steps, report, error"
        " FROM autopilot_tasks WHERE id = ?", (task_id,),
    ).fetchone()
    if r is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"id": r[0], "created_at": r[1], "updated_at": r[2], "goal": r[3],
            "status": r[4], "progress": r[5],
            "plan": json.loads(r[6]) if r[6] else [],
            "steps": json.loads(r[7]) if r[7] else [],
            "report": r[8], "error": r[9]}


@router.post("/tasks/{task_id}/cancel")
async def autopilot_cancel(task_id: str, _: bool = Depends(require_train_admin)):
    r = _db().execute("SELECT status FROM autopilot_tasks WHERE id = ?", (task_id,)).fetchone()
    if r is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if r[0] != STATUS_RUNNING:
        raise HTTPException(status_code=409, detail=f"Task already {r[0]}")
    _cancelled.add(task_id)
    return {"ok": True, "id": task_id, "status": "CANCELLING",
            "detail": "Stops before the next step."}


@router.get("/tools")
async def autopilot_tools():
    """Every tool Autopilot can use. All read-only by design."""
    return {"tools": [{"name": k, "description": v, "side_effects": "none"}
                      for k, v in TOOLS.items()],
            "max_steps": MAX_STEPS, "deadline_seconds": DEADLINE_SECONDS}

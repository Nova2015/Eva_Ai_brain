"""SQLite-backed persistent async ingestion jobs for EVA RAG."""
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
import time
from pathlib import Path
from typing import Any, Dict, Optional

QUEUED="QUEUED"; PARSING="PARSING"; CHUNKING="CHUNKING"; VALIDATING="VALIDATING"
EMBEDDING="EMBEDDING"; INDEXING="INDEXING"; COMPLETE="COMPLETE"; FAILED="FAILED"; CANCELLED="CANCELLED"
TERMINAL=frozenset({COMPLETE,FAILED,CANCELLED})
ACTIVE=frozenset({QUEUED,PARSING,CHUNKING,VALIDATING,EMBEDDING,INDEXING})
_ROOT=Path(__file__).resolve().parents[2]
_DEFAULT_DB=_ROOT/"data"/"eva_rag_jobs.db"
_JOBS: Dict[str,Dict[str,Any]]={}
_LOCK=threading.RLock()

def _db_path():
    return Path(os.getenv("EVA_RAG_JOB_DB",str(_DEFAULT_DB))).expanduser()

@contextmanager
def _connect():
    path=_db_path(); path.parent.mkdir(parents=True,exist_ok=True)
    conn=sqlite3.connect(str(path),timeout=30)
    conn.row_factory=sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS rag_ingestion_jobs(
            job_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, workspace_id TEXT NOT NULL,
            status TEXT NOT NULL, updated_at REAL NOT NULL, payload_json TEXT NOT NULL)""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rag_jobs_owner_updated ON rag_ingestion_jobs(owner_id,updated_at)")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def _persist(job):
    payload=json.dumps(job,ensure_ascii=False,separators=(",",":"))
    with _connect() as conn:
        conn.execute("""INSERT INTO rag_ingestion_jobs
        (job_id,owner_id,workspace_id,status,updated_at,payload_json) VALUES(?,?,?,?,?,?)
        ON CONFLICT(job_id) DO UPDATE SET owner_id=excluded.owner_id,
        workspace_id=excluded.workspace_id,status=excluded.status,
        updated_at=excluded.updated_at,payload_json=excluded.payload_json""",
        (job["job_id"],job.get("owner_id","anonymous"),job.get("workspace_id","default"),
         job["status"],float(job["updated_at"]),payload))

def _load(job_id):
    with _connect() as conn:
        row=conn.execute("SELECT payload_json FROM rag_ingestion_jobs WHERE job_id=?",(job_id,)).fetchone()
    if row is None: return None
    try:
        value=json.loads(row["payload_json"])
        return value if isinstance(value,dict) else None
    except (TypeError,ValueError): return None

def create_job(job_id: str, filename: str="", *, owner_id: str="anonymous", workspace_id: str="default", retry_of: Optional[str]=None):
    now=time.time()
    job={"job_id":job_id,"filename":filename,"owner_id":owner_id,"workspace_id":workspace_id,
         "status":QUEUED,"stage":QUEUED,"progress":0,"current_step":"Queued","message":"Queued",
         "created_at":now,"updated_at":now,"chunk_count":0,"error":None,"result":None,
         "data_id":None,"document_path":None,"retry_of":retry_of,"cancel":False}
    with _LOCK:
        _persist(job); _JOBS[job_id]=job
    return _public(job)

def update_job(job_id: str, *, status: Optional[str]=None, progress: Optional[int]=None,
               current_step: Optional[str]=None, message: Optional[str]=None,
               chunk_count: Optional[int]=None, error: Optional[str]=None,
               result: Optional[Dict[str,Any]]=None, data_id: Optional[str]=None,
               document_path: Optional[str]=None):
    with _LOCK:
        job=_JOBS.get(job_id) or _load(job_id)
        if not job: return None
        if status is not None: job["status"]=status; job["stage"]=status
        if progress is not None: job["progress"]=max(job["progress"],min(100,int(progress)))
        if current_step is not None: job["current_step"]=current_step; job["message"]=current_step
        if message is not None: job["message"]=message
        if chunk_count is not None: job["chunk_count"]=max(0,int(chunk_count))
        if error is not None: job["error"]=error
        if result is not None: job["result"]=result
        if data_id is not None: job["data_id"]=data_id
        if document_path is not None: job["document_path"]=document_path
        job["updated_at"]=time.time(); _persist(job); _JOBS[job_id]=job
        return _public(job)

def get_job_internal(job_id: str, *, owner_id: str):
    """Return an owner-scoped internal record, including private recovery fields."""
    with _LOCK:
        job=_JOBS.get(job_id)
        if job is None:
            job=_load(job_id)
            if job is not None: _JOBS[job_id]=job
        if not job or job.get("owner_id","anonymous") != owner_id:
            return None
        return dict(job)

def get_job(job_id: str, *, owner_id: Optional[str]=None):
    with _LOCK:
        job=_JOBS.get(job_id)
        if job is None:
            job=_load(job_id)
            if job is not None: _JOBS[job_id]=job
        if job and owner_id is not None and job.get("owner_id","anonymous")!=owner_id: job=None
        if not job:
            return {"job_id":job_id,"status":"NOT_FOUND","stage":"NOT_FOUND","progress":0,
                    "current_step":None,"message":"Job not found","chunk_count":0,
                    "error":"job not found","result":None}
        return _public(job)

def recover_interrupted_jobs():
    """Mark jobs active at previous process shutdown as failed; retain governed source files."""
    recovered=0
    with _LOCK:
        with _connect() as conn:
            rows=conn.execute("SELECT payload_json FROM rag_ingestion_jobs WHERE status NOT IN (?,?,?)",
                              (COMPLETE,FAILED,CANCELLED)).fetchall()
        for row in rows:
            try: job=json.loads(row["payload_json"])
            except (TypeError,ValueError): continue
            if not isinstance(job,dict) or job.get("status") not in ACTIVE: continue
            job["status"]=FAILED; job["stage"]=FAILED
            job["progress"]=min(int(job.get("progress",0)),99)
            job["current_step"]="Interrupted"
            job["message"]="Processing was interrupted by a server restart; the governed source file was retained."
            job["error"]="PROCESS_RESTART: retry is required"
            job["updated_at"]=time.time(); _persist(job); _JOBS[job["job_id"]]=job; recovered+=1
    return recovered

def cleanup_finished_jobs(max_age_seconds: int=30*24*60*60):
    """Expire terminal job metadata after 30 days by default; never delete document files."""
    cutoff=time.time()-max(0,int(max_age_seconds))
    with _LOCK:
        with _connect() as conn:
            rows=conn.execute("SELECT job_id FROM rag_ingestion_jobs WHERE status IN (?,?,?) AND updated_at<?",
                              (COMPLETE,FAILED,CANCELLED,cutoff)).fetchall()
            ids=[row["job_id"] for row in rows]
            if ids: conn.executemany("DELETE FROM rag_ingestion_jobs WHERE job_id=?",[(x,) for x in ids])
        for job_id in ids: _JOBS.pop(job_id,None)
    return len(ids)

def _public(job):
    return {"job_id":job["job_id"],"filename":job.get("filename",""),"status":job["status"],
            "stage":job.get("stage",job["status"]),"progress":job["progress"],
            "current_step":job.get("current_step"),"message":job.get("message") or job.get("current_step"),
            "created_at":job.get("created_at"),"updated_at":job.get("updated_at"),
            "chunk_count":job.get("chunk_count",0),"error":job.get("error"),
            "result":job.get("result"),"data_id":job.get("data_id"),
            "retry_of":job.get("retry_of")}

def _reset_for_tests():
    with _LOCK: _JOBS.clear()

# A restarted process must not leave jobs indefinitely stuck in active states.
recover_interrupted_jobs()

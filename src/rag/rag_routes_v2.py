"""
EVA RAG Routes v2
=================
Drop-in FastAPI router. Add to app.py:

    from rag.rag_routes_v2 import router as rag_router
    app.include_router(rag_router, prefix="/rag", tags=["RAG"])

Endpoints:
    POST /rag/query       — hybrid search + answer
    POST /rag/upload      — index uploaded file
    POST /rag/ingest-url  — index a URL
    GET  /rag/documents   — list indexed docs
    GET  /rag/stats       — collection stats
    DELETE /rag/document/{doc_id} — remove a doc
    POST /rag/chat        — multi-turn RAG chat
"""

import os
import uuid
import logging
import threading
import shutil
import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

try:
    from .hybrid_rag_v2 import (
        answer,
        chunk_text,
        delete_indexed_document,
        delete_orphaned_data_id,
        hybrid_search,
        index_document,
        _get_chroma,
        list_indexed_documents,
        normalize_rag_mode,
        rag_runtime_status,
        rag_stats_snapshot,
        reconcile_data_sources,
    )
except ImportError:
    from hybrid_rag_v2 import (
        answer,
        chunk_text,
        delete_indexed_document,
        delete_orphaned_data_id,
        hybrid_search,
        index_document,
        _get_chroma,
        list_indexed_documents,
        normalize_rag_mode,
        rag_runtime_status,
        rag_stats_snapshot,
        reconcile_data_sources,
    )

try:
    from ..data_governance import approve_content_revision, get_record, list_audit_events, list_reconciliation_events, record_governance_event, record_reconciliation_event, register_import, recover_location, sha256_file
except ImportError:
    from data_governance import approve_content_revision, get_record, list_audit_events, list_reconciliation_events, record_governance_event, record_reconciliation_event, register_import, recover_location, sha256_file

try:
    import upload_policy
    import job_manager as jm
except ImportError:
    from . import upload_policy
    from . import job_manager as jm

try:
    from . import graph_store
except ImportError:
    import graph_store

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer(auto_error=False)

UPLOAD_DIR = Path(os.getenv("EVA_UPLOAD_DIR", "/tmp/eva_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
_REPO_ROOT = Path(__file__).resolve().parents[2]
PERSISTENT_DOCUMENT_DIR = Path(
    os.getenv("EVA_RAG_DOCUMENTS_DIR", str(_REPO_ROOT / "data" / "rag_documents"))
).expanduser()

SUPPORTED_EXTS = {".pdf", ".txt", ".md", ".docx", ".csv", ".json", ".py", ".js", ".ts", ".html"}
MAX_INGEST_URL_BYTES = int(os.getenv("EVA_RAG_MAX_URL_BYTES", str(5 * 1024 * 1024)))
ALLOWED_INGEST_CONTENT_TYPES = {
    "text/html", "application/xhtml+xml", "text/plain"
}


def _validate_public_ingest_url(url: str) -> str:
    """Reject non-public destinations before making an outbound URL-ingest request."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("A valid HTTP(S) URL is required")
    parsed = urlsplit(url.strip())
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only HTTP and HTTPS URLs are supported")
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise ValueError("URL host is invalid")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("URL port is invalid") from exc
    expected_port = 443 if parsed.scheme.lower() == "https" else 80
    if port not in (None, expected_port):
        raise ValueError("Only standard HTTP and HTTPS ports are allowed")

    host = parsed.hostname.rstrip(".").lower()
    if host in {"localhost"} or host.endswith((".localhost", ".local", ".internal", ".test")):
        raise ValueError("Local and internal hostnames are not allowed")
    try:
        literal = ipaddress.ip_address(host)
        addresses = [literal]
    except ValueError:
        try:
            resolved = socket.getaddrinfo(host, port or expected_port, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise ValueError("URL hostname could not be resolved") from exc
        addresses = []
        for item in resolved:
            try:
                addresses.append(ipaddress.ip_address(item[4][0].split("%", 1)[0]))
            except ValueError:
                continue
    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("URL must resolve exclusively to public IP addresses")
    return parsed.geturl()


def _run_upload_job(job_id: str, filename: str, tmp_file: str, size_bytes: int, user_id: str, workspace_id: Optional[str], ext: str, existing_data_id: Optional[str] = None, existing_document_path: Optional[str] = None):
    tmp_path = Path(tmp_file)
    try:
        jm.cleanup_finished_jobs()
        size_error = upload_policy.validate_upload_size(size_bytes)
        if size_error:
            raise ValueError(size_error["detail"])

        jm.update_job(job_id, status=jm.PARSING, progress=5, current_step="Extracting text")
        text = _extract_text(tmp_path, ext)
        if not text or not text.strip():
            raise ValueError("Could not extract text from file")

        jm.update_job(job_id, status=jm.CHUNKING, progress=25, current_step="Chunking text")
        chunks = chunk_text(text)

        jm.update_job(
            job_id,
            status=jm.VALIDATING,
            progress=40,
            current_step="Validating chunk count",
            chunk_count=len(chunks),
        )
        chunk_error = upload_policy.validate_chunk_count(len(chunks))
        if chunk_error:
            raise ValueError(chunk_error["detail"])

        jm.update_job(job_id, status=jm.EMBEDDING, progress=55, current_step="Creating embeddings", chunk_count=len(chunks))

        # Initial uploads get a new governed copy and Data ID. Retries reuse
        # the existing governed record; the staging copy is disposable only.
        if existing_data_id:
            if not existing_document_path:
                raise ValueError("Retry source path is missing")
            stored_path = Path(existing_document_path).expanduser().resolve(strict=True)
            if not stored_path.is_file():
                raise FileNotFoundError("Governed source file is missing; retry cannot proceed")
            data_record = get_record(existing_data_id)
            if not data_record:
                raise ValueError("Governance record for retry source was not found")
            expected_workspace = effective_workspace_id(workspace_id, user_id)
            if data_record.get("owner_id") != user_id or data_record.get("workspace_id") != expected_workspace:
                raise PermissionError("Retry source ownership or workspace mismatch")
            if Path(data_record.get("file_path", "")).resolve() != stored_path:
                raise ValueError("Retry source path does not match its governance record")
        else:
            PERSISTENT_DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)
            stored_path = PERSISTENT_DOCUMENT_DIR / f"{uuid.uuid4().hex}{ext}"
            shutil.copy2(tmp_path, stored_path)
            try:
                data_record = register_import(
                    stored_path,
                    source="rag_upload",
                    owner_id=user_id or "anonymous",
                    workspace_id=effective_workspace_id(workspace_id, user_id),
                    actor_id=user_id or "anonymous",
                    classification=os.getenv("EVA_RAG_DEFAULT_CLASSIFICATION", "restricted"),
                )
            except Exception:
                stored_path.unlink(missing_ok=True)
                raise

        # Persist the governed source reference before indexing so a restart
        # never loses the Data ID or the retained local document location.
        jm.update_job(
            job_id,
            data_id=data_record["data_id"],
            document_path=str(stored_path),
        )
        jm.update_job(job_id, status=jm.INDEXING, progress=70, current_step="Indexing document", chunk_count=len(chunks))
        result = index_document(
            text=text,
            source=filename,
            user_id=user_id,
            workspace_id=effective_workspace_id(workspace_id, user_id),
            metadata={
                "filename": filename,
                "ext": ext,
                "size": size_bytes,
                "data_id": data_record["data_id"],
                "classification": data_record["classification"],
            },
        )
        result["data_id"] = data_record["data_id"]
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "Indexing failed"))

        jm.update_job(
            job_id,
            status=jm.COMPLETE,
            progress=100,
            current_step="Complete",
            chunk_count=result.get("chunks", len(chunks)),
            result=result,
        )
    except Exception as exc:
        jm.update_job(job_id, status=jm.FAILED, progress=100, current_step="Failed", error=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)


# ── Auth helper (reuse existing JWT or allow anonymous) ───────────────────────
def get_user_id(creds: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> str:
    if not creds:
        return "anonymous"
    try:
        from jose import jwt as jose_jwt
        # No fallback secret. The old default accepted any token signed with
        # the string baked into source; with the key absent this now validates
        # nothing and every caller is anonymous — least privilege, not forged
        # identity. In the backend the key is required at startup, so absent
        # never happens here; this only removes the dangerous default.
        secret = (os.getenv("SECRET_KEY") or "").strip()
        if not secret:
            return "anonymous"
        payload = jose_jwt.decode(creds.credentials, secret, algorithms=["HS256"])
        identity = payload.get("sub") or payload.get("email") or payload.get("user_id")
        if not isinstance(identity, str) or not identity.strip():
            return "anonymous"
        return identity.strip()
    except Exception:
        return "anonymous"


# ── Request / Response models ─────────────────────────────────────────────────
def effective_workspace_id(workspace_id: Optional[str], user_id: Optional[str]) -> str:
    requested = (workspace_id or "").strip()
    if requested:
        return requested
    user_namespace = (user_id or "").strip()
    return user_namespace if user_namespace and user_namespace != "anonymous" else "default"


def _require_authenticated_user(user_id: Optional[str]) -> str:
    """Fail closed before any user-scoped RAG data is read or modified."""
    if not user_id or user_id == "anonymous":
        raise HTTPException(status_code=401, detail="Authentication required")
    return user_id


class OrphanIndexCleanupRequest(BaseModel):
    data_id: str
    approved: bool = False
    workspace_id: Optional[str] = None


class ContentRevisionApprovalRequest(BaseModel):
    data_id: str
    expected_sha256: str
    approved: bool = False
    workspace_id: Optional[str] = None


class DataLocationRecoveryRequest(BaseModel):
    data_id: str
    new_file_path: str
    approved: bool = False
    workspace_id: Optional[str] = None


class ReconciliationRepairRequest(BaseModel):
    data_id: str
    action: str = "reindex_missing"
    approved: bool = False
    workspace_id: Optional[str] = None


class QueryRequest(BaseModel):
    query: str
    workspace_id: Optional[str] = None
    top_k: int = 5
    mode: str = "auto"             # auto | hybrid | vector | vectorless
    min_confidence: float = 0.25
    memory: str = ""
    history: Optional[List[Dict]] = []
    chat_id: Optional[str] = None
    intent: Optional[str] = None   # override auto-detection


class ChatRequest(BaseModel):
    message: str = ""
    query: Optional[str] = None
    workspace_id: Optional[str] = None
    history: Optional[List[Dict]] = []
    chat_id: Optional[str] = None
    top_k: int = 5
    mode: str = "auto"
    memory: str = ""
    intent: Optional[str] = None


class IngestURLRequest(BaseModel):
    url: str
    workspace_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class GraphEntityCandidate(BaseModel):
    name: str
    entity_type: str
    chunk_id: str
    evidence_text: str
    confidence: float = 0.5


class GraphRelationshipCandidate(BaseModel):
    source_name: str
    source_type: str
    target_name: str
    target_type: str
    predicate: str
    chunk_id: str
    evidence_text: str
    confidence: float = 0.5


class GraphProposalRequest(BaseModel):
    data_id: str
    workspace_id: Optional[str] = None
    entities: List[GraphEntityCandidate] = []
    relationships: List[GraphRelationshipCandidate] = []


class GraphEvidenceReviewRequest(BaseModel):
    evidence_id: str
    status: str
    workspace_id: Optional[str] = None


class GraphQueryRequest(BaseModel):
    query: str
    workspace_id: Optional[str] = None
    max_hops: int = 2
    limit: int = 30


# ── Endpoints ─────────────────────────────────────────────────────────────────

def _normalized_evidence_text(value: str) -> str:
    import re
    return re.sub(r"\\s+", " ", (value or "").strip()).casefold()


def _load_current_governed_text(data_id: str, user_id: str, workspace_id: str):
    record = get_record(data_id)
    if not record or record.get("owner_id") != user_id or record.get("workspace_id") != workspace_id:
        raise HTTPException(status_code=404, detail="Governed source not found")
    source_path = Path(record.get("file_path") or "")
    if not source_path.is_file():
        raise HTTPException(status_code=409, detail="Governed source file is unavailable")
    if sha256_file(source_path) != record.get("sha256"):
        raise HTTPException(status_code=409, detail="Source content differs from its approved Governance revision")
    ext = source_path.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(status_code=415, detail="Unsupported governed source type for graph extraction")
    source_text = _extract_text(source_path, ext)
    if not source_text or not source_text.strip():
        raise HTTPException(status_code=422, detail="No extractable text found in governed source")
    return record, source_text


@router.post("/graph/proposals")
async def rag_graph_proposals(req: GraphProposalRequest, user_id: str = Depends(get_user_id)):
    """Validate model-generated graph candidates against the current governed source."""
    _require_authenticated_user(user_id)
    workspace_id = effective_workspace_id(req.workspace_id, user_id)
    if len(req.entities) > 200 or len(req.relationships) > 300:
        raise HTTPException(status_code=413, detail="Graph proposal batch exceeds the allowed candidate count")
    if not req.entities and not req.relationships:
        raise HTTPException(status_code=400, detail="At least one entity or relationship candidate is required")
    record, source_text = _load_current_governed_text(req.data_id, user_id, workspace_id)
    normalized_source = _normalized_evidence_text(source_text)
    # Validate the entire batch before writing any graph rows; malformed later
    # candidates must not leave a partially accepted proposal batch.
    for candidate in req.entities:
        evidence = _normalized_evidence_text(candidate.evidence_text)
        if not evidence or evidence not in normalized_source:
            raise HTTPException(status_code=422, detail="Entity evidence must be an exact excerpt from the governed source")
    for candidate in req.relationships:
        evidence = _normalized_evidence_text(candidate.evidence_text)
        if not evidence or evidence not in normalized_source:
            raise HTTPException(status_code=422, detail="Relationship evidence must be an exact excerpt from the governed source")
    entity_ids = {}
    entity_results = []
    relationship_results = []
    try:
        for candidate in req.entities:
            item = graph_store.upsert_entity(
                candidate.name, candidate.entity_type,
                data_id=req.data_id, owner_id=user_id, workspace_id=workspace_id,
                actor_id=user_id, chunk_id=candidate.chunk_id,
                evidence_text=candidate.evidence_text, confidence=candidate.confidence,
            )
            entity_ids[(candidate.name.strip().casefold(), candidate.entity_type.strip().casefold())] = item["entity_id"]
            entity_results.append(item)

        for candidate in req.relationships:
            source_key = (candidate.source_name.strip().casefold(), candidate.source_type.strip().casefold())
            target_key = (candidate.target_name.strip().casefold(), candidate.target_type.strip().casefold())
            source_id = entity_ids.get(source_key)
            target_id = entity_ids.get(target_key)
            if not source_id:
                source_item = graph_store.upsert_entity(
                    candidate.source_name, candidate.source_type,
                    data_id=req.data_id, owner_id=user_id, workspace_id=workspace_id,
                    actor_id=user_id, chunk_id=candidate.chunk_id,
                    evidence_text=candidate.evidence_text, confidence=candidate.confidence,
                )
                source_id = source_item["entity_id"]
                entity_ids[source_key] = source_id
                entity_results.append(source_item)
            if not target_id:
                target_item = graph_store.upsert_entity(
                    candidate.target_name, candidate.target_type,
                    data_id=req.data_id, owner_id=user_id, workspace_id=workspace_id,
                    actor_id=user_id, chunk_id=candidate.chunk_id,
                    evidence_text=candidate.evidence_text, confidence=candidate.confidence,
                )
                target_id = target_item["entity_id"]
                entity_ids[target_key] = target_id
                entity_results.append(target_item)
            relationship_results.append(graph_store.upsert_relationship(
                source_id, target_id, candidate.predicate,
                data_id=req.data_id, owner_id=user_id, workspace_id=workspace_id,
                actor_id=user_id, chunk_id=candidate.chunk_id,
                evidence_text=candidate.evidence_text, confidence=candidate.confidence,
            ))
    except HTTPException:
        raise
    except (ValueError, PermissionError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    record_governance_event(
        req.data_id, action="graph_proposals_created", owner_id=user_id,
        workspace_id=workspace_id, actor_id=user_id,
        details={"entity_candidates": len(entity_results), "relationship_candidates": len(relationship_results)},
    )
    return {
        "ok": True, "data_id": req.data_id, "source_version": record.get("version", 1),
        "workspace_id": workspace_id, "status": "proposed",
        "entities": entity_results, "relationships": relationship_results,
        "approval_required": True,
    }


@router.post("/graph/review")
async def rag_graph_review(req: GraphEvidenceReviewRequest, user_id: str = Depends(get_user_id)):
    _require_authenticated_user(user_id)
    workspace_id = effective_workspace_id(req.workspace_id, user_id)
    if req.status not in {"approved", "rejected"}:
        raise HTTPException(status_code=400, detail="status must be approved or rejected")
    try:
        result = graph_store.review_evidence(
            req.evidence_id, req.status, owner_id=user_id,
            workspace_id=workspace_id, actor_id=user_id,
        )
        return {"ok": True, **result, "workspace_id": workspace_id}
    except KeyError:
        raise HTTPException(status_code=404, detail="Graph evidence not found")
    except (ValueError, PermissionError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/graph/query")
async def rag_graph_query(req: GraphQueryRequest, user_id: str = Depends(get_user_id)):
    _require_authenticated_user(user_id)
    workspace_id = effective_workspace_id(req.workspace_id, user_id)
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")
    try:
        seeds = graph_store.search_entities(
            req.query, owner_id=user_id, workspace_id=workspace_id, limit=min(max(req.limit, 1), 100),
        )
        edges = graph_store.traverse_verified_relationships(
            [item["entity_id"] for item in seeds],
            owner_id=user_id, workspace_id=workspace_id,
            max_hops=min(max(req.max_hops, 1), 5), limit=min(max(req.limit, 1), 100),
        ) if seeds else []
        return {
            "ok": True, "workspace_id": workspace_id,
            "entities": seeds, "relationships": edges,
            "retrieval_mode": "approved_graph_only",
        }
    except (ValueError, PermissionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/limits")
async def rag_limits():
    """Backend-driven upload size and chunk limits for the UI."""
    return upload_policy.limits_payload()


@router.get("/status")
async def rag_status():
    """RAG route health without forcing model loading."""
    try:
        return rag_runtime_status()
    except Exception as e:
        return {
            "online": True,
            "vector_rag": "unavailable",
            "vectorless_rag": "unavailable",
            "default_mode": "auto",
            "error": str(e),
        }


@router.post("/query")
async def rag_query(req: QueryRequest, user_id: str = Depends(get_user_id)):
    """Hybrid RAG query — retrieves + routes to best LLM + returns answer."""
    _require_authenticated_user(user_id)
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")
    workspace_id = effective_workspace_id(req.workspace_id, user_id)
    try:
        result = answer(
            query=req.query.strip(),
            user_id=user_id,
            workspace_id=workspace_id,
            top_k=req.top_k,
            mode=normalize_rag_mode(req.mode),
            min_confidence=req.min_confidence,
            memory=req.memory,
            history=req.history or [],
            chat_id=req.chat_id,
            intent=req.intent,
        )
        return result
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
async def rag_chat(req: ChatRequest, user_id: str = Depends(get_user_id)):
    """Multi-turn RAG chat endpoint."""
    _require_authenticated_user(user_id)
    prompt = (req.message or req.query or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="message cannot be empty")
    workspace_id = effective_workspace_id(req.workspace_id, user_id)
    try:
        result = answer(
            query=prompt,
            user_id=user_id,
            workspace_id=workspace_id,
            top_k=req.top_k,
            mode=normalize_rag_mode(req.mode),
            memory=req.memory,
            history=req.history or [],
            chat_id=req.chat_id,
            intent=req.intent,
        )
        return {
            "answer": result["answer"],
            "response": result["answer"],
            "provider": result.get("provider"),
            "model": result.get("model"),
            "intent": result.get("intent"),
            "confidence": result.get("confidence"),
            "sources": result.get("sources", []),
            "fallback": result.get("fallback", False),
            "fallback_used": result.get("fallback_used", result.get("fallback", False)),
            "latency_ms": result.get("latency_ms"),
            "mode": result.get("mode", normalize_rag_mode(req.mode)),
            "mode_used": result.get("mode_used", result.get("mode", normalize_rag_mode(req.mode))),
            "retrievers_used": result.get("retrievers_used", []),
            "warning": result.get("warning"),
            "warnings": result.get("warnings", []),
            "prompt_enhanced": result.get("prompt_enhanced", False),
            "vector_rag": result.get("vector_rag"),
            "vectorless_rag": result.get("vectorless_rag"),
            "workspace_id": workspace_id,
        }
    except Exception as e:
        logger.error(f"RAG chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def rag_upload(
    file: UploadFile = File(...),
    workspace_id: Optional[str] = Form(None),
    user_id: str = Depends(get_user_id),
):
    """Upload and index a document into the vector store."""
    _require_authenticated_user(user_id)
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Allowed: {', '.join(SUPPORTED_EXTS)}")

    tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    size_bytes = 0
    try:
        with tmp_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size_bytes += len(chunk)
                out.write(chunk)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to stage upload: {exc}")

    namespace = effective_workspace_id(workspace_id, user_id)
    job_id = uuid.uuid4().hex[:12]
    jm.create_job(job_id, filename=file.filename, owner_id=user_id, workspace_id=namespace)
    thread = threading.Thread(
        target=_run_upload_job,
        args=(job_id, file.filename, str(tmp_path), size_bytes, user_id, namespace, ext),
        daemon=True,
    )
    thread.start()
    return {
        "ok": True,
        "job_id": job_id,
        "status": "QUEUED",
        "filename": file.filename,
        "chunks_created": 0,
        "workspace_id": namespace,
        "async_jobs": True,
        "message": "Upload accepted. Processing in the background.",
    }


@router.post("/ingest-url")
async def rag_ingest_url(req: IngestURLRequest, user_id: str = Depends(get_user_id)):
    """Safely fetch a public web page, retain it locally and index it as governed data."""
    _require_authenticated_user(user_id)
    try:
        url = await asyncio.to_thread(_validate_public_ingest_url, req.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    workspace_id = effective_workspace_id(req.workspace_id, user_id)
    try:
        import httpx
        body = bytearray()
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0), follow_redirects=False) as client:
            async with client.stream(
                "GET", url, headers={"User-Agent": "EVA-RAG/2.0"}
            ) as resp:
                if 300 <= resp.status_code < 400:
                    raise HTTPException(status_code=400, detail="Redirects are not followed by secure URL ingestion")
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                if content_type not in ALLOWED_INGEST_CONTENT_TYPES:
                    raise HTTPException(status_code=415, detail="Only HTML and plain-text web pages can be ingested")
                declared_size = resp.headers.get("content-length")
                if declared_size:
                    try:
                        if int(declared_size) > MAX_INGEST_URL_BYTES:
                            raise HTTPException(status_code=413, detail="URL content exceeds the ingestion size limit")
                    except ValueError:
                        raise HTTPException(status_code=502, detail="Invalid Content-Length from URL source")
                async for chunk in resp.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_INGEST_URL_BYTES:
                        raise HTTPException(status_code=413, detail="URL content exceeds the ingestion size limit")

        raw = bytes(body).decode("utf-8", errors="replace")
        extracted_text = _strip_html(raw) if content_type != "text/plain" else raw
        if not extracted_text or len(extracted_text.strip()) < 100:
            raise HTTPException(status_code=422, detail="URL returned insufficient text content")

        PERSISTENT_DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)
        web_source_dir = PERSISTENT_DOCUMENT_DIR / "web_sources"
        web_source_dir.mkdir(parents=True, exist_ok=True)
        stored_path = web_source_dir / f"{uuid.uuid4().hex}.txt"
        stored_path.write_text(extracted_text, encoding="utf-8")
        try:
            data_record = register_import(
                stored_path,
                source="rag_url_ingest",
                owner_id=user_id,
                workspace_id=workspace_id,
                actor_id=user_id,
                classification=os.getenv("EVA_RAG_DEFAULT_CLASSIFICATION", "restricted"),
            )
        except Exception:
            stored_path.unlink(missing_ok=True)
            raise

        result = index_document(
            text=extracted_text,
            source=stored_path.name,
            user_id=user_id,
            workspace_id=workspace_id,
            metadata={
                "filename": stored_path.name,
                "ext": ".txt",
                "size": stored_path.stat().st_size,
                "data_id": data_record["data_id"],
                "classification": data_record["classification"],
                "source_url": url,
                "type": "web",
            },
        )
        if not result.get("ok"):
            # Keep the governed source: reconciliation can safely repair indexing later.
            raise HTTPException(
                status_code=500,
                detail={"message": result.get("error", "Indexing failed"), "data_id": data_record["data_id"]},
            )

        record_governance_event(
            data_record["data_id"], action="rag_url_ingested", actor_id=user_id,
            details={"source_url": url, "chunks": result.get("chunks", 0)},
        )
        return {
            "ok": True,
            "url": url,
            "data_id": data_record["data_id"],
            "classification": data_record["classification"],
            "workspace_id": workspace_id,
            "chunks_created": result["chunks"],
            "vector_indexed": result.get("vector_indexed", False),
            "vectorless_indexed": result.get("vectorless_indexed", False),
            "warnings": result.get("warnings", []),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("URL ingest failed: %s", exc)
        raise HTTPException(status_code=500, detail="URL ingestion failed")


@router.get("/jobs/{job_id}")
async def rag_job_status(job_id: str, user_id: str = Depends(get_user_id)):
    if user_id == "anonymous":
        raise HTTPException(status_code=401, detail="Authentication required to view upload job status")
    job = jm.get_job(job_id, owner_id=user_id)
    if job.get("status") == "NOT_FOUND":
        raise HTTPException(status_code=404, detail=job["error"])
    return job


@router.post("/jobs/{job_id}/retry")
async def rag_retry_upload_job(job_id: str, user_id: str = Depends(get_user_id)):
    """Retry a failed upload using its retained, owner-authorized governed source."""
    if user_id == "anonymous":
        raise HTTPException(status_code=401, detail="Authentication required to retry upload jobs")
    original = jm.get_job_internal(job_id, owner_id=user_id)
    if not original:
        raise HTTPException(status_code=404, detail="Job not found")
    if original.get("status") != jm.FAILED:
        raise HTTPException(status_code=409, detail="Only failed upload jobs can be retried")
    data_id = original.get("data_id")
    document_path = original.get("document_path")
    if not data_id or not document_path:
        raise HTTPException(status_code=409, detail="This job has no retained governed source; upload the file again")
    data_record = get_record(data_id)
    if not data_record or data_record.get("owner_id") != user_id:
        raise HTTPException(status_code=404, detail="Governed retry source not found")
    workspace_id = original.get("workspace_id") or "default"
    if data_record.get("workspace_id") != workspace_id:
        raise HTTPException(status_code=404, detail="Governed retry source not found")
    source_path = Path(document_path).expanduser()
    if not source_path.is_file() or source_path.resolve() != Path(data_record["file_path"]).resolve():
        raise HTTPException(status_code=409, detail="Governed source file is unavailable or has moved; reconcile its Data ID first")
    ext = source_path.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(status_code=409, detail=f"Unsupported retained source type: {ext}")
    size_bytes = source_path.stat().st_size
    size_error = upload_policy.validate_upload_size(size_bytes)
    if size_error:
        raise HTTPException(status_code=413, detail=size_error["detail"])

    retry_tmp = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    try:
        shutil.copy2(source_path, retry_tmp)
    except Exception as exc:
        retry_tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Could not stage retry source: {exc}")

    retry_job_id = uuid.uuid4().hex[:12]
    filename = original.get("filename") or source_path.name
    jm.create_job(
        retry_job_id,
        filename=filename,
        owner_id=user_id,
        workspace_id=workspace_id,
        retry_of=job_id,
    )
    thread = threading.Thread(
        target=_run_upload_job,
        args=(retry_job_id, filename, str(retry_tmp), size_bytes, user_id, workspace_id, ext, data_id, str(source_path)),
        daemon=True,
    )
    thread.start()
    return {
        "ok": True,
        "job_id": retry_job_id,
        "retry_of": job_id,
        "data_id": data_id,
        "status": jm.QUEUED,
        "workspace_id": workspace_id,
        "async_jobs": True,
        "message": "Retry accepted. The existing governed source and Data ID will be reused.",
    }


@router.post("/reconciliation/approve-revision")
async def rag_reconciliation_approve_revision(
    req: ContentRevisionApprovalRequest,
    user_id: str = Depends(get_user_id),
):
    """Approve changed governed bytes as a new version after explicit confirmation."""
    if user_id == "anonymous":
        raise HTTPException(status_code=401, detail="Authentication required for revision approval")
    if req.approved is not True:
        raise HTTPException(status_code=400, detail="Explicit approval is required before revision recording")

    workspace_id = effective_workspace_id(req.workspace_id, user_id)
    try:
        result = approve_content_revision(
            req.data_id,
            owner_id=user_id,
            workspace_id=workspace_id,
            actor_id=user_id,
            expected_sha256=req.expected_sha256,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Governed data item not found")
    except FileNotFoundError:
        raise HTTPException(status_code=409, detail="Governed source file is missing; recover its location first")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return {
        "ok": True,
        "action": "approve_revision",
        "approved": True,
        **result,
        "workspace_id": workspace_id,
        "audit_recorded": bool(result.get("changed")),
    }


@router.post("/reconciliation/recover-location")
async def rag_reconciliation_recover_location(
    req: DataLocationRecoveryRequest,
    user_id: str = Depends(get_user_id),
):
    """Restore a missing file reference after explicit approval and hash validation."""
    if user_id == "anonymous":
        raise HTTPException(status_code=401, detail="Authentication required for location recovery")
    if req.approved is not True:
        raise HTTPException(status_code=400, detail="Explicit approval is required before location recovery")

    workspace_id = effective_workspace_id(req.workspace_id, user_id)
    try:
        record = recover_location(
            req.data_id,
            req.new_file_path,
            owner_id=user_id,
            workspace_id=workspace_id,
            actor_id=user_id,
            allowed_root=PERSISTENT_DOCUMENT_DIR,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Governed data item not found")
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return {
        "ok": True,
        "action": "recover_location",
        "approved": True,
        "data_id": record["data_id"],
        "file_name": record["file_name"],
        "workspace_id": record["workspace_id"],
        "sha256_verified": True,
        "governance_record_changed": True,
        "content_revision_changed": False,
        "source_file_deleted": False,
    }


@router.post("/reconciliation/repair")
async def rag_reconciliation_repair(
    req: ReconciliationRepairRequest,
    user_id: str = Depends(get_user_id),
):
    """Apply one explicitly approved, safe repair: re-index an unchanged governed file."""
    if user_id == "anonymous":
        raise HTTPException(status_code=401, detail="Authentication required for data repair")
    if req.action != "reindex_missing":
        raise HTTPException(status_code=400, detail="Unsupported repair action")
    if req.approved is not True:
        raise HTTPException(status_code=400, detail="Explicit approval is required before repair")

    workspace_id = effective_workspace_id(req.workspace_id, user_id)
    record = get_record(req.data_id)
    if not record or record.get("owner_id") != user_id or record.get("workspace_id") != workspace_id:
        raise HTTPException(status_code=404, detail="Governed data item not found")

    report = reconcile_data_sources(user_id, workspace_id=workspace_id)
    missing_ids = {item["data_id"] for item in report.get("missing_from_index", [])}
    if req.data_id not in missing_ids:
        raise HTTPException(status_code=409, detail="This Data ID is not currently reported as missing from the RAG index")

    source_path = Path(record["file_path"])
    if not source_path.is_file():
        raise HTTPException(status_code=409, detail="Governed source file is missing; location recovery is required")
    try:
        actual_hash = sha256_file(source_path)
    except OSError:
        raise HTTPException(status_code=409, detail="Governed source file cannot be read")
    if actual_hash != record.get("sha256"):
        raise HTTPException(status_code=409, detail="Source content has changed; revision review is required before re-indexing")

    ext = source_path.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(status_code=415, detail=f"Unsupported governed source type: {ext}")
    size_bytes = source_path.stat().st_size
    size_error = upload_policy.validate_upload_size(size_bytes)
    if size_error:
        raise HTTPException(status_code=413, detail=size_error["detail"])

    extracted_text = _extract_text(source_path, ext)
    if not extracted_text or not extracted_text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from governed source")
    chunks = chunk_text(extracted_text)
    chunk_error = upload_policy.validate_chunk_count(len(chunks))
    if chunk_error:
        raise HTTPException(status_code=413, detail=chunk_error["detail"])

    result = index_document(
        text=extracted_text,
        source=record["file_name"],
        user_id=user_id,
        workspace_id=workspace_id,
        metadata={
            "filename": record["file_name"],
            "ext": ext,
            "size": size_bytes,
            "data_id": record["data_id"],
            "classification": record["classification"],
        },
    )
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("error", "Re-indexing failed"))
    record_governance_event(
        record["data_id"],
        "rag_index_repaired",
        owner_id=user_id,
        workspace_id=workspace_id,
        actor_id=user_id,
        details={
            "action": "reindex_missing",
            "chunks": result.get("chunks", len(chunks)),
            "vector_indexed": result.get("vector_indexed", False),
            "vectorless_indexed": result.get("vectorless_indexed", False),
        },
    )
    return {
        "ok": True,
        "action": "reindex_missing",
        "approved": True,
        "data_id": record["data_id"],
        "workspace_id": workspace_id,
        "chunks": result.get("chunks", len(chunks)),
        "vector_indexed": result.get("vector_indexed", False),
        "vectorless_indexed": result.get("vectorless_indexed", False),
        "warnings": result.get("warnings", []),
        "source_file_deleted": False,
        "governance_record_changed": False,
    }


@router.post("/reconciliation/cleanup-orphan")
async def rag_reconciliation_cleanup_orphan(
    req: OrphanIndexCleanupRequest,
    user_id: str = Depends(get_user_id),
):
    """Delete chunks for one confirmed orphan Data ID after explicit approval."""
    if user_id == "anonymous":
        raise HTTPException(status_code=401, detail="Authentication required for orphan cleanup")
    if req.approved is not True:
        raise HTTPException(status_code=400, detail="Explicit approval is required before orphan cleanup")

    workspace_id = effective_workspace_id(req.workspace_id, user_id)
    try:
        report = reconcile_data_sources(user_id, workspace_id=workspace_id)
    except Exception:
        logger.exception("Orphan cleanup preflight reconciliation failed")
        raise HTTPException(status_code=500, detail="Could not verify orphan status; no cleanup was attempted")
    orphan_ids = {item["data_id"] for item in report.get("index_without_governance", [])}
    if req.data_id not in orphan_ids:
        raise HTTPException(status_code=409, detail="Data ID is not currently confirmed as an orphan in this workspace")

    try:
        result = delete_orphaned_data_id(
            req.data_id,
            user_id=user_id,
            workspace_id=workspace_id,
        )
    except PermissionError:
        raise HTTPException(status_code=401, detail="Authenticated owner is required")
    except Exception:
        logger.exception("Orphan index cleanup failed")
        raise HTTPException(status_code=500, detail="Orphan cleanup failed; inspect the reconciliation report before retrying")

    record_reconciliation_event(
        req.data_id,
        owner_id=user_id,
        workspace_id=workspace_id,
        action="orphan_rag_index_cleaned",
        actor_id=user_id,
        details={
            "deleted_vector_chunks": result["deleted_vector_chunks"],
            "deleted_vectorless_chunks": result["deleted_vectorless_chunks"],
            "deleted_chunks": result["deleted_chunks"],
            "source_files_deleted": False,
        },
    )
    return {
        "ok": True,
        "action": "cleanup_orphan",
        "approved": True,
        **result,
        "workspace_id": workspace_id,
        "source_files_deleted": False,
        "governance_records_changed": False,
        "audit_recorded": True,
    }


@router.get("/reconciliation/audit")
async def rag_reconciliation_audit(
    data_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    user_id: str = Depends(get_user_id),
):
    """Read owner/workspace-scoped reconciliation events, including orphan cleanup."""
    if user_id == "anonymous":
        raise HTTPException(status_code=401, detail="Authentication required to view reconciliation audit")
    namespace = effective_workspace_id(workspace_id, user_id)
    return {
        "ok": True,
        "workspace_id": namespace,
        "events": list_reconciliation_events(
            owner_id=user_id,
            workspace_id=namespace,
            data_id=data_id,
        ),
    }


@router.get("/documents/{data_id}/audit")
async def rag_governed_document_audit(
    data_id: str,
    workspace_id: Optional[str] = None,
    user_id: str = Depends(get_user_id),
):
    """Return the audit history only to the owning user in the selected workspace."""
    if user_id == "anonymous":
        raise HTTPException(status_code=401, detail="Authentication required to view data audit history")
    namespace = effective_workspace_id(workspace_id, user_id)
    record = get_record(data_id)
    if not record or record.get("owner_id") != user_id or record.get("workspace_id") != namespace:
        raise HTTPException(status_code=404, detail="Governed data item not found")
    return {
        "ok": True,
        "data_id": data_id,
        "workspace_id": namespace,
        "events": list_audit_events(data_id),
    }


@router.get("/reconciliation")
async def rag_data_reconciliation(
    workspace_id: Optional[str] = None,
    user_id: str = Depends(get_user_id),
):
    """Read-only Governance/RAG/local-file consistency report."""
    if user_id == "anonymous":
        raise HTTPException(status_code=401, detail="Authentication required for data reconciliation")
    try:
        return reconcile_data_sources(
            user_id,
            workspace_id=effective_workspace_id(workspace_id, user_id),
        )
    except Exception as exc:
        logger.exception("Data reconciliation failed")
        raise HTTPException(status_code=500, detail="Data reconciliation failed")


@router.get("/documents")
async def rag_documents(workspace_id: Optional[str] = None, user_id: str = Depends(get_user_id)):
    """List all indexed documents for this user."""
    _require_authenticated_user(user_id)
    try:
        return list_indexed_documents(user_id, workspace_id=effective_workspace_id(workspace_id, user_id))
    except Exception as e:
        logger.error(f"Documents list failed: {e}")
        return {"documents": [], "error": str(e)}


@router.get("/stats")
async def rag_stats(workspace_id: Optional[str] = None, user_id: str = Depends(get_user_id)):
    """Collection statistics."""
    _require_authenticated_user(user_id)
    try:
        stats = rag_stats_snapshot(user_id, workspace_id=effective_workspace_id(workspace_id, user_id))
        return {
            **stats,
            "total_documents": stats.get("total_documents", stats.get("document_count", 0)),
            "document_count": stats.get("document_count", stats.get("total_documents", 0)),
            "total_chunks": stats.get("total_chunks", stats.get("chunk_count", 0)),
            "chunk_count": stats.get("chunk_count", stats.get("total_chunks", 0)),
        }
    except Exception as e:
        return {"total_documents": 0, "total_chunks": 0, "error": str(e)}


@router.delete("/document/{doc_id}")
async def rag_delete_document(doc_id: str, workspace_id: Optional[str] = None, user_id: str = Depends(get_user_id)):
    """Delete a specific document chunk by ID."""
    _require_authenticated_user(user_id)
    try:
        return delete_indexed_document(doc_id, user_id, workspace_id=effective_workspace_id(workspace_id, user_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Text extraction helpers ───────────────────────────────────────────────────

def _extract_text(path: Path, ext: str) -> str:
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return ""
    if ext == ".docx":
        try:
            from docx import Document
            doc = Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.error(f"Text read failed: {e}")
        return ""


def _strip_html(html: str) -> str:
    import re
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"\s{3,}", "\n\n", text)
    return text.strip()


def _guess_type(source: str) -> str:
    src = source.lower()
    if src.endswith(".pdf"):      return "pdf"
    if src.endswith(".docx"):     return "docx"
    if src.endswith(".md"):       return "md"
    if src.startswith("http"):    return "url"
    if any(src.endswith(e) for e in [".py", ".js", ".ts"]): return "code"
    return "txt"

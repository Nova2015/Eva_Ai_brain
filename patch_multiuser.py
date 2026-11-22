"""
Run this script from E:\\code\\Rag project\\
It patches app.py with multi-user knowledge base routes.
"""

with open('app.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# ── CHANGE 1: Modify /chat to use user-specific RAG ──────────
old1 = '''        if use_rag:
            result = rag.query(req.question, user_memory=user_memory)'''

new1 = '''        if use_rag:
            try:
                from multi_user_rag import get_user_rag
                user_rag = get_user_rag(req.user_id)
                result = user_rag.query(req.question, user_memory=user_memory)
                # Fall back to global RAG if user has no personal docs
                if not result.get("answer") or result.get("confidence", 0) < 0.1:
                    result = rag.query(req.question, user_memory=user_memory)
            except Exception:
                result = rag.query(req.question, user_memory=user_memory)'''

# ── CHANGE 2: Modify /upload-pdf to index per user ───────────
old2 = '''        t = threading.Thread(target=_index_in_background, args=(file.filename,), daemon=True)
        t.start()
        return {"message": f"{file.filename} uploaded! Indexing in background — ready in ~10s."}'''

new2 = '''        t = threading.Thread(target=_index_in_background, args=(file.filename,), daemon=True)
        t.start()
        # Also index into user's personal knowledge base
        try:
            from multi_user_rag import get_user_rag
            user_rag = get_user_rag(current_user)
            threading.Thread(target=user_rag.add_document, args=(str(dest),), daemon=True).start()
            logger.info(f"Indexing {file.filename} into {current_user}'s personal KB")
        except Exception as _ue:
            logger.warning(f"User KB indexing skipped: {_ue}")
        return {"message": f"{file.filename} uploaded! Indexing in background — ready in ~10s."}'''

# ── CHANGE 3: Add multi-user routes before /documents route ──
old3 = '''@app.get("/documents")
async def list_documents(current_user: str = Depends(get_current_user)):'''

new3 = '''# ── Multi-User Knowledge Base Routes ─────────────────────────

@app.get("/user/kb/stats")
@limiter.limit("30/minute")
async def user_kb_stats(request: Request, current_user: str = Depends(get_current_user)):
    """Get current user's knowledge base stats"""
    try:
        from multi_user_rag import get_user_rag
        mgr = get_user_rag(current_user)
        return mgr.get_stats()
    except Exception as e:
        return {"user_id": current_user, "error": str(e), "doc_count": 0}

@app.get("/user/kb/documents")
@limiter.limit("30/minute")
async def user_kb_documents(request: Request, current_user: str = Depends(get_current_user)):
    """List all documents in user's personal knowledge base"""
    try:
        from multi_user_rag import get_user_rag
        mgr = get_user_rag(current_user)
        stats = mgr.get_stats()
        return {
            "user_id":   current_user,
            "documents": stats.get("docs", []),
            "count":     stats.get("doc_count", 0),
        }
    except Exception as e:
        return {"user_id": current_user, "documents": [], "count": 0, "error": str(e)}

@app.delete("/user/kb/document/{filename}")
@limiter.limit("10/minute")
async def delete_user_document(request: Request, filename: str, current_user: str = Depends(get_current_user)):
    """Delete a document from user's personal knowledge base"""
    try:
        from multi_user_rag import get_user_rag
        mgr = get_user_rag(current_user)
        return mgr.delete_document(filename)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.delete("/user/kb/clear")
@limiter.limit("5/minute")
async def clear_user_kb(request: Request, current_user: str = Depends(get_current_user)):
    """Clear ALL documents from user's personal knowledge base"""
    try:
        from multi_user_rag import get_user_rag
        mgr = get_user_rag(current_user)
        return mgr.clear_all_documents()
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/admin/users/stats")
@limiter.limit("10/minute")
async def admin_all_users(request: Request, current_user: str = Depends(get_current_user)):
    """Admin only: view all users knowledge base stats"""
    if current_user != os.getenv("ADMIN_USERNAME", "admin"):
        raise HTTPException(status_code=403, detail="Admin only")
    try:
        from multi_user_rag import get_all_user_stats
        return {"users": get_all_user_stats()}
    except Exception as e:
        raise HTTPException(500, str(e))

# ─────────────────────────────────────────────────────────────

@app.get("/documents")
async def list_documents(current_user: str = Depends(get_current_user)):'''

c1 = old1 in content
c2 = old2 in content
c3 = old3 in content

if c1: content = content.replace(old1, new1)
if c2: content = content.replace(old2, new2)
if c3: content = content.replace(old3, new3)

print(f"Change 1 (user RAG in /chat):      {'✅ Applied' if c1 else '❌ NOT FOUND'}")
print(f"Change 2 (user KB in /upload-pdf): {'✅ Applied' if c2 else '❌ NOT FOUND'}")
print(f"Change 3 (multi-user routes):      {'✅ Applied' if c3 else '❌ NOT FOUND'}")

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("\nDone! Now create multi_user_rag.py and restart uvicorn.")

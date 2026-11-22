with open('app.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

old1 = '''        if use_rag:
            result = rag.query(req.question, user_memory=user_memory)
        else:'''

new1 = '''        if use_rag:
            result = rag.query(req.question, user_memory=user_memory)
            RAG_THRESHOLD = float(os.getenv("RAG_CONFIDENCE_THRESHOLD", "0.35"))
            if result.get("confidence", 1.0) < RAG_THRESHOLD:
                fb = await _llm_fallback(req.question, [])
                if fb:
                    result["answer"] = fb
                    result["fallback_mode"] = True
        else:'''

c1 = old1 in content
if c1: content = content.replace(old1, new1)
print(f"Change 1: {'✅ Applied' if c1 else '❌ NOT FOUND'}")

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Done!")
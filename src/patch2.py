with open('app.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Change 2 - LLM cascade
old2 = '''            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                max_tokens=1024,
                temperature=0.7,
            )
            answer = response.choices[0].message.content
            result = {"answer": answer, "sources": [], "confidence": 1.0, "healed": False, "fallback_mode": False}'''

new2 = '''            answer = None
            try:
                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=messages, max_tokens=1024, temperature=0.7,
                )
                answer = response.choices[0].message.content
            except Exception as _groq_err:
                print(f"Groq failed: {_groq_err}")

            if not answer:
                _openai_key = os.getenv("OPENAI_API_KEY", "")
                if _openai_key:
                    try:
                        import openai as _openai_mod
                        import asyncio
                        _oa = _openai_mod.OpenAI(api_key=_openai_key)
                        _oa_r = _oa.chat.completions.create(
                            model="gpt-4o-mini", messages=messages, max_tokens=1024,
                        )
                        answer = _oa_r.choices[0].message.content
                        print("✅ OpenAI fallback used")
                    except Exception as _oa_err:
                        print(f"OpenAI failed: {_oa_err}")

            if not answer:
                _gemini_key = os.getenv("GEMINI_API_KEY", "")
                if _gemini_key:
                    try:
                        import google.generativeai as _genai
                        _genai.configure(api_key=_gemini_key)
                        _gm = _genai.GenerativeModel("gemini-3.5-flash")
                        _r = _gm.generate_content(messages[-1]["content"])
                        answer = _r.text
                        print("✅ Gemini fallback used")
                    except Exception as _gm_err:
                        print(f"Gemini failed: {_gm_err}")

            if not answer:
                answer = "All AI providers unavailable. Check API keys in Settings."
            result = {"answer": answer, "sources": [], "confidence": 1.0, "healed": False, "fallback_mode": False}'''

# Change 3 - _llm_fallback helper
old3 = '''# ______________________________________________
# TTS  (protected)
# ______________________________________________'''

new3 = '''async def _llm_fallback(question: str, history: list) -> str:
    msgs = [
        {"role": "system", "content": "You are EVA, a helpful AI assistant. Answer clearly."},
        {"role": "user", "content": question},
    ]
    _openai_key = os.getenv("OPENAI_API_KEY", "")
    if _openai_key:
        try:
            import openai as _openai_mod
            _oa = _openai_mod.OpenAI(api_key=_openai_key)
            _r = _oa.chat.completions.create(model="gpt-4o-mini", messages=msgs, max_tokens=1024)
            return _r.choices[0].message.content
        except Exception as e:
            print(f"Fallback OpenAI: {e}")
    _gemini_key = os.getenv("GEMINI_API_KEY", "")
    if _gemini_key:
        try:
            import google.generativeai as _genai
            _genai.configure(api_key=_gemini_key)
            _gm = _genai.GenerativeModel("gemini-3.5-flash")
            return _gm.generate_content(question).text
        except Exception as e:
            print(f"Fallback Gemini: {e}")
    return ""

# ______________________________________________
# TTS  (protected)
# ______________________________________________'''

c2 = old2 in content
c3 = old3 in content
if c2: content = content.replace(old2, new2)
if c3: content = content.replace(old3, new3)

print(f"Change 2 (LLM cascade):    {'✅ Applied' if c2 else '❌ NOT FOUND'}")
print(f"Change 3 (fallback helper): {'✅ Applied' if c3 else '❌ NOT FOUND'}")

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Done!")
from openhands.sdk import LLM
import inspect

print("LLM signature:")
print(inspect.signature(LLM))

print("\nOllama-related source:")
src = inspect.getsource(LLM)
for i, line in enumerate(src.splitlines(), 1):
    if "ollama" in line.lower() or "provider" in line.lower():
        print(f"{i}: {line}")

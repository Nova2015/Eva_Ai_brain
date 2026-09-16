from openhands.sdk.llm.llm import LLMProvider
import inspect

print("=== SIGNATURE ===")
print(inspect.signature(LLMProvider.from_model))

print("\n=== SOURCE ===")
src = inspect.getsource(LLMProvider.from_model)
print(src)

print("\n=== PROVIDER CLASS ===")
print(LLMProvider)

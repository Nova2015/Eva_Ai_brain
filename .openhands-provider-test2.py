from openhands.sdk.llm.llm import LLMProvider

tests = [
    ("qwen2.5-coder:3b", None),
    ("qwen2.5-coder:3b", "http://127.0.0.1:11434"),
    ("ollama/qwen2.5-coder:3b", None),
    ("ollama/qwen2.5-coder:3b", "http://127.0.0.1:11434"),
]

for model, api_base in tests:
    print("\n==============================")
    print("MODEL:", model)
    print("API_BASE:", api_base)

    p = LLMProvider.from_model(
        model=model,
        api_base=api_base,
    )

    print("PARSED MODEL:", p.model)
    print("PROVIDER:", p.name)
    print("API BASE:", p.api_base)

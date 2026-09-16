from openhands.sdk import LLM
from openhands.sdk.llm.llm import LLMProvider

models = [
    "qwen2.5-coder:3b",
    "ollama/qwen2.5-coder:3b",
]

for model in models:
    print("\n==============================")
    print("MODEL:", model)

    try:
        provider = LLMProvider.from_model(
            model=model,
            ollama_base_url="http://127.0.0.1:11434",
        )
        print("PROVIDER:", provider)
        print("NAME:", provider.name)
        print("API BASE:", provider.api_base)
        print("API KEY:", provider.api_key_for_litellm("ollama"))
        print("LITELLM KWARGS:", provider.as_litellm_call_kwargs(api_key="ollama"))
    except Exception as e:
        print("ERROR:", type(e).__name__, str(e))

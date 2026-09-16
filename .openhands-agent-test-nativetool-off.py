from pathlib import Path

from openhands.sdk import LLM, Conversation
from openhands.tools.preset.default import get_default_agent

workspace = Path(r"D:\Eva_Ai_brain\.openhands-test-workspace")
workspace.mkdir(parents=True, exist_ok=True)

llm = LLM(
    model="ollama/qwen2.5-coder:3b",
    ollama_base_url="http://127.0.0.1:11434",
    api_key="ollama",
    native_tool_calling=False,
    reasoning_effort="none",
)

agent = get_default_agent(
    llm=llm,
    cli_mode=True,
)

print("AGENT:", agent.name)
print("TOOLS:", [tool.name for tool in agent.tools])

conversation = Conversation(
    agent=agent,
    workspace=workspace,
    max_iteration_per_run=30,
)

conversation.send_message(
    """
Work only inside the current workspace.

Build a tiny Python calculator project.

Requirements:
1. Create calculator.py with add(a, b) and subtract(a, b).
2. Create test_calculator.py with tests for both functions.
3. Run the tests using the terminal.
4. If tests fail, inspect the failure and fix the code.
5. Run the tests again.
6. Do not modify anything outside the current workspace.
7. Do not stop merely because the files were created.
8. Finish only after verifying that the tests pass.
"""
)

conversation.run()

print("\n=== OPENHANDS AUTONOMOUS TEST FINISHED ===")

from pathlib import Path

from openhands.sdk import LLM, Conversation
from openhands.tools.preset.default import get_default_agent

workspace = Path(r"D:\Eva_Ai_brain\.openhands-test-workspace")
workspace.mkdir(parents=True, exist_ok=True)

llm = LLM(
    model="ollama/qwen2.5:3b",
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
    r"""
You are running on Windows with PowerShell.

The workspace is:
D:\Eva_Ai_brain\.openhands-test-workspace

Use only this workspace. Do not use /workspace or Unix-style paths.

First use the terminal tool to run:
Get-Location

Then use the terminal tool to create calculator.py and test_calculator.py
in the workspace above. Do not use FileEditorAction for this task.

calculator.py must define add(a, b) and subtract(a, b).
test_calculator.py must test both functions.

Run the tests using:
python -m unittest -v

If tests fail, inspect the error, fix the files, and run the tests again.
Do not claim success unless the terminal output confirms that the tests pass.
"""
)
conversation.run()

print("\n=== OPENHANDS AUTONOMOUS TEST FINISHED ===")

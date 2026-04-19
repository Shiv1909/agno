"""
E2B Cloud Sandbox
=================
Run code in an isolated E2B cloud Linux container.

E2BSandbox gives the agent a secure, ephemeral Linux environment.
All file operations and code execution happen inside the cloud container —
nothing touches the host machine.

BaseSandbox (which E2BSandbox inherits) implements all file ops by running
Python3 scripts inside the container via execute(), so the agent gets the
full BackendProtocol (ls, read, write, edit, grep, glob) plus execute().

Prerequisites:
    pip install e2b-code-interpreter

Environment variables required:
    E2B_API_KEY
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_DEPLOYMENT

Run:
    .venvs/demo/bin/python cookbook/94_backends/04_e2b_sandbox.py
"""

import os

from agno.agent import Agent
from agno.backends.e2b import E2BSandbox
from agno.models.azure import AzureOpenAI
from agno.tools.backend import BackendToolkit

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = AzureOpenAI(
    id=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
)

# ---------------------------------------------------------------------------
# Sandbox + Toolkit
# ---------------------------------------------------------------------------
sandbox = E2BSandbox(api_key=os.environ["E2B_API_KEY"])
toolkit = BackendToolkit(sandbox)

print(f"Sandbox ID: {sandbox.id}")

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
agent = Agent(
    model=model,
    tools=[toolkit],
    instructions=[
        "You have access to an isolated Linux cloud sandbox.",
        "Always use 'python3' to run Python scripts.",
        "Verify command output before reporting results.",
    ],
    markdown=True,
    debug_mode=True,
)

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n--- Verify sandbox environment ---\n")
    agent.print_response("Run uname -a and python3 --version.", stream=True)

    print("\n--- Write and run a script ---\n")
    agent.print_response(
        "Write a Python script at /sandbox/primes.py that prints all prime "
        "numbers up to 50, then run it with python3.",
        stream=True,
    )

    print("\n--- File operations in sandbox ---\n")
    agent.print_response(
        "Create a directory /sandbox/data, write a CSV file /sandbox/data/nums.csv "
        "with numbers 1 to 5 one per line, then read it back.",
        stream=True,
    )

    print("\n--- Install a package ---\n")
    agent.print_response(
        "Install the 'requests' package with pip3, then run "
        "python3 -c \"import requests; print(requests.__version__)\" to confirm.",
        stream=True,
    )

    # Clean up
    sandbox.close()
    print("\nSandbox closed.")

# dynamic_subagent — Test Log

### 01_basic.py

**Status:** NOT RUN

**Description:** Orchestrator spawns a writing specialist for two independent tasks. The LLM decides to use spawn_agent without any developer-side routing code.

**Result:** Awaiting test run with live API key.

---

### 02_with_tools.py

**Status:** NOT RUN

**Description:** Orchestrator delegates DuckDuckGo search tools to subagents via a whitelist. Subagent tool outputs stay isolated; the orchestrator only receives final summaries.

**Result:** Awaiting test run with live API key.

---

### 03_parallel.py

**Status:** NOT RUN

**Description:** Three async subagents spawned concurrently in one LLM turn. Uses aprint_response with max_concurrent=3.

**Result:** Awaiting test run with live API key.

---

### 04_team.py

**Status:** NOT RUN

**Description:** Team leader spawns subagents alongside registered members. Spawned subagents carry team_id in metadata for session-level observability.

**Result:** Awaiting test run with live API key.

---

### 05_model_tiers.py

**Status:** NOT RUN

**Description:** Cost-aware orchestrator selects model tier per subtask. LLM picks fast/standard/powerful labels; developer controls the model-ID mapping.

**Result:** Awaiting test run with live API key.

---

### 06_context_isolation.py

**Status:** NOT RUN

**Description:** Customer support scenario with mock DB and knowledge base tools. Large payloads are delegated via spawn_agent; orchestrator context only sees short summaries.

**Result:** Awaiting test run with live API key.

---

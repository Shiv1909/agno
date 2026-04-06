"""Unit tests for SubAgentConfig and SubAgentToolkit."""

from __future__ import annotations

from unittest.mock import MagicMock

from agno.agent.subagent import SubAgentConfig, SubAgentToolkit

# ---------------------------------------------------------------------------
# SubAgentConfig tests
# ---------------------------------------------------------------------------


def test_subagent_config_defaults():
    """Verify all defaults match the spec."""
    cfg = SubAgentConfig()

    # Model
    assert cfg.model is None
    assert cfg.fallback_models is None

    # Tools
    assert cfg.tools is None
    assert cfg.allowed_tools is None
    assert cfg.inherit_parent_tools is False
    assert cfg.tool_call_limit is None
    assert cfg.tool_hooks is None

    # Knowledge / RAG
    assert cfg.knowledge is None
    assert cfg.inherit_parent_knowledge is False
    assert cfg.knowledge_filters is None
    assert cfg.search_knowledge is True
    assert cfg.add_knowledge_to_context is False
    assert cfg.references_format == "json"

    # Context from parent
    assert cfg.inject_session_state is False
    assert cfg.additional_context is None

    # Output
    assert cfg.markdown is False
    assert cfg.parse_response is True
    assert cfg.structured_outputs is None
    assert cfg.use_json_mode is False

    # Reasoning
    assert cfg.reasoning is False
    assert cfg.reasoning_model is None
    assert cfg.reasoning_min_steps == 1
    assert cfg.reasoning_max_steps == 10

    # Retry
    assert cfg.retries == 0
    assert cfg.delay_between_retries == 1
    assert cfg.exponential_backoff is False

    # Hooks
    assert cfg.pre_hooks is None
    assert cfg.post_hooks is None

    # Limits
    assert cfg.max_concurrent == 5

    # LLM overrides
    assert cfg.allow_model_override is False
    assert cfg.allow_tool_selection is True

    # Debug
    assert cfg.debug_mode is False


def test_subagent_config_accepts_model():
    """Verify model, markdown, and tool_call_limit can be overridden."""
    mock_model = MagicMock()
    cfg = SubAgentConfig(
        model=mock_model,
        markdown=True,
        tool_call_limit=5,
    )
    assert cfg.model is mock_model
    assert cfg.markdown is True
    assert cfg.tool_call_limit == 5


def test_subagent_config_accepts_other_fields():
    """Verify a broader set of non-default values round-trip correctly."""
    cfg = SubAgentConfig(
        tools=["tool_a"],
        allowed_tools=["tool_b"],
        inherit_parent_tools=True,
        inherit_parent_knowledge=True,
        inject_session_state=True,
        additional_context="ctx",
        references_format="yaml",
        reasoning=True,
        reasoning_min_steps=2,
        reasoning_max_steps=8,
        retries=3,
        delay_between_retries=2,
        exponential_backoff=True,
        max_concurrent=10,
        allow_model_override=True,
        allow_tool_selection=False,
        debug_mode=True,
    )
    assert cfg.tools == ["tool_a"]
    assert cfg.allowed_tools == ["tool_b"]
    assert cfg.inherit_parent_tools is True
    assert cfg.inherit_parent_knowledge is True
    assert cfg.inject_session_state is True
    assert cfg.additional_context == "ctx"
    assert cfg.references_format == "yaml"
    assert cfg.reasoning is True
    assert cfg.reasoning_min_steps == 2
    assert cfg.reasoning_max_steps == 8
    assert cfg.retries == 3
    assert cfg.delay_between_retries == 2
    assert cfg.exponential_backoff is True
    assert cfg.max_concurrent == 10
    assert cfg.allow_model_override is True
    assert cfg.allow_tool_selection is False
    assert cfg.debug_mode is True


# ---------------------------------------------------------------------------
# SubAgentToolkit tests
# ---------------------------------------------------------------------------


def _make_toolkit() -> SubAgentToolkit:
    """Return a SubAgentToolkit with a minimal mock parent."""
    parent = MagicMock()
    parent.model = None
    parent.tools = []
    parent.knowledge = None
    parent.session_state = None
    parent.id = "parent-id"
    config = SubAgentConfig()
    return SubAgentToolkit(parent=parent, config=config)


def test_subagent_toolkit_registers_tool():
    """spawn_agent must appear in both functions and async_functions."""
    toolkit = _make_toolkit()

    assert "spawn_agent" in toolkit.functions, "spawn_agent missing from toolkit.functions"
    assert "spawn_agent" in toolkit.async_functions, "spawn_agent missing from toolkit.async_functions"


def test_subagent_toolkit_tool_description():
    """The sync function object must have a description and the expected parameter properties."""
    toolkit = _make_toolkit()

    fn = toolkit.functions["spawn_agent"]

    # process_entrypoint populates description and parameters (done lazily by agent at run time;
    # we call it explicitly here so we can inspect the schema without a full agent run).
    fn.process_entrypoint()

    # description must be set (not None / empty)
    assert fn.description, "spawn_agent Function.description should not be empty"

    # parameters schema must include role, instructions, task
    props = fn.parameters.get("properties", {})
    for expected_param in ("role", "instructions", "task"):
        assert expected_param in props, f"Expected parameter '{expected_param}' not found in spawn_agent schema"


def test_subagent_toolkit_async_tool_description():
    """The async function object must also have the correct parameter schema."""
    toolkit = _make_toolkit()

    async_fn = toolkit.async_functions["spawn_agent"]

    # process_entrypoint populates description and parameters for the async variant too.
    async_fn.process_entrypoint()

    assert async_fn.description, "async spawn_agent Function.description should not be empty"

    props = async_fn.parameters.get("properties", {})
    for expected_param in ("role", "instructions", "task"):
        assert expected_param in props, f"Expected parameter '{expected_param}' missing from async spawn_agent schema"


# ---------------------------------------------------------------------------
# Agent integration tests
# ---------------------------------------------------------------------------


def test_agent_has_dynamic_subagent_fields():
    """Agent accepts enable_dynamic_subagents and subagent_config."""
    from agno.agent.agent import Agent

    agent = Agent(name="test", enable_dynamic_subagents=False)
    assert agent.enable_dynamic_subagents is False
    assert agent.subagent_config is None


def test_agent_wires_toolkit_when_enabled():
    """SubAgentToolkit is added to agent.tools after initialize_agent runs."""
    from agno.agent.agent import Agent

    agent = Agent(name="test", enable_dynamic_subagents=True)
    # initialize_agent is called lazily (on first run); invoke explicitly for testing
    agent.initialize_agent()

    toolkit_found = any(isinstance(t, SubAgentToolkit) for t in (agent.tools or []))
    assert toolkit_found, "SubAgentToolkit should be in agent.tools when enable_dynamic_subagents=True"


def test_agent_wires_toolkit_with_custom_config():
    """Custom SubAgentConfig is threaded through to the toolkit."""
    from agno.agent.agent import Agent

    config = SubAgentConfig(markdown=True, max_concurrent=2)
    agent = Agent(name="test", enable_dynamic_subagents=True, subagent_config=config)
    agent.initialize_agent()

    toolkit = next((t for t in (agent.tools or []) if isinstance(t, SubAgentToolkit)), None)
    assert toolkit is not None
    assert toolkit._config.markdown is True
    assert toolkit._config.max_concurrent == 2

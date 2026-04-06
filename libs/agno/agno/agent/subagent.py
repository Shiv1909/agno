from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict

from agno.tools import Toolkit
from agno.utils.log import log_warning

if TYPE_CHECKING:
    from agno.agent.agent import Agent
    from agno.team.team import Team


class SubAgentConfig(BaseModel):
    """Pre-configuration template for dynamically spawned subagents."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Model — defaults to parent's model when None
    model: Optional[Any] = None
    fallback_models: Optional[List[Any]] = None

    # Tools
    tools: Optional[List[Any]] = None  # always given to subagents
    allowed_tools: Optional[List[str]] = None  # names LLM can pick from parent
    inherit_parent_tools: bool = False  # give subagents ALL parent tools
    tool_call_limit: Optional[int] = None
    tool_hooks: Optional[List[Any]] = None

    # Knowledge / RAG
    knowledge: Optional[Any] = None
    inherit_parent_knowledge: bool = False
    knowledge_filters: Optional[Dict[str, Any]] = None
    search_knowledge: bool = True
    add_knowledge_to_context: bool = False
    references_format: Literal["json", "yaml"] = "json"

    # Context from parent
    inject_session_state: bool = False  # pass parent session_state read-only
    additional_context: Optional[str] = None

    # Output
    markdown: bool = False
    parse_response: bool = True
    structured_outputs: Optional[bool] = None
    use_json_mode: bool = False

    # Reasoning
    reasoning: bool = False
    reasoning_model: Optional[Any] = None
    reasoning_min_steps: int = 1
    reasoning_max_steps: int = 10

    # Retry
    retries: int = 0
    delay_between_retries: int = 1
    exponential_backoff: bool = False

    # Hooks
    pre_hooks: Optional[List[Any]] = None
    post_hooks: Optional[List[Any]] = None

    # Limits
    max_concurrent: int = 5

    # LLM overrides at spawn time
    allow_model_override: bool = False
    allow_tool_selection: bool = True

    # Debug
    debug_mode: bool = False


class SubAgentToolkit(Toolkit):
    """Toolkit that exposes spawn_agent (sync) and aspawn_agent (async) to a parent agent or team."""

    def __init__(self, parent: Union[Agent, Team], config: SubAgentConfig) -> None:
        self._parent = parent
        self._config = config
        super().__init__(
            name="subagent",
            tools=[self.spawn_agent],
            async_tools=[(self.aspawn_agent, "spawn_agent")],
        )

    def spawn_agent(
        self,
        role: str,
        instructions: str,
        task: str,
        tools: Optional[List[str]] = None,
        expected_output: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """Spawn an ephemeral subagent and run it synchronously.

        Args:
            role: The name/role of the subagent to spawn.
            instructions: System instructions for the subagent.
            task: The task to run.
            tools: Optional list of tool names to pass from the parent.
            expected_output: Optional description of expected output format.
            model: Optional model override (only used if allow_model_override is True).

        Returns:
            The string content returned by the subagent, or a default message.
        """
        subagent = self._build_subagent(role, instructions, tools, expected_output, model)
        session_state = self._parent_session_state() if self._config.inject_session_state else None
        result = subagent.run(input=task, session_state=session_state, stream=False)
        if result and result.content:
            return str(result.content)
        return "Subagent completed with no output."

    async def aspawn_agent(
        self,
        role: str,
        instructions: str,
        task: str,
        tools: Optional[List[str]] = None,
        expected_output: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """Async version of spawn_agent.

        Args:
            role: The name/role of the subagent to spawn.
            instructions: System instructions for the subagent.
            task: The task to run.
            tools: Optional list of tool names to pass from the parent.
            expected_output: Optional description of expected output format.
            model: Optional model override (only used if allow_model_override is True).

        Returns:
            The string content returned by the subagent, or a default message.
        """
        subagent = self._build_subagent(role, instructions, tools, expected_output, model)
        session_state = self._parent_session_state() if self._config.inject_session_state else None
        result = await subagent.arun(input=task, session_state=session_state, stream=False)
        if result and result.content:
            return str(result.content)
        return "Subagent completed with no output."

    def _build_subagent(
        self,
        role: str,
        instructions: str,
        tool_names: Optional[List[str]],
        expected_output: Optional[str],
        model_override: Optional[str],
    ) -> Agent:
        from agno.agent.agent import Agent

        model = self._resolve_model(model_override)
        resolved_tools = self._resolve_tools(tool_names)
        knowledge = self._resolve_knowledge()
        additional_context = self._build_additional_context()
        return Agent(
            name=role,
            model=model,
            fallback_models=self._config.fallback_models,
            description=f"Ephemeral subagent: {role}",
            instructions=instructions,
            expected_output=expected_output,
            additional_context=additional_context,
            tools=resolved_tools or [],
            tool_call_limit=self._config.tool_call_limit,
            tool_hooks=self._config.tool_hooks,
            knowledge=knowledge,
            knowledge_filters=self._config.knowledge_filters,
            search_knowledge=self._config.search_knowledge and knowledge is not None,
            add_knowledge_to_context=self._config.add_knowledge_to_context,
            references_format=self._config.references_format,
            markdown=self._config.markdown,
            parse_response=self._config.parse_response,
            structured_outputs=self._config.structured_outputs,
            use_json_mode=self._config.use_json_mode,
            reasoning=self._config.reasoning,
            reasoning_model=self._config.reasoning_model,
            reasoning_min_steps=self._config.reasoning_min_steps,
            reasoning_max_steps=self._config.reasoning_max_steps,
            retries=self._config.retries,
            delay_between_retries=self._config.delay_between_retries,
            exponential_backoff=self._config.exponential_backoff,
            pre_hooks=self._config.pre_hooks,
            post_hooks=self._config.post_hooks,
            team_id=getattr(self._parent, "id", None),
            db=None,
            stream=False,
            telemetry=False,
            debug_mode=self._config.debug_mode,
        )

    def _resolve_model(self, model_override: Optional[str]) -> Any:
        if model_override and self._config.allow_model_override:
            try:
                from agno.models.utils import get_model

                return get_model(model_override)
            except Exception:
                log_warning(f"Could not resolve model override '{model_override}'. Using parent model.")
        return self._config.model or getattr(self._parent, "model", None)

    def _resolve_tools(self, tool_names: Optional[List[str]]) -> Optional[List[Any]]:
        if self._config.inherit_parent_tools:
            return getattr(self._parent, "tools", None)
        result: List[Any] = list(self._config.tools or [])
        if not tool_names or not self._config.allow_tool_selection:
            return result or None
        parent_tools = getattr(self._parent, "tools", None) or []
        allowed = set(self._config.allowed_tools) if self._config.allowed_tools else None
        requested = set(tool_names)
        permitted = (allowed & requested) if allowed else requested
        from agno.tools import Toolkit as _Toolkit
        from agno.tools.function import Function as _Function

        for tool in parent_tools:
            if isinstance(tool, _Toolkit):
                if any(fn in permitted for fn in tool.functions):
                    result.append(tool)
            elif isinstance(tool, _Function):
                if tool.name in permitted:
                    result.append(tool)
            elif callable(tool):
                name = getattr(tool, "__name__", None)
                if name and name in permitted:
                    result.append(tool)
        return result or None

    def _resolve_knowledge(self) -> Any:
        if self._config.inherit_parent_knowledge:
            return getattr(self._parent, "knowledge", None)
        return self._config.knowledge

    def _build_additional_context(self) -> Optional[str]:
        parts: List[str] = []
        if self._config.inject_session_state:
            state = self._parent_session_state()
            if state:
                parts.append("Parent session state (read-only):\n" + json.dumps(state, indent=2, default=str))
        if self._config.additional_context:
            parts.append(self._config.additional_context)
        return "\n\n".join(parts) if parts else None

    def _parent_session_state(self) -> Optional[Dict[str, Any]]:
        return getattr(self._parent, "session_state", None)

"""LangGraph StateGraph Workflow Composition and Checkpoint Management."""

import logging
from typing import Any, Dict, Literal, Optional
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    audit_and_log,
    escalate_to_human,
    generate_response,
    intent_classifier,
    retrieve_knowledge,
)
from agent.state import SupportAgentState
from agent.tools.registry import get_all_tools, register_all
from agent.tools.registry import execute_tools as execute_tools_node
from app.monitoring import set_log_context, trace_agent_node

logger = logging.getLogger("support_agent.graph")


def _instrument(node_name: str, fn):
    """Wrap a LangGraph node function with observability.

    Adds per-node OpenTelemetry spans + Prometheus duration metrics, and
    propagates session/user/ticket context into the structured log context
    so every log line inside the node carries request metadata.
    """
    if not callable(fn):
        return fn

    async def instrumented(state: SupportAgentState) -> Dict[str, Any]:
        session_id = state.get("session_id")
        set_log_context(
            session_id=str(session_id) if session_id else None,
            user_id=str(state.get("customer_id")) if state.get("customer_id") else None,
            ticket_id=str(state.get("current_ticket_id")) if state.get("current_ticket_id") else None,
        )
        async with trace_agent_node(node_name, session_id=str(session_id) if session_id else None):
            return await fn(state)

    instrumented.__name__ = getattr(fn, "__name__", node_name)
    return instrumented


def route_after_triage(
    state: SupportAgentState,
) -> Literal["escalate_to_human", "retrieve_context", "execute_tools"]:
    """Conditional edge routing logic following triage and intent classification."""
    # 1. Immediate escalation conditions
    if state.get("escalation_needed", False) or state.get("sentiment") == "frustrated":
        return "escalate_to_human"

    intent = state.get("intent", "faq")

    # 2. Informational questions route to RAG
    if intent in ["faq", "policy", "troubleshooting"]:
        return "retrieve_context"

    # 3. Actionable operations route to tool execution
    # (kept in sync with TOOL_INTENTS in agent/state.py)
    if intent in [
        "order_status",
        "billing",
        "account_update",
        "ticket_lookup",
        "ticket_create",
        "ticket_update",
        "email_draft",
    ]:
        return "execute_tools"

    # Default fallback
    return "retrieve_context"


def route_after_tools(
    state: SupportAgentState,
) -> Literal["escalate_to_human", "generate_response"]:
    """Determines whether tool outputs necessitate human intervention or response generation."""
    if state.get("escalation_needed", False):
        return "escalate_to_human"
    return "generate_response"


def get_agent_tools() -> list:
    """Return the registered LangChain tool list for binding to an LLM."""
    register_all()
    return list(get_all_tools().values())


def create_support_graph(checkpointer: Optional[Any] = None):
    """Factory function compiling the LangGraph Customer Support workflow.
    
    Args:
        checkpointer: State persistence checkpointer (e.g. MemorySaver, AsyncPostgresSaver).
        
    Returns:
        Compiled LangGraph runnable workflow.
        
    TODO:
        - Integrate production AsyncPostgresSaver or RedisSaver when configured.
        - Add interrupt_before hooks for human-in-the-loop review on critical actions.
    """
    workflow = StateGraph(SupportAgentState)

    # Register workflow nodes (wrapped with tracing + metrics instrumentation)
    workflow.add_node("triage", _instrument("triage", intent_classifier))
    workflow.add_node("retrieve_context", _instrument("retrieve_context", retrieve_knowledge))
    workflow.add_node("execute_tools", _instrument("execute_tools", execute_tools_node))
    workflow.add_node("generate_response", _instrument("generate_response", generate_response))
    workflow.add_node("escalate_to_human", _instrument("escalate_to_human", escalate_to_human))
    workflow.add_node("audit", _instrument("audit", audit_and_log))

    # Establish workflow transitions
    workflow.add_edge(START, "triage")

    workflow.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "escalate_to_human": "escalate_to_human",
            "retrieve_context": "retrieve_context",
            "execute_tools": "execute_tools",
        },
    )

    workflow.add_edge("retrieve_context", "generate_response")

    workflow.add_conditional_edges(
        "execute_tools",
        route_after_tools,
        {
            "escalate_to_human": "escalate_to_human",
            "generate_response": "generate_response",
        },
    )

    workflow.add_edge("generate_response", "audit")
    workflow.add_edge("escalate_to_human", "audit")
    workflow.add_edge("audit", END)

    # Default to in-memory checkpointer if none supplied
    active_checkpointer = checkpointer or MemorySaver()

    compiled_graph = workflow.compile(
        checkpointer=active_checkpointer,
        # TODO: Add interrupt_before=["escalate_to_human"] for human-in-the-loop approvals
    )

    logger.info("LangGraph Customer Support StateGraph compiled successfully.")
    return compiled_graph


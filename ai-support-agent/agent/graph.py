"""LangGraph StateGraph Workflow Composition and Checkpoint Management."""

import logging
from typing import Literal, Optional
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    escalate_to_human_node,
    execute_tools_node,
    generate_response_node,
    post_process_and_audit_node,
    retrieve_context_node,
    triage_and_classify_node,
)
from agent.state import SupportAgentState

logger = logging.getLogger("support_agent.graph")


def route_after_triage(
    state: SupportAgentState,
) -> Literal["escalate_to_human", "retrieve_context", "execute_tools"]:
    """Conditional edge routing logic following triage and intent classification."""
    # 1. Immediate escalation conditions
    if state.get("escalation_required", False) or state.get("sentiment") == "frustrated":
        return "escalate_to_human"

    intent = state.get("intent", "faq")

    # 2. Informational questions route to RAG
    if intent in ["faq", "policy", "troubleshooting"]:
        return "retrieve_context"

    # 3. Actionable operations route to tool execution
    if intent in ["order_status", "billing", "account_update", "ticket_lookup"]:
        return "execute_tools"

    # Default fallback
    return "retrieve_context"


def route_after_tools(
    state: SupportAgentState,
) -> Literal["escalate_to_human", "generate_response"]:
    """Determines whether tool outputs necessitate human intervention or response generation."""
    if state.get("escalation_required", False):
        return "escalate_to_human"
    return "generate_response"


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

    # Register workflow nodes
    workflow.add_node("triage", triage_and_classify_node)
    workflow.add_node("retrieve_context", retrieve_context_node)
    workflow.add_node("execute_tools", execute_tools_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.add_node("escalate_to_human", escalate_to_human_node)
    workflow.add_node("audit", post_process_and_audit_node)

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


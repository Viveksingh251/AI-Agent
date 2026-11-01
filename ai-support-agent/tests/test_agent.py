"""Unit and Workflow Tests for LangGraph Customer Support Agent."""

import pytest
from langchain_core.messages import HumanMessage

from agent.graph import create_support_graph, route_after_triage
from agent.nodes import triage_and_classify_node
from agent.state import SupportAgentState


def test_graph_compilation(agent_graph):
    """Verifies that the LangGraph StateGraph builds and compiles without errors."""
    assert agent_graph is not None
    # LangGraph compiled runnable should expose get_graph()
    assert hasattr(agent_graph, "get_graph")


def test_routing_logic_faq():
    """Verifies that informational intent routes to retrieve_context."""
    state: SupportAgentState = {
        "messages": [HumanMessage(content="What is your return policy?")],
        "customer_id": "test_cust",
        "session_id": "sess_01",
        "ticket_id": None,
        "intent": "faq",
        "sentiment": "neutral",
        "urgency": "low",
        "retrieved_docs": [],
        "tool_outputs": [],
        "escalation_required": False,
        "escalation_reason": None,
        "draft_response": None,
        "final_response": None,
        "confidence_score": 1.0,
        "turn_count": 1,
    }
    assert route_after_triage(state) == "retrieve_context"


def test_routing_logic_escalation():
    """Verifies that frustrated sentiment triggers escalation_to_human."""
    state: SupportAgentState = {
        "messages": [HumanMessage(content="I want to speak with a manager right now!")],
        "customer_id": "test_cust",
        "session_id": "sess_02",
        "ticket_id": None,
        "intent": "complaint",
        "sentiment": "frustrated",
        "urgency": "high",
        "retrieved_docs": [],
        "tool_outputs": [],
        "escalation_required": True,
        "escalation_reason": "Customer expressed extreme frustration",
        "draft_response": None,
        "final_response": None,
        "confidence_score": 0.9,
        "turn_count": 1,
    }
    assert route_after_triage(state) == "escalate_to_human"


@pytest.mark.asyncio
async def test_triage_node_execution():
    """Verifies that triage_and_classify_node produces valid state updates."""
    state: SupportAgentState = {
        "messages": [HumanMessage(content="Can you track order 1234?")],
        "customer_id": "cust_1",
        "session_id": "sess_03",
        "ticket_id": None,
        "intent": None,
        "sentiment": None,
        "urgency": None,
        "retrieved_docs": [],
        "tool_outputs": [],
        "escalation_required": False,
        "escalation_reason": None,
        "draft_response": None,
        "final_response": None,
        "confidence_score": 1.0,
        "turn_count": 0,
    }
    result = await triage_and_classify_node(state)
    assert "intent" in result
    assert "sentiment" in result
    assert result["turn_count"] == 1


"""Integration tests for the LangGraph support agent with a mocked LLM.

Runs the compiled graph end-to-end: triage → routing → tools/RAG → response
— with ``agent.nodes._get_llm`` replaced by a deterministic FakeLLM, so no
OpenAI calls are made. Also covers registry-level execution paths.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import HumanMessage

from agent.graph import create_support_graph, route_after_tools, route_after_triage
from agent.state import initial_state
from agent.tools.registry import _intent_fallback_calls, register_all

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Compilation & routing (pure logic, no LLM)
# ---------------------------------------------------------------------------


class TestGraphCompilation:
    def test_compiles(self, agent_graph):
        assert agent_graph is not None
        assert hasattr(agent_graph, "get_graph")

    def test_expected_nodes_present(self, agent_graph):
        names = set(agent_graph.get_graph().nodes)
        assert {"triage", "retrieve_context", "execute_tools",
                "generate_response", "escalate_to_human", "audit"} <= names


def _state(**overrides) -> dict:
    s = initial_state(session_id="t", customer_id="1")
    s.update(overrides)
    return s


class TestRouting:
    def test_faq_routes_to_rag(self):
        assert route_after_triage(_state(intent="faq")) == "retrieve_context"

    def test_order_status_routes_to_tools(self):
        assert route_after_triage(_state(intent="order_status")) == "execute_tools"

    def test_billing_routes_to_tools(self):
        assert route_after_triage(_state(intent="billing")) == "execute_tools"

    def test_general_defaults_to_rag(self):
        assert route_after_triage(_state(intent="general_inquiry")) == "retrieve_context"

    def test_escalation_flag_routes_to_human(self):
        assert route_after_triage(_state(intent="faq", escalation_needed=True)) == "escalate_to_human"

    def test_frustrated_sentiment_routes_to_human(self):
        assert route_after_triage(_state(intent="faq", sentiment="frustrated")) == "escalate_to_human"

    def test_route_after_tools_escalates_when_flagged(self):
        assert route_after_tools(_state(escalation_needed=True)) == "escalate_to_human"

    def test_route_after_tools_normal_path(self):
        assert route_after_tools(_state(escalation_needed=False)) == "generate_response"


# ---------------------------------------------------------------------------
# Full graph runs with mocked LLM
# ---------------------------------------------------------------------------


class TestAgentEndToEnd:
    async def test_happy_path_faq_produces_final_response(self, mock_llm, agent_graph):
        mock_llm.default = json.dumps({
            "intent": "faq", "sub_intent": None, "sentiment": "neutral",
            "urgency": "low", "escalation_required": False,
            "escalation_reason": None, "confidence": 0.9,
        })
        state = initial_state(session_id="s-e2e-1", customer_id="1")
        state["messages"] = [HumanMessage(content="What is your return policy?")]

        final = await agent_graph.ainvoke(state, {"configurable": {"thread_id": "s-e2e-1"}})

        assert final["final_response"], "agent must produce a final answer"
        assert final["intent"] == "faq"
        assert "retrieve_knowledge" in final["tools_used"]
        assert "generate_response" in final["tools_used"]
        assert final["escalation_needed"] is False
        assert any(isinstance(m, HumanMessage) for m in final["messages"])

    async def test_angry_sentiment_triggers_escalation(self, mock_llm, agent_graph):
        mock_llm.default = json.dumps({
            "intent": "complaint", "sub_intent": None, "sentiment": "very_frustrated",
            "urgency": "critical", "escalation_required": True,
            "escalation_reason": "Customer furious", "confidence": 0.9,
        })
        state = initial_state(session_id="s-e2e-2", customer_id="1")
        state["messages"] = [HumanMessage(content="THIS IS UNACCEPTABLE!")]

        final = await agent_graph.ainvoke(state, {"configurable": {"thread_id": "s-e2e-2"}})

        assert final["escalation_needed"] is True
        assert final["final_response"], "escalation must still give the customer a message"
        assert final["sentiment"] == "very_frustrated"

    async def test_intent_classifier_sets_turn_count(self, mock_llm, agent_graph):
        state = initial_state(session_id="s-e2e-3", customer_id="1")
        state["messages"] = [HumanMessage(content="hello")]
        final = await agent_graph.ainvoke(state, {"configurable": {"thread_id": "s-e2e-3"}})
        assert final["turn_count"] == 1

    async def test_session_thread_isolation(self, mock_llm, agent_graph):
        """Two different thread_ids keep independent message histories."""
        for tid in ("t-a", "t-b"):
            state = initial_state(session_id=tid, customer_id="1")
            state["messages"] = [HumanMessage(content=f"msg for {tid}")]
            await agent_graph.ainvoke(state, {"configurable": {"thread_id": tid}})
        snap_a = await agent_graph.aget_state({"configurable": {"thread_id": "t-a"}})
        contents = [str(m.content) for m in (snap_a.values or {}).get("messages", [])]
        assert any("msg for t-a" in c for c in contents)
        assert not any("msg for t-b" in c for c in contents)

    async def test_mocked_llm_receives_triage_prompt(self, mock_llm, agent_graph):
        state = initial_state(session_id="s-e2e-5", customer_id="1")
        state["messages"] = [HumanMessage(content="refund please")]
        await agent_graph.ainvoke(state, {"configurable": {"thread_id": "s-e2e-5"}})
        assert mock_llm.prompts, "LLM must have been invoked"


# ---------------------------------------------------------------------------
# Tool registry execution paths
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_register_all_registers_eight_tools(self):
        tools = register_all()
        expected = {
            "search_knowledge_base", "get_customer_info", "get_order_status",
            "create_support_ticket", "update_ticket_status", "draft_followup_email",
            "escalate_ticket", "get_ticket_history",
        }
        assert expected <= set(tools)

    def test_system_injection_overrides_llm_role(self):
        from agent.tools.registry import _system_injected_args

        state = {"customer_id": "42", "metadata": {"role": "customer"}}
        injected = _system_injected_args(state, "escalate_ticket")
        assert injected["requesting_customer_id"] == "42"
        assert injected["role"] == "customer"  # LLM cannot self-promote

    def test_fallback_calls_kb_for_faq(self):
        state = {"intent": "faq", "messages": [HumanMessage(content="return policy?")]}
        calls = _intent_fallback_calls(state)
        assert calls and calls[0]["name"] == "search_knowledge_base"
        assert "return policy" in calls[0]["args"]["query"]

    def test_fallback_skips_order_without_customer_id(self):
        state = {"intent": "order_status",
                 "messages": [HumanMessage(content="where is ORD-123?")]}
        assert _intent_fallback_calls(state) == []

    def test_fallback_builds_order_lookup(self):
        state = {"intent": "order_status", "customer_id": "7",
                 "messages": [HumanMessage(content="where is ORD-123 please?")]}
        calls = _intent_fallback_calls(state)
        assert calls and calls[0]["name"] == "get_order_status"
        assert calls[0]["args"]["order_number"].upper().startswith("ORD-")

    async def test_execute_tools_unknown_tool_returns_error_message(self, monkeypatch):
        from langchain_core.messages import AIMessage

        from agent.tools.registry import execute_tools

        state = initial_state(session_id="s-t", customer_id="1")
        state["messages"] = [AIMessage(content="", tool_calls=[
            {"id": "c1", "name": "nonexistent_tool", "args": {}},
        ])]
        result = await execute_tools(state)
        tool_msgs = [m for m in result["messages"]]
        assert tool_msgs and "Unknown tool" in tool_msgs[0].content

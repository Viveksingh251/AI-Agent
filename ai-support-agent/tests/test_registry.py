"""Smoke tests for the tool registry shared helpers (no DB required)."""

from __future__ import annotations

from agent.tools.registry import register_all
from agent.state import ALL_INTENTS, ESCALATING_SENTIMENTS


def test_all_intents_defined():
    assert {"faq", "order_status", "billing", "complaint"} <= ALL_INTENTS


def test_escalating_sentiments_defined():
    assert "very_frustrated" in ESCALATING_SENTIMENTS


def test_state_factory_defaults():
    from agent.state import initial_state

    s = initial_state(session_id="x", customer_id="5")
    assert s["customer_id"] == "5"
    assert s["escalation_needed"] is False
    assert s["tools_used"] == []
    assert s["messages"] == []


def test_register_all_idempotent():
    tools1 = register_all()
    tools2 = register_all()
    assert tools1 is tools2
    assert len(tools1) == 8

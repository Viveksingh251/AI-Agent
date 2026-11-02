"""Tool registry: every LangChain tool the support agent can call, plus an
``execute_tools`` graph node that runs them.

``execute_tools`` reads pending tool calls from the last AIMessage (standard
LangGraph tool-calling convention) *or* falls back to a simple intent → tool
mapping so the workflow also works without a tool-calling LLM.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool

from agent.state import SupportAgentState

logger = logging.getLogger("support_agent.tools.registry")

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_TOOLS: Dict[str, BaseTool] = {}


def register_all() -> Dict[str, BaseTool]:
    """Import and register every tool exactly once (idempotent)."""
    if ALL_TOOLS:
        return ALL_TOOLS

    from agent.tools.customer_tools import get_customer_info, get_order_status
    from agent.tools.kb_tools import search_knowledge_base
    from agent.tools.ticket_tools import (
        create_support_ticket,
        draft_followup_email,
        escalate_ticket,
        get_ticket_history,
        update_ticket_status,
    )

    for t in (
        search_knowledge_base,
        get_customer_info,
        get_order_status,
        create_support_ticket,
        update_ticket_status,
        draft_followup_email,
        escalate_ticket,
        get_ticket_history,
    ):
        ALL_TOOLS[t.name] = t

    logger.info("Registered %d agent tools: %s", len(ALL_TOOLS), sorted(ALL_TOOLS))
    return ALL_TOOLS


# Convenience export for binding to an LLM:
get_all_tools = register_all


def _system_injected_args(state: SupportAgentState, tool_name: str) -> Dict[str, Any]:
    """Authorization context injected server-side — the LLM can never set it."""
    injected: Dict[str, Any] = {
        "requesting_customer_id": state.get("customer_id"),
    }
    role = (state.get("metadata") or {}).get("role", "customer")
    injected["role"] = role
    # Escalation drafted by the AI assistant on the customer's behalf runs
    # with staff clearance only when the session itself is staffed.
    if tool_name == "escalate_ticket" and role not in ("support_agent", "admin"):
        injected["role"] = role  # keep customer clearance; deny() will fire
    return injected


# ---------------------------------------------------------------------------
# Graph node: execute_tools
# ---------------------------------------------------------------------------


async def execute_tools(state: SupportAgentState) -> Dict[str, Any]:
    """Run pending tool calls and append ToolMessages to the conversation.

    Two invocation paths:
    1. Tool-calling LLM: the last AIMessage carries ``tool_calls``.
    2. Fallback intent map: picks tools heuristically from classified intent
       and the latest human message.

    Every failure is caught per-tool so one broken call never crashes the run.
    """
    tools = register_all()
    messages = list(state.get("messages", []))
    calls: List[Dict[str, Any]] = []

    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    if last_ai is not None and getattr(last_ai, "tool_calls", None):
        for tc in last_ai.tool_calls:
            calls.append({"id": tc.get("id") or tc.get("tool_call_id") or "call_0", "name": tc["name"], "args": dict(tc.get("args") or {})})

    if not calls:
        calls = _intent_fallback_calls(state)

    new_messages: List[ToolMessage] = []
    tools_used = list(state.get("tools_used", []))

    for call in calls:
        name, args = call["name"], call["args"]
        tool_obj = tools.get(name)
        if tool_obj is None:
            new_messages.append(
                ToolMessage(content=json.dumps({"success": False, "error": f"Unknown tool '{name}'."}), tool_call_id=call["id"])
            )
            continue

        args.update(_system_injected_args(state, name))
        logger.info("execute_tools: running %s args=%s", name, args)
        try:
            result = await tool_obj.ainvoke(args)
            if isinstance(result, BaseTool):
                result = {"success": False, "error": "tool returned itself"}
            payload = result if isinstance(result, dict) else {"success": True, "data": result}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tool %s raised", name)
            payload = {"success": False, "error": f"Tool '{name}' failed unexpectedly."}

        tools_used.append(name)
        # Surface key results into state for downstream nodes.
        state_update: Dict[str, Any] = {"tools_used": tools_used, "messages": new_messages}
        if name == "get_customer_info" and payload.get("success"):
            state_update["customer_data"] = payload.get("customer")
        if name == "get_order_status" and payload.get("success"):
            state_update["order_data"] = payload.get("order")
        if name in ("create_support_ticket", "update_ticket_status", "escalate_ticket") and payload.get("success"):
            state_update["ticket_data"] = payload.get("ticket")
            state_update["current_ticket_id"] = (payload.get("ticket") or {}).get("id")
        if name == "escalate_ticket" and payload.get("success"):
            state_update["escalation_needed"] = True
        if name == "search_knowledge_base" and payload.get("success"):
            state_update["retrieved_docs"] = payload.get("results", [])

        new_messages.append(
            ToolMessage(content=json.dumps(payload, default=str), tool_call_id=call["id"])
        )
        # keep the reference for the final partial return
        state_update["messages"] = new_messages

    return {"messages": new_messages, "tools_used": tools_used, **_derived_state(state, calls, tools)}


def _derived_state(state: SupportAgentState, calls: List[Dict[str, Any]], tools: Dict[str, BaseTool]) -> Dict[str, Any]:
    """Nothing extra for now — placeholder keeps the merge surface explicit."""
    return {}


# ---------------------------------------------------------------------------
# Intent fallback (no tool-calling LLM available)
# ---------------------------------------------------------------------------

_FALLBACK_MAP: Dict[str, List[str]] = {
    "faq": ["search_knowledge_base"],
    "policy": ["search_knowledge_base"],
    "technical_issue": ["search_knowledge_base"],
    "order_status": ["get_order_status"],
    "billing": ["get_order_status", "search_knowledge_base"],
    "ticket_create": ["create_support_ticket"],
    "ticket_update": ["update_ticket_status"],
    "ticket_lookup": ["get_ticket_history"],
    "email_draft": ["draft_followup_email"],
    "escalation_request": ["escalate_ticket"],
}


def _intent_fallback_calls(state: SupportAgentState) -> List[Dict[str, Any]]:
    """Build naive tool calls from classified intent + regex-extracted IDs."""
    import re

    intent = state.get("intent") or "faq"
    text = ""
    for m in reversed(state.get("messages", [])):
        if getattr(m, "type", "") == "human":
            text = str(m.content)
            break

    calls: List[Dict[str, Any]] = []
    for name in _FALLBACK_MAP.get(intent, ["search_knowledge_base"]):
        args: Dict[str, Any] = {}
        if name == "search_knowledge_base":
            args["query"] = text[:300] or intent
        elif name == "get_order_status":
            m = re.search(r"\b(ORD[-_]?\w+|\d{6,})\b", text, re.IGNORECASE)
            if not m or not state.get("customer_id"):
                continue
            args = {"order_number": m.group(0), "customer_id": state["customer_id"]}
        elif name == "get_ticket_history":
            m = re.search(r"\b(TCK[-_]?\w+|TICK[-_]?\w+|\d+)\b", text, re.IGNORECASE)
            if not m:
                continue
            args = {"ticket_id": m.group(0)}
        elif name == "create_support_ticket":
            if not text:
                continue
            args = {"subject": text.strip()[:100], "description": text.strip(), "priority": "medium"}
        elif name in ("update_ticket_status", "escalate_ticket", "draft_followup_email"):
            continue  # need an explicit ticket id; skip rather than guess
        if args or name == "search_knowledge_base":
            calls.append({"id": f"fallback_{name}_{len(calls)}", "name": name, "args": args})
    return calls

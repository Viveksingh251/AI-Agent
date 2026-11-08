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
from app.input_sanitizer import redact_mapping
from app.monitoring import trace_tool_call, record_tool_usage, add_span_event
from app.resilience import get_breaker, run_tool_with_resilience
from app.config import get_settings

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


def _dependency_for_tool(tool_name: str) -> str:
    """Map a tool to the external dependency its failure is attributed to
    (drives which circuit breaker opens)."""
    if tool_name == "search_knowledge_base":
        return "vector_store"
    if tool_name == "draft_followup_email":
        return "llm"
    return "database"


def _system_injected_args(state: SupportAgentState, tool_name: str) -> Dict[str, Any]:
    """Authorization context injected server-side — the LLM can never set it.

    The role is taken ONLY from ``state['metadata']['role']`` (set by the
    authenticated API user) and normalised through the whitelist in
    ``agent.tools.base.normalize_role``: any unexpected value collapses to
    'customer', so a tampered or LLM-supplied role can never grant staff
    clearance.
    """
    from agent.tools.base import normalize_role  # noqa: PLC0415

    injected: Dict[str, Any] = {
        "requesting_customer_id": state.get("customer_id"),
    }
    injected["role"] = normalize_role((state.get("metadata") or {}).get("role"))
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
    merged: Dict[str, Any] = {}

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
        logger.info("execute_tools: running %s args=%s", name, redact_mapping(args))
        settings = get_settings()
        breaker = get_breaker(
            _dependency_for_tool(name),
            failure_threshold=settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            reset_timeout=settings.CIRCUIT_BREAKER_RESET_SECONDS,
            enabled=settings.CIRCUIT_BREAKER_ENABLED,
        )
        async with trace_tool_call(name, session_id=str(state.get("session_id")) if state.get("session_id") else None):
            try:
                result = await run_tool_with_resilience(
                    lambda: tool_obj.ainvoke(args),  # noqa: B023 — bound per iteration
                    tool_name=name,
                    timeout_seconds=settings.TOOL_TIMEOUT_SECONDS,
                    breaker=breaker,
                )
                if isinstance(result, BaseTool):
                    result = {"success": False, "error": "tool returned itself"}
                payload = result if isinstance(result, dict) else {"success": True, "data": result}
            except Exception as exc:  # noqa: BLE001 — timeouts, open circuits, tool errors
                logger.warning("Tool %s failed: %s: %s", name, type(exc).__name__, exc)
                payload = {
                    "success": False,
                    "error": f"Tool '{name}' is temporarily unavailable. Please try again shortly.",
                }
        record_tool_usage(name, success=bool(payload.get("success", False)))
        add_span_event("tool.completed", {"success": bool(payload.get("success", False))})

        tools_used.append(name)

        # Surface key results into state for downstream nodes. Updates are
        # accumulated into ``merged`` and returned once after the loop so
        # every tool's promotion (ticket ref, order data, …) survives.
        if name == "get_customer_info" and payload.get("success"):
            merged["customer_data"] = payload.get("customer")
        if name == "get_order_status" and payload.get("success"):
            merged["order_data"] = payload.get("order")
        if name in ("create_support_ticket", "update_ticket_status", "escalate_ticket") and payload.get("success"):
            merged["ticket_data"] = payload.get("ticket")
            ticket_ref = (payload.get("ticket") or {}).get("id")
            if ticket_ref:
                merged["current_ticket_id"] = str(ticket_ref)
        if name == "escalate_ticket" and payload.get("success"):
            merged["escalation_needed"] = True
        if name == "search_knowledge_base" and payload.get("success"):
            merged["retrieved_docs"] = payload.get("results", [])
        if name == "draft_followup_email" and payload.get("success"):
            merged["draft_email"] = payload.get("email")

        new_messages.append(
            ToolMessage(content=json.dumps(payload, default=str), tool_call_id=call["id"])
        )

    return {
        "messages": new_messages,
        "tools_used": tools_used,
        **merged,
        **_derived_state(state, calls, tools),
    }


def _derived_state(state: SupportAgentState, calls: List[Dict[str, Any]], tools: Dict[str, BaseTool]) -> Dict[str, Any]:
    """Nothing extra for now — placeholder keeps the merge surface explicit."""
    del state, calls, tools  # unused; placeholder extension point
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
    "general_inquiry": ["search_knowledge_base"],
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
            # These tools need an explicit ticket id. Use the ticket created
            # or looked up earlier in this session when available; otherwise
            # skip rather than guess an identifier.
            ticket_ref = state.get("current_ticket_id") or (state.get("ticket_data") or {}).get("ticket_number") or (state.get("ticket_data") or {}).get("id")
            if not ticket_ref:
                continue
            args = {"ticket_id": str(ticket_ref)}
            if name == "escalate_ticket":
                args["reason"] = text[:200] or "Escalated during live conversation"
            if name == "draft_followup_email":
                args["tone"] = "apologetic"
        if args or name == "search_knowledge_base":
            calls.append({"id": f"fallback_{name}_{len(calls)}", "name": name, "args": args})
    return calls

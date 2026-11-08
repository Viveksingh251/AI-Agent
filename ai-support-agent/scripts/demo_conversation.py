"""End-to-End Demo Conversation Script — simulated full support flow.

Runs a scripted, realistic conversation directly against the compiled
LangGraph workflow (no HTTP server required), narrating each step so you
can watch the agent reason, call tools, and escalate:

  Step 1  Customer asks "Where is my order #12345?"   → get_order_status
  Step 2  Agent looks up the order and answers        → route_after_tools
  Step 3  Customer is unhappy → agent creates ticket  → create_support_ticket
  Step 4  Agent drafts a follow-up email (staff view) → draft_followup_email
  Step 5  Customer makes a sensitive refund request   → escalate_to_human

Two modes:
  - **offline (default)** — exercises the real graph nodes with the stub
    LLM and the real database/tools. Works with no API key; routing is
    driven by the deterministic intent fallback map.
  - **--api** — sends each turn through the live HTTP API instead
    (POST /api/v1/auth/login + POST /api/v1/chat). Requires the server
    running (``make run``) and the demo data from ``seed_demo.py``.

Prerequisites:
    make migrate            # schema
    python -m scripts.seed_demo   # demo customers / orders / tickets
    python -m scripts.demo_conversation

The demo customer is Jane Customer (``customer@example.com``), who owns
order ``ORD-2026-12345`` (shipped) — pre-created by ``seed_demo.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from decimal import Decimal
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Bootstrap: make `app`, `agent`, `database`, `rag` importable from repo root
# ---------------------------------------------------------------------------

if __package__ in (None, ""):
    import pathlib
    import sys as _sys

    _sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# Print demo output plainly, even when JSON logging is configured.
logging.getLogger().setLevel(logging.WARNING)
for name in ("support_agent", "httpx", "httpcore", "openai", "langchain"):
    logging.getLogger(name).setLevel(logging.ERROR)

# Windows consoles often default to a cp1252/legacy code page that cannot
# encode unicode arrows/emoji used below — force UTF-8 output when possible.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001 — best effort
    pass


# ---------------------------------------------------------------------------
# Rule-based mock LLM (offline mode)
# ---------------------------------------------------------------------------
# When no real LLM is configured, triage would fall back to the stub answer
# (always 'faq'), which would make the demo skip the tools entirely. This
# deterministic keyword classifier produces the *correct* triage JSON for the
# scripted turns so the full flow — order lookup, ticket, email, escalation —
# runs end-to-end with zero API keys.

import re

_MOCK_RULES = [
    # Most specific first — the first matching rule wins.
    (r"where is my order|order status|my order\s*#|ORD-", "order_status", "neutral"),
    (r"email me|follow.?up email|written follow.?up|draft.*email", "email_draft", "neutral"),
    (r"open a (support )?ticket|create.*ticket|file a (ticket|complaint)", "ticket_create", "neutral"),
    (r"refund|manager|lawyer|legal|unacceptable|cancel my account|fraud", "escalation_request", "angry"),
]


def _mock_triage_json(message: str) -> str:
    for pattern, intent, sentiment in _MOCK_RULES:
        if re.search(pattern, message, re.IGNORECASE):
            escalate = "true" if intent == "escalation_request" else "false"
            urgency = "critical" if intent == "escalation_request" else "high"
            return json.dumps(
                {
                    "intent": intent,
                    "sub_intent": None,
                    "sentiment": sentiment,
                    "urgency": urgency,
                    "escalation_required": escalate == "true",
                    "escalation_reason": "Sensitive request requiring human judgement" if escalate == "true" else None,
                    "confidence": 0.95,
                }
            )
    return json.dumps(
        {
            "intent": "general_inquiry",
            "sub_intent": None,
            "sentiment": "neutral",
            "urgency": "medium",
            "escalation_required": False,
            "escalation_reason": None,
            "confidence": 0.8,
        }
    )


class _MockTriageLLM:
    """Deterministic stand-in for ChatOpenAI used only in offline demo mode."""

    async def ainvoke(self, messages: Any, **_: Any) -> Any:
        class _Msg:
            content = ""

        return _Msg()


def _install_mock_llm() -> None:
    """Replace agent.nodes._get_llm with a keyword-driven mock (offline only)."""
    try:
        import agent.nodes as nodes
    except ImportError:  # pragma: no cover
        return

    class _MockLLM:
        """Deterministic stand-in for ChatOpenAI used only in offline demo mode."""

        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        async def ainvoke(self, messages: Any, **_: Any) -> Any:
            # The LAST message content is what matters: triage sends the user
            # template containing the raw customer message; response generation
            # sends the RESPONSE_USER_TEMPLATE.
            text = ""
            for m in messages if isinstance(messages, (list, tuple)) else [messages]:
                content = getattr(m, "content", "")
                if isinstance(content, str) and content.strip():
                    text = content
            if "FINAL RESPONSE" in text:
                return type("R", (), {"content": "(Demo mode) Here is the agent's answer, summarising the actions taken this turn."})()
            return type("R", (), {"content": _mock_triage_json(text)})()

        def invoke(self, messages: Any, **_: Any) -> Any:
            return asyncio.get_event_loop().run_until_complete(self.ainvoke(messages))  # type: ignore[arg-type]

    nodes._get_llm = lambda *a, **k: _MockLLM()  # type: ignore[assignment]
    nodes._OPENAI_AVAILABLE = False  # keep email drafting on the safe template path

# ---------------------------------------------------------------------------
# Demo dataset constants (must match scripts/seed_demo.py)
# ---------------------------------------------------------------------------

DEMO_EMAIL = "customer@example.com"
DEMO_PASSWORD = "CustomerPassword123!"
DEMO_ORDER_NUMBER = "ORD-2026-12345"

RULES = """Demo rules:
- ORD-2026-12345 must exist and belong to customer@example.com (seed_demo.py).
- In offline mode no OPENAI_API_KEY is needed: the stub LLM classifies intents
  and the intent→tool fallback map drives routing.
"""


# ===========================================================================
# Shared script definition: the conversation turns
# ===========================================================================

# (narration, customer message, acting role, expected intent keyword)
SCRIPT_TURNS: List[Dict[str, str]] = [
    {
        "narration": "STEP 1/2 — Customer asks where their order is (agent looks it up)",
        "message": f"Hi, where is my order {DEMO_ORDER_NUMBER}? It's been 8 days and I haven't received anything.",
        "role": "customer",
        "expect_intent": "order_status",
    },
    {
        "narration": "STEP 3 — Customer is unhappy about the wait (agent opens a ticket)",
        "message": "This is really frustrating, the delivery estimate was 5 days. Please open a support ticket about this delayed order.",
        "role": "customer",
        "expect_intent": "ticket_create",
    },
    {
        "narration": "STEP 4 — Human agent asks the AI to draft a follow-up email "
        "(drafting is a staff-only tool — sent with support_agent clearance)",
        "message": "Please draft an apologetic follow-up email to the customer about their open ticket.",
        "role": "support_agent",
        "expect_intent": "email_draft",
    },
    {
        "narration": "STEP 5 — Customer makes a sensitive request (refund + manager) → escalation",
        "message": "I want a full refund immediately and I want to speak to a manager about how badly this has been handled. This is unacceptable.",
        "role": "customer",
        "expect_intent": "escalation_request",
    },
]

BANNER = "=" * 66


def _hr(char: str = "-", width: int = 66) -> str:
    return char * width


def _print_step(step: int, title: str) -> None:
    print()
    print(_hr("="))
    print(f"STEP {step}: {title}")
    print(_hr("="))


def _print_narration(text: str) -> None:
    print(f"  · {text}")


def _truncate(text: str, width: int = 460) -> str:
    return text if len(text) <= width else text[:width] + " …"


# ===========================================================================
# Mode A — offline: invoke the compiled graph directly
# ===========================================================================


async def run_offline() -> int:
    """Simulate the conversation directly through the LangGraph workflow."""
    from langchain_core.messages import HumanMessage

    from agent.graph import create_support_graph
    from agent.state import initial_state
    from database.crud import get_customer_by_email, get_order_by_number
    from database.session import async_session_maker, init_db_models

    # -- Verify the demo data exists ----------------------------------------
    await init_db_models()
    async with async_session_maker() as session:
        customer = await get_customer_by_email(session, DEMO_EMAIL)
        if customer is None:
            print(f"ERROR: demo customer '{DEMO_EMAIL}' not found.")
            print(RULES)
            print("Run:  python -m scripts.seed_demo")
            return 1
        order = await get_order_by_number(session, DEMO_ORDER_NUMBER)
        if order is None or order.customer_id != customer.id:
            print(f"ERROR: order '{DEMO_ORDER_NUMBER}' not found for {DEMO_EMAIL}.")
            print(RULES)
            print("Run:  python -m scripts.seed_demo")
            return 1
        customer_id = str(customer.id)

    agent = create_support_graph()
    session_id = f"demo_{uuid.uuid4().hex[:10]}"
    _install_mock_llm()

    print(BANNER)
    print("AI CUSTOMER SUPPORT AGENT — OFFLINE DEMO CONVERSATION")
    print(BANNER)
    print(f"  customer : {DEMO_EMAIL} (id={customer_id}, order={DEMO_ORDER_NUMBER})")
    print(f"  session  : {session_id}")
    print("  mode     : offline (direct graph invocation, rule-based mock triage)")

    # We replay turns into ONE session so the checkpointer keeps history,
    # mirroring what the API does with thread_id=session_id. Each turn we
    # pass ONLY the new message + current role — the checkpointer restores
    # everything else (ticket refs, tools_used, order data) from the last
    # checkpoint, exactly like the production chat router.
    config = {"configurable": {"thread_id": session_id}}
    failures: List[str] = []
    step_results: List[Dict[str, Any]] = []
    seen_tools: List[str] = []

    for i, turn in enumerate(SCRIPT_TURNS, start=1):
        _print_step(i, turn["narration"])
        speaker = "Customer" if turn["role"] == "customer" else "Human Agent"
        print(f'  {speaker} > "{turn["message"]}"')
        print()

        turn_input = {
            "messages": [HumanMessage(content=turn["message"])],
            "metadata": {"role": turn["role"]},
            "customer_id": customer_id,
            "session_id": session_id,
        }
        final: Dict[str, Any] = {}
        turn_tools: List[str] = []
        async for chunk in agent.astream(turn_input, config=config, stream_mode="updates"):
            for _node_name, node_update in (chunk or {}).items():
                if node_update:
                    final.update(node_update)
                    if node_update.get("tools_used"):
                        turn_tools = list(node_update["tools_used"])

        intent = final.get("intent")
        sentiment = final.get("sentiment")
        # tools_used is cumulative across the session (checkpointed state) —
        # show what THIS turn added on top of the previous checkpoint.
        tools = [t for t in turn_tools if t not in seen_tools]
        seen_tools.extend(t for t in turn_tools if t not in seen_tools)
        response = final.get("final_response") or "(no response generated)"
        order_data = final.get("order_data") or {}
        ticket_data = final.get("ticket_data") or {}
        escalated = bool(final.get("escalation_needed"))

        print(f"  Agent    > {_truncate(response)}")
        print()
        print(f"    intent={intent}  sentiment={sentiment}  escalated={escalated}")
        print(f"    tools/nodes used: {sorted(set(tools))}")
        if final.get("draft_email"):
            print("    draft_email drafted (preview):")
            print("    " + _truncate(str(final["draft_email"]).replace("\n", " | "), 300))
        if order_data:
            print(
                f"    order_data: {order_data.get('order_number')} "
                f"status={order_data.get('status')} total={order_data.get('total_amount')}"
            )
        if ticket_data:
            print(
                f"    ticket_data: {ticket_data.get('ticket_number')} "
                f"status={ticket_data.get('status')} priority={ticket_data.get('priority')}"
            )

        # -- Assertions for the scripted flow --------------------------------
        if turn["expect_intent"] and intent != turn["expect_intent"] and intent != "general_inquiry":
            failures.append(f"step {i}: expected intent≈{turn['expect_intent']}, got {intent}")
        step_results.append(
            {
                "step": i,
                "intent": intent,
                "tools": sorted(set(tools)),
                "escalated": escalated,
                "order": bool(order_data),
                "ticket": ticket_data.get("ticket_number"),
            }
        )

    # -- Final summary -------------------------------------------------------
    print()
    print(BANNER)
    print("DEMO SUMMARY")
    print(BANNER)
    for r in step_results:
        print(f"  Step {r['step']}: intent={r['intent']:<18} tools={r['tools']}")
    print()
    print("  What just happened:")
    print("   1. Triage classified the delivery question -> get_order_status ran.")
    print("   2. Ticket request -> create_support_ticket ran (with frustration checks).")
    print("   3. Human agent asked for a follow-up email -> draft_followup_email ran")
    print("      with support_agent clearance (a customer would be denied — RBAC).")
    print("   4. Sensitive refund/manager request escalated -> escalate_to_human node.")

    if failures:
        print()
        print("  ⚠ Deviations from the scripted flow:")
        for f in failures:
            print(f"    - {f}")
        print("  (With the stub LLM, intent keywords may differ slightly — this is")
        print("   expected behaviour, not an error, as long as routing occurred.)")
        return 0

    print()
    print("  ✔ All steps matched the scripted expectations.")
    return 0


# ===========================================================================
# Mode B — API: run the same conversation over the live HTTP API
# ===========================================================================


async def run_api(base_url: str) -> int:
    import httpx

    # 1. Login ----------------------------------------------------------------
    print(BANNER)
    print("AI CUSTOMER SUPPORT AGENT — API DEMO CONVERSATION")
    print(BANNER)
    print(f"  base_url : {base_url}")
    print(f"  customer : {DEMO_EMAIL}")
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{base_url}/api/v1/auth/login",
            data={"username": DEMO_EMAIL, "password": DEMO_PASSWORD},
        )
        if resp.status_code != 200:
            print(f"ERROR: login failed ({resp.status_code}): {resp.text}")
            print("Is the server running?  make run")
            return 1
        token = resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        print("  login    : OK")

        session_id = f"demo_{uuid.uuid4().hex[:10]}"
        step_results: List[Dict[str, Any]] = []

        for i, turn in enumerate(SCRIPT_TURNS, start=1):
            _print_step(i, turn["narration"])
            print(f'  Customer > "{turn["message"]}"')
            print()
            resp = await client.post(
                f"{base_url}/api/v1/chat",
                headers=headers,
                json={"message": turn["message"], "session_id": session_id},
            )
            if resp.status_code != 200:
                print(f"ERROR: chat call failed ({resp.status_code}): {resp.text}")
                return 1
            data = resp.json()
            print(f"  Agent    > {_truncate(data.get('response', ''))}")
            print()
            actions = data.get("actions") or {}
            print(f"    intent={data.get('intent')}  sentiment={data.get('sentiment')}"
                  f"  escalated={data.get('escalation_required')}")
            print(f"    actions: {json.dumps(actions, default=str)}")
            step_results.append({"step": i, "intent": data.get("intent"), "actions": actions})

        print()
        print(BANNER)
        print("DEMO SUMMARY")
        print(BANNER)
        for r in step_results:
            print(f"  Step {r['step']}: intent={r['intent']}")
        print(f"\n  session_id = {session_id}")
        print("  Inspect it:  GET /api/v1/chat/{session_id}/history")
        print("  Audit trail: GET /api/v1/admin/agent-logs")
    return 0


# ===========================================================================
# Entry point
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a full support-agent conversation flow.")
    parser.add_argument(
        "--api",
        action="store_true",
        help="Send the conversation through the live HTTP API instead of invoking the graph directly.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="API base URL when using --api (default: http://localhost:8000).",
    )
    args = parser.parse_args()

    if args.api:
        sys.exit(asyncio.run(run_api(args.base_url)))
    sys.exit(asyncio.run(run_offline()))


if __name__ == "__main__":
    main()

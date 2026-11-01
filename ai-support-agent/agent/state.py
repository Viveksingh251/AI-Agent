"""State schema for the LangGraph Customer Support Agent.

This module defines the single source of truth for all data flowing through
the StateGraph.  Every node reads from and writes to a ``SupportAgentState``
dict — LangGraph merges partial-update dicts returned by nodes using the
declared reducers.

Design notes
------------
- ``messages`` uses the built-in ``add_messages`` reducer so each node can
  append messages without overwriting the full list.
- All other fields use the default *replace* reducer (last-write wins) which
  is correct for scalar status values that nodes mutate one at a time.
- Optional fields default to ``None``/``[]``/``{}`` so nodes can be entered
  in any order without KeyError.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Intent taxonomy — kept here so nodes, prompts, and routing all share one
# canonical set of values.
# ---------------------------------------------------------------------------

INTENT_FAQ = "faq"
INTENT_ORDER_STATUS = "order_status"
INTENT_BILLING = "billing"
INTENT_TECHNICAL = "technical_issue"
INTENT_COMPLAINT = "complaint"
INTENT_TICKET_CREATE = "ticket_create"
INTENT_TICKET_UPDATE = "ticket_update"
INTENT_ESCALATION = "escalation_request"
INTENT_EMAIL_DRAFT = "email_draft"
INTENT_PERFORMANCE = "performance_summary"
INTENT_GENERAL = "general_inquiry"

ALL_INTENTS = {
    INTENT_FAQ,
    INTENT_ORDER_STATUS,
    INTENT_BILLING,
    INTENT_TECHNICAL,
    INTENT_COMPLAINT,
    INTENT_TICKET_CREATE,
    INTENT_TICKET_UPDATE,
    INTENT_ESCALATION,
    INTENT_EMAIL_DRAFT,
    INTENT_PERFORMANCE,
    INTENT_GENERAL,
}

# Intents that immediately trigger the RAG retrieve_knowledge node.
RAG_INTENTS = {INTENT_FAQ, INTENT_TECHNICAL, "policy"}

# Intents that require tool execution (DB lookups, mutations).
TOOL_INTENTS = {
    INTENT_ORDER_STATUS,
    INTENT_BILLING,
    INTENT_TICKET_CREATE,
    INTENT_TICKET_UPDATE,
}

# Intents that immediately escalate to a human (besides explicit flags).
ESCALATION_INTENTS = {INTENT_ESCALATION, INTENT_COMPLAINT}

# Sentiment values that auto-trigger escalation.
ESCALATING_SENTIMENTS = {"angry", "very_frustrated"}


# ---------------------------------------------------------------------------
# SupportAgentState
# ---------------------------------------------------------------------------


class SupportAgentState(TypedDict, total=False):
    """Central state propagated through every node of the support graph.

    Field descriptions
    ------------------
    messages
        Accumulated conversation messages (HumanMessage / AIMessage /
        ToolMessage).  Uses the ``add_messages`` reducer — nodes *append*
        to this list, never replace it.

    customer_id
        UUID or opaque identifier for the authenticated customer.  ``None``
        for anonymous / pre-login conversations.

    session_id
        Stable identifier for the current chat session (used for
        checkpointing and audit logging).

    intent
        Classified customer intent from the ``intent_classifier`` node.
        One of the ``INTENT_*`` constants above.

    sub_intent
        Optional fine-grained intent qualifier (e.g. ``"refund_request"``
        within ``INTENT_BILLING``).

    sentiment
        Customer sentiment detected by the classifier:
        ``positive | neutral | frustrated | angry | very_frustrated``

    urgency
        Inferred urgency level: ``low | medium | high | critical``

    current_ticket_id
        ID of the active support ticket created or looked up during this
        session.  ``None`` until a ticket is created / found.

    order_id
        ID of the order being discussed, populated by ``lookup_order``.

    escalation_needed
        ``True`` if the conversation must be handed off to a human agent.

    escalation_reason
        Human-readable explanation for the escalation decision.

    tools_used
        Ordered list of tool / node names invoked during this turn.
        Useful for audit logs and performance analysis.

    retrieved_docs
        Knowledge-base documents returned by ``retrieve_knowledge``.
        Each entry is a dict with ``content``, ``score``, ``metadata``.

    customer_data
        Profile and account data returned by ``lookup_customer``.

    order_data
        Order details returned by ``lookup_order``.

    ticket_data
        Ticket details created or fetched during the session.

    draft_email
        Email body drafted by the ``draft_email`` node, pending approval.

    draft_response
        Intermediate response draft before final polish.

    final_response
        The polished, customer-facing response string produced by
        ``generate_response``.

    confidence_score
        Self-reported confidence of the last LLM call (0.0 – 1.0).

    turn_count
        Number of full agent reasoning turns taken in this session.

    performance_data
        Admin-only summary metrics produced by ``performance_summary``.

    metadata
        Flexible key-value store for arbitrary context (feature flags,
        A/B experiment IDs, request tracing headers, etc.).

    error
        Last error message if a node failed gracefully.
    """

    # ── Core conversation ──────────────────────────────────────────────────
    messages: Annotated[List[BaseMessage], add_messages]

    # ── Identity ───────────────────────────────────────────────────────────
    customer_id: Optional[str]
    session_id: str

    # ── Classification ─────────────────────────────────────────────────────
    intent: Optional[str]
    sub_intent: Optional[str]
    sentiment: Optional[str]
    urgency: Optional[str]

    # ── References ─────────────────────────────────────────────────────────
    current_ticket_id: Optional[str]
    order_id: Optional[str]

    # ── Escalation ─────────────────────────────────────────────────────────
    escalation_needed: bool
    escalation_reason: Optional[str]

    # ── Tool tracking ──────────────────────────────────────────────────────
    tools_used: List[str]

    # ── Data payloads ──────────────────────────────────────────────────────
    retrieved_docs: List[Dict[str, Any]]
    customer_data: Optional[Dict[str, Any]]
    order_data: Optional[Dict[str, Any]]
    ticket_data: Optional[Dict[str, Any]]

    # ── Generated content ──────────────────────────────────────────────────
    draft_email: Optional[str]
    draft_response: Optional[str]
    final_response: str

    # ── Telemetry ──────────────────────────────────────────────────────────
    confidence_score: float
    turn_count: int
    performance_data: Optional[Dict[str, Any]]

    # ── Miscellaneous ──────────────────────────────────────────────────────
    metadata: Dict[str, Any]
    error: Optional[str]


def initial_state(
    session_id: str,
    customer_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> SupportAgentState:
    """Return a fully-initialised default ``SupportAgentState``.

    Use this factory to create the first state dict passed into the graph
    so no node encounters a missing key.

    Args:
        session_id: Unique session identifier (e.g. UUID).
        customer_id: Authenticated customer's ID, or ``None`` for anonymous.
        metadata: Extra context to attach (trace IDs, feature flags, etc.).

    Returns:
        A ``SupportAgentState`` dict with all fields populated with safe
        defaults.
    """
    return SupportAgentState(
        messages=[],
        customer_id=customer_id,
        session_id=session_id,
        intent=None,
        sub_intent=None,
        sentiment=None,
        urgency=None,
        current_ticket_id=None,
        order_id=None,
        escalation_needed=False,
        escalation_reason=None,
        tools_used=[],
        retrieved_docs=[],
        customer_data=None,
        order_data=None,
        ticket_data=None,
        draft_email=None,
        draft_response=None,
        final_response="",
        confidence_score=0.0,
        turn_count=0,
        performance_data=None,
        metadata=metadata or {},
        error=None,
    )

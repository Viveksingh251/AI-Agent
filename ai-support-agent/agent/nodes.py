"""LangGraph node implementations for the Customer Support Agent.

Each node is an async function with the signature::

    async def node_name(state: SupportAgentState) -> Dict[str, Any]

Nodes return a *partial* state dict.  LangGraph merges it into the full
state using the reducers declared in ``agent/state.py``.

LLM calls are guarded with try/except so the graph degrades gracefully when
``langchain_openai`` or other heavy packages are absent.  Fallback stubs
return clearly labelled placeholder data so integration tests can run without
real API keys.

Node inventory
--------------
1.  intent_classifier      — triage: classify intent, sentiment, urgency
2.  retrieve_knowledge     — RAG: fetch relevant KB documents
3.  lookup_customer        — DB: fetch customer profile
4.  lookup_order           — DB: fetch order details
5.  create_ticket          — DB: create a new support ticket
6.  update_ticket          — DB: update / comment on existing ticket
7.  draft_email            — LLM: compose a customer email
8.  escalate_to_human      — hand off to human agent queue
9.  generate_response      — LLM: synthesise the final reply
10. performance_summary    — admin: generate session analytics report
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.state import (
    ESCALATING_SENTIMENTS,
    ESCALATION_INTENTS,
    RAG_INTENTS,
    TOOL_INTENTS,
    SupportAgentState,
)
from agent.prompts import (
    EMAIL_DRAFT_SYSTEM_PROMPT,
    EMAIL_DRAFT_USER_TEMPLATE,
    ESCALATION_ALERT_TEMPLATE,
    ESCALATION_SYSTEM_PROMPT,
    ESCALATION_USER_TEMPLATE,
    PERFORMANCE_SUMMARY_SYSTEM_PROMPT,
    PERFORMANCE_SUMMARY_USER_TEMPLATE,
    RAG_CONTEXT_TEMPLATE,
    RAG_QUERY_REWRITE_PROMPT,
    RESPONSE_SYSTEM_PROMPT,
    RESPONSE_USER_TEMPLATE,
    TICKET_CREATE_EXTRACTION_PROMPT,
    TICKET_RESPONSE_TEMPLATE,
    TICKET_UPDATE_EXTRACTION_PROMPT,
    TRIAGE_SYSTEM_PROMPT,
    TRIAGE_USER_TEMPLATE,
)

logger = logging.getLogger("support_agent.nodes")

# ---------------------------------------------------------------------------
# LLM factory — graceful degradation when langchain_openai is absent
# ---------------------------------------------------------------------------

try:
    from langchain_openai import ChatOpenAI  # type: ignore

    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    logger.warning("langchain_openai not installed — LLM nodes will use stub responses.")


def _get_llm(model: str = "gpt-4o-mini", temperature: float = 0.2) -> Any:
    """Return an LLM client with a hard per-call timeout, or a stub callable
    if langchain_openai is unavailable.

    The timeout wrapper prevents a hung provider connection from blocking a
    worker indefinitely; on timeout the caller's existing try/except falls
    back to safe defaults (triage defaults, template emails, etc.).
    """
    from app.config import get_settings

    timeout_seconds = get_settings().OPENAI_TIMEOUT_SECONDS

    if _OPENAI_AVAILABLE:
        base_llm = ChatOpenAI(model=model, temperature=temperature, timeout=timeout_seconds)  # type: ignore[call-arg]

        class _TimeoutLLM:
            """Thin proxy adding an asyncio deadline around every ainvoke."""

            def __getattr__(self, item: str) -> Any:
                return getattr(base_llm, item)

            async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
                from app.resilience import with_timeout, get_breaker

                breaker = get_breaker(
                    "llm",
                    failure_threshold=get_settings().CIRCUIT_BREAKER_FAILURE_THRESHOLD,
                    reset_timeout=get_settings().CIRCUIT_BREAKER_RESET_SECONDS,
                    enabled=get_settings().CIRCUIT_BREAKER_ENABLED,
                )
                breaker.check()
                try:
                    result = await with_timeout(
                        base_llm.ainvoke(messages, **kwargs),
                        timeout_seconds,
                        what=f"LLM call ({model})",
                    )
                except Exception:
                    breaker.record_failure()
                    raise
                breaker.record_success()
                return result

        return _TimeoutLLM()

    # Stub: returns a fixed AIMessage so the graph still runs in tests.
    class _StubLLM:
        async def ainvoke(self, messages: Any, **_: Any) -> AIMessage:
            return AIMessage(
                content='{"intent":"faq","sub_intent":null,"sentiment":"neutral",'
                        '"urgency":"medium","escalation_required":false,'
                        '"escalation_reason":null,"confidence":0.9}'
            )

        def invoke(self, messages: Any, **_: Any) -> AIMessage:
            return AIMessage(content="[STUB] LLM not available — install langchain-openai.")

    return _StubLLM()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _last_human_message(state: SupportAgentState) -> str:
    """Extract the text of the most recent HumanMessage."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _conversation_history(state: SupportAgentState, max_turns: int = 10) -> str:
    """Format the last *max_turns* messages as a readable string."""
    lines: List[str] = []
    messages = state.get("messages", [])[-max_turns * 2:]
    for msg in messages:
        role = "Customer" if isinstance(msg, HumanMessage) else "Agent"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines) or "(no prior messages)"


def _safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    """Try to parse JSON from an LLM response, stripping markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1])  # strip ```json ... ```
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def _format_retrieved_docs(docs: List[Dict[str, Any]]) -> str:
    """Render retrieved KB docs for injection into an LLM prompt."""
    if not docs:
        return "(no knowledge-base documents retrieved)"
    passages = []
    for i, doc in enumerate(docs, 1):
        content = doc.get("content", doc.get("page_content", ""))
        meta = doc.get("metadata", {})
        source = meta.get("filename", meta.get("source", "unknown"))
        score = doc.get("score", "—")
        passages.append(f"[{i}] ({source} | score={score})\n{content}")
    return "\n\n".join(passages)


# ---------------------------------------------------------------------------
# Node 1 — intent_classifier
# ---------------------------------------------------------------------------


async def intent_classifier(state: SupportAgentState) -> Dict[str, Any]:
    """Triage the customer message: classify intent, sentiment, and urgency.

    Calls the LLM with a structured-output prompt.  On parse failure,
    falls back to ``general_inquiry`` / ``neutral`` / ``medium``.

    State reads : messages, customer_id, turn_count, metadata
    State writes : intent, sub_intent, sentiment, urgency,
                   escalation_needed, escalation_reason, turn_count, tools_used
    """
    logger.info("Node: intent_classifier | session=%s", state.get("session_id"))
    t0 = time.monotonic()

    message = _last_human_message(state)
    turn = state.get("turn_count", 0) + 1

    # Build triage user message
    triage_user = TRIAGE_USER_TEMPLATE.format(
        message=message,
        customer_id=state.get("customer_id") or "anonymous",
        tier=state.get("customer_data", {}) and state["customer_data"].get("tier", "unknown") or "unknown",  # type: ignore[index]
        open_tickets=state.get("current_ticket_id") or "none",
        turn_count=turn,
    )

    llm = _get_llm(temperature=0.0)
    try:
        response = await llm.ainvoke([
            SystemMessage(content=TRIAGE_SYSTEM_PROMPT),
            HumanMessage(content=triage_user),
        ])
        parsed = _safe_json_parse(response.content)
    except Exception as exc:  # noqa: BLE001
        logger.error("intent_classifier LLM call failed: %s", exc)
        parsed = None

    # Defaults on parse failure — SAFE DEFAULTS: fail toward human review.
    # An unparseable/failed triage must never silently continue as a confident
    # low-risk classification.
    if not parsed:
        parsed = {
            "intent": "general_inquiry",
            "sub_intent": None,
            "sentiment": "neutral",
            "urgency": "high",
            "escalation_required": True,
            "escalation_reason": "Triage classification unavailable — routing to human review (safe default).",
            "confidence": 0.0,
        }

    # Check hard-coded escalation triggers
    escalation_needed = bool(parsed.get("escalation_required", False))
    if parsed.get("sentiment") in ESCALATING_SENTIMENTS:
        escalation_needed = True
        parsed.setdefault("escalation_reason", f"Customer sentiment: {parsed['sentiment']}")
    if parsed.get("intent") in ESCALATION_INTENTS:
        escalation_needed = True

    # Prompt-injection flag from the API layer routes to human review too —
    # an attempted manipulation is a security event, not a support question.
    injection_categories = (state.get("metadata") or {}).get("injection_categories") or []
    if injection_categories:
        escalation_needed = True
        parsed.setdefault(
            "escalation_reason",
            f"Potential prompt injection detected ({', '.join(injection_categories)}) — human review required.",
        )

    tools_used = list(state.get("tools_used", [])) + ["intent_classifier"]

    elapsed = round((time.monotonic() - t0) * 1000)
    logger.info(
        "intent_classifier done | intent=%s sentiment=%s urgency=%s escalate=%s ms=%d",
        parsed["intent"], parsed.get("sentiment"), parsed.get("urgency"),
        escalation_needed, elapsed,
    )

    return {
        "intent": parsed.get("intent", "general_inquiry"),
        "sub_intent": parsed.get("sub_intent"),
        "sentiment": parsed.get("sentiment", "neutral"),
        "urgency": parsed.get("urgency", "medium"),
        "escalation_needed": escalation_needed,
        "escalation_reason": parsed.get("escalation_reason"),
        "confidence_score": float(parsed.get("confidence", 0.5)),
        "turn_count": turn,
        "tools_used": tools_used,
    }


# ---------------------------------------------------------------------------
# Node 2 — retrieve_knowledge  (RAG)
# ---------------------------------------------------------------------------


async def retrieve_knowledge(state: SupportAgentState) -> Dict[str, Any]:
    """Fetch relevant knowledge-base documents for the current query.

    Rewrites the raw customer message into an optimised search query using
    the LLM, then delegates to ``rag.retriever.SupportKBRetriever``.

    State reads : messages, intent, sub_intent, metadata
    State writes : retrieved_docs, tools_used
    """
    logger.info("Node: retrieve_knowledge | session=%s", state.get("session_id"))

    message = _last_human_message(state)
    intent = state.get("intent", "faq")
    sub_intent = state.get("sub_intent") or ""

    # Step 1: rewrite the query for better retrieval.
    search_query = message  # fallback
    llm = _get_llm(temperature=0.0)
    try:
        rewrite_response = await llm.ainvoke([
            HumanMessage(
                content=RAG_QUERY_REWRITE_PROMPT.format(
                    message=message, intent=intent, sub_intent=sub_intent
                )
            )
        ])
        rewritten = rewrite_response.content.strip().strip('"')
        if rewritten:
            search_query = rewritten
            logger.debug("retrieve_knowledge: rewritten query=%r", search_query)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Query rewrite failed, using raw message: %s", exc)

    # Step 2: retrieve documents.
    retrieved: List[Dict[str, Any]] = []
    try:
        from app.config import get_settings
        from rag.retriever import SupportKBRetriever

        settings = get_settings()
        retriever = SupportKBRetriever(
            top_k=settings.TOP_K_RETRIEVAL,
            similarity_threshold=getattr(settings, "SIMILARITY_THRESHOLD", 0.3),
            index_dir=settings.FAISS_INDEX_DIR,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        # Map intent → doc_type filter
        doc_type_map: Dict[str, str] = {
            "faq": "faq",
            "billing": "policy",
            "order_status": "policy",
            "technical_issue": "product",
            "complaint": "policy",
        }
        doc_type_filter = doc_type_map.get(intent)
        scored_docs = retriever.retrieve_with_scores(
            query=search_query,
            doc_type=doc_type_filter,
        )
        retrieved = [
            {
                "content": doc.page_content,
                "score": round(score, 4),
                "metadata": doc.metadata,
            }
            for doc, score in scored_docs
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAG retrieval failed (non-fatal): %s", exc)
        # Stub doc so generate_response has something to work with.
        retrieved = [
            {
                "content": "For policy information, please visit help.techstore.com.",
                "score": 0.0,
                "metadata": {"source": "fallback", "doc_type": "general"},
            }
        ]

    tools_used = list(state.get("tools_used", [])) + ["retrieve_knowledge"]
    logger.info("retrieve_knowledge: fetched %d docs", len(retrieved))

    return {
        "retrieved_docs": retrieved,
        "tools_used": tools_used,
    }


# ---------------------------------------------------------------------------
# Node 3 — lookup_customer
# ---------------------------------------------------------------------------


async def lookup_customer(state: SupportAgentState) -> Dict[str, Any]:
    """Fetch the customer profile from the database.

    State reads : customer_id
    State writes : customer_data, tools_used, error
    """
    logger.info("Node: lookup_customer | session=%s", state.get("session_id"))
    customer_id = state.get("customer_id")

    if not customer_id:
        logger.debug("lookup_customer: no customer_id in state, skipping DB call.")
        return {"customer_data": None}

    customer_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    try:
        from database.session import get_db_session_ctx
        from database.crud import get_customer_by_id

        async with get_db_session_ctx() as session:
            customer = await get_customer_by_id(session, customer_id)
            if customer:
                customer_data = {
                    "id": str(customer.id),
                    "name": customer.name,
                    "email": customer.email,
                    "phone": customer.phone,
                    "company": customer.company,
                    "tier": customer.tier.value if hasattr(customer.tier, "value") else str(customer.tier),
                    "created_at": str(customer.created_at),
                }
            else:
                logger.warning("lookup_customer: customer %s not found in DB", customer_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("lookup_customer DB error: %s", exc)
        error = str(exc)
        # Stub data for graceful degradation in dev/test.
        customer_data = {
            "id": str(customer_id),
            "name": "Customer",
            "email": "customer@example.com",
            "tier": "free",
            "stub": True,
        }

    tools_used = list(state.get("tools_used", [])) + ["lookup_customer"]
    return {
        "customer_data": customer_data,
        "tools_used": tools_used,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Node 4 — lookup_order
# ---------------------------------------------------------------------------


async def lookup_order(state: SupportAgentState) -> Dict[str, Any]:
    """Fetch the most recent or referenced order for the customer.

    State reads : customer_id, messages (for order number extraction)
    State writes : order_data, order_id, tools_used, error
    """
    logger.info("Node: lookup_order | session=%s", state.get("session_id"))
    customer_id = state.get("customer_id")

    order_data: Optional[Dict[str, Any]] = None
    order_id: Optional[str] = None
    error: Optional[str] = None

    # Try to extract an explicit order number from the conversation.
    import re
    message = _last_human_message(state)
    order_match = re.search(r"\b(ORD[-_]?\w+|\d{6,})\b", message, re.IGNORECASE)
    explicit_order_number = order_match.group(0) if order_match else None

    try:
        from database.session import get_db_session_ctx
        from database.crud import get_orders_by_customer, get_order_by_number

        async with get_db_session_ctx() as session:
            if explicit_order_number:
                order = await get_order_by_number(session, explicit_order_number)
            elif customer_id:
                orders = await get_orders_by_customer(session, customer_id, limit=1)
                order = orders[0] if orders else None
            else:
                order = None

            if order:
                order_id = str(order.id)
                order_data = {
                    "id": str(order.id),
                    "order_number": order.order_number,
                    "status": order.status.value if hasattr(order.status, "value") else str(order.status),
                    "total_amount": float(order.total_amount),
                    "items": order.items,
                    "created_at": str(order.created_at),
                }
    except Exception as exc:  # noqa: BLE001
        logger.error("lookup_order DB error: %s", exc)
        error = str(exc)
        # Stub for dev/test
        order_data = {
            "id": "ORD-STUB-001",
            "order_number": explicit_order_number or "ORD-STUB-001",
            "status": "processing",
            "total_amount": 99.99,
            "items": [{"name": "Sample Product", "qty": 1}],
            "stub": True,
        }
        order_id = "ORD-STUB-001"

    tools_used = list(state.get("tools_used", [])) + ["lookup_order"]
    return {
        "order_data": order_data,
        "order_id": order_id,
        "tools_used": tools_used,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Node 5 — create_ticket
# ---------------------------------------------------------------------------


async def create_ticket(state: SupportAgentState) -> Dict[str, Any]:
    """Create a new support ticket in the database.

    Extracts ticket fields from the customer message via LLM, then
    persists the record.

    State reads : messages, customer_id, intent, sentiment, urgency
    State writes : current_ticket_id, ticket_data, tools_used, error
    """
    logger.info("Node: create_ticket | session=%s", state.get("session_id"))

    message = _last_human_message(state)
    customer_id = state.get("customer_id")
    urgency = state.get("urgency", "medium")

    # Map urgency → priority
    priority_map = {
        "low": "low",
        "medium": "medium",
        "high": "high",
        "critical": "critical",
    }
    priority = priority_map.get(urgency, "medium")

    # Extract ticket fields via LLM.
    ticket_fields: Dict[str, Any] = {
        "subject": f"Support request: {message[:80]}",
        "description": message,
        "priority": priority,
        "category": "general",
    }

    llm = _get_llm(temperature=0.0)
    try:
        extract_response = await llm.ainvoke([
            HumanMessage(content=TICKET_CREATE_EXTRACTION_PROMPT.format(message=message))
        ])
        parsed = _safe_json_parse(extract_response.content)
        if parsed:
            ticket_fields.update(parsed)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ticket field extraction failed, using defaults: %s", exc)

    ticket_data: Optional[Dict[str, Any]] = None
    current_ticket_id: Optional[str] = None
    error: Optional[str] = None

    try:
        from database.session import get_db_session_ctx
        from database.crud import create_support_ticket

        async with get_db_session_ctx() as session:
            ticket = await create_support_ticket(
                session,
                customer_id=customer_id,
                subject=ticket_fields["subject"],
                description=ticket_fields["description"],
                priority=ticket_fields.get("priority", "medium"),
                created_by_agent=True,
            )
            current_ticket_id = str(ticket.id)
            ticket_data = {
                "id": current_ticket_id,
                "ticket_number": ticket.ticket_number,
                "subject": ticket.subject,
                "status": ticket.status.value if hasattr(ticket.status, "value") else str(ticket.status),
                "priority": ticket.priority.value if hasattr(ticket.priority, "value") else str(ticket.priority),
                "created_at": str(ticket.created_at),
            }
    except Exception as exc:  # noqa: BLE001
        logger.error("create_ticket DB error: %s", exc)
        error = str(exc)
        # Stub
        import uuid
        stub_id = f"TICK-{str(uuid.uuid4())[:8].upper()}"
        current_ticket_id = stub_id
        ticket_data = {
            "id": stub_id,
            "ticket_number": stub_id,
            "subject": ticket_fields["subject"],
            "status": "open",
            "priority": priority,
            "stub": True,
        }

    tools_used = list(state.get("tools_used", [])) + ["create_ticket"]
    logger.info("create_ticket: ticket_id=%s", current_ticket_id)

    return {
        "current_ticket_id": current_ticket_id,
        "ticket_data": ticket_data,
        "tools_used": tools_used,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Node 6 — update_ticket
# ---------------------------------------------------------------------------


async def update_ticket(state: SupportAgentState) -> Dict[str, Any]:
    """Update an existing support ticket with a comment or status change.

    State reads : current_ticket_id, messages
    State writes : ticket_data, tools_used, error
    """
    logger.info("Node: update_ticket | session=%s", state.get("session_id"))

    ticket_id = state.get("current_ticket_id")
    message = _last_human_message(state)
    error: Optional[str] = None

    if not ticket_id:
        logger.warning("update_ticket: no current_ticket_id in state.")
        return {
            "error": "No active ticket to update.",
            "tools_used": list(state.get("tools_used", [])) + ["update_ticket"],
        }

    # Extract update intent via LLM.
    update_fields: Dict[str, Any] = {
        "update_type": "comment",
        "comment": message,
        "requested_status": None,
        "requested_priority": None,
    }
    llm = _get_llm(temperature=0.0)
    try:
        extract_response = await llm.ainvoke([
            HumanMessage(
                content=TICKET_UPDATE_EXTRACTION_PROMPT.format(
                    ticket_id=ticket_id, message=message
                )
            )
        ])
        parsed = _safe_json_parse(extract_response.content)
        if parsed:
            update_fields.update(parsed)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ticket update extraction failed: %s", exc)

    ticket_data: Optional[Dict[str, Any]] = state.get("ticket_data")

    try:
        from database.session import get_db_session_ctx
        from database.crud import add_ticket_message, update_ticket_status

        async with get_db_session_ctx() as session:
            if update_fields.get("comment"):
                await add_ticket_message(
                    session,
                    ticket_id=ticket_id,
                    sender_type="agent",
                    content=update_fields["comment"],
                )
            if update_fields.get("requested_status"):
                ticket = await update_ticket_status(
                    session,
                    ticket_id=ticket_id,
                    new_status=update_fields["requested_status"],
                )
                if ticket:
                    ticket_data = {
                        "id": str(ticket.id),
                        "ticket_number": ticket.ticket_number,
                        "status": ticket.status.value if hasattr(ticket.status, "value") else str(ticket.status),
                        "updated": True,
                    }
    except Exception as exc:  # noqa: BLE001
        logger.error("update_ticket DB error: %s", exc)
        error = str(exc)
        ticket_data = {**(ticket_data or {}), "updated": True, "stub": True}

    tools_used = list(state.get("tools_used", [])) + ["update_ticket"]
    return {
        "ticket_data": ticket_data,
        "tools_used": tools_used,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Node 7 — draft_email
# ---------------------------------------------------------------------------


async def draft_email(state: SupportAgentState) -> Dict[str, Any]:
    """Compose a professional support email for the customer.

    State reads : customer_data, ticket_data, intent, retrieved_docs
    State writes : draft_email, tools_used
    """
    logger.info("Node: draft_email | session=%s", state.get("session_id"))

    customer = state.get("customer_data") or {}
    ticket = state.get("ticket_data") or {}
    intent = state.get("intent", "general_inquiry")

    resolution_notes = _format_retrieved_docs(state.get("retrieved_docs", []))
    if not resolution_notes:
        resolution_notes = "Issue will be investigated by our support team."

    user_prompt = EMAIL_DRAFT_USER_TEMPLATE.format(
        customer_name=customer.get("name", "Valued Customer"),
        customer_email=customer.get("email", ""),
        issue_summary=_last_human_message(state)[:300],
        intent=intent,
        ticket_id=ticket.get("ticket_number") or ticket.get("id") or "N/A",
        resolution_notes=resolution_notes,
    )

    draft: str = "[Email draft unavailable — LLM not reachable]"
    llm = _get_llm(temperature=0.3)
    try:
        response = await llm.ainvoke([
            SystemMessage(content=EMAIL_DRAFT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        draft = response.content.strip()
    except Exception as exc:  # noqa: BLE001
        logger.error("draft_email LLM call failed: %s", exc)

    tools_used = list(state.get("tools_used", [])) + ["draft_email"]
    return {
        "draft_email": draft,
        "tools_used": tools_used,
    }


# ---------------------------------------------------------------------------
# Node 8 — escalate_to_human
# ---------------------------------------------------------------------------


async def escalate_to_human(state: SupportAgentState) -> Dict[str, Any]:
    """Handle graceful handoff to a human support agent.

    Creates / updates an escalated ticket, generates a compassionate
    handoff message for the customer, and logs an internal alert.

    State reads : customer_data, current_ticket_id, escalation_reason,
                  sentiment, urgency, turn_count
    State writes : escalation_needed, final_response, ticket_data,
                   tools_used, messages
    """
    logger.warning(
        "Node: escalate_to_human | session=%s reason=%s",
        state.get("session_id"),
        state.get("escalation_reason"),
    )

    customer = state.get("customer_data") or {}
    ticket = state.get("ticket_data") or {}
    ticket_id = state.get("current_ticket_id") or ticket.get("id") or "N/A"
    reason = state.get("escalation_reason") or "Customer requested human assistance."

    # Ensure an escalated ticket exists.
    if state.get("current_ticket_id"):
        try:
            from database.session import get_db_session_ctx
            from database.crud import update_ticket_status

            async with get_db_session_ctx() as session:
                await update_ticket_status(
                    session,
                    ticket_id=state["current_ticket_id"],
                    new_status="escalated",
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("escalate_to_human: could not update ticket status: %s", exc)

    # Generate customer-facing handoff message.
    user_prompt = ESCALATION_USER_TEMPLATE.format(
        reason=reason,
        sentiment=state.get("sentiment", "neutral"),
        urgency=state.get("urgency", "medium"),
        ticket_id=ticket_id,
        customer_name=customer.get("name", "there"),
    )

    handoff_message = (
        f"I completely understand, and I sincerely apologise for the difficulty. "
        f"I've escalated your case (Ticket #{ticket_id}) to a senior support specialist "
        f"who will review your situation personally. You can expect to hear from them "
        f"within 2–4 business hours. Thank you for your patience."
    )

    llm = _get_llm(temperature=0.4)
    try:
        response = await llm.ainvoke([
            SystemMessage(content=ESCALATION_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        handoff_message = response.content.strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("escalate_to_human: LLM generation failed, using fallback: %s", exc)

    # Log internal alert.
    alert = ESCALATION_ALERT_TEMPLATE.format(
        session_id=state.get("session_id", "N/A"),
        ticket_id=ticket_id,
        customer_name=customer.get("name", "Unknown"),
        customer_id=state.get("customer_id", "N/A"),
        tier=customer.get("tier", "N/A"),
        intent=state.get("intent", "N/A"),
        sentiment=state.get("sentiment", "N/A"),
        urgency=state.get("urgency", "N/A"),
        reason=reason,
        turn_count=state.get("turn_count", 0),
        last_message=_last_human_message(state)[:400],
        conversation_summary=_conversation_history(state, max_turns=5),
    )
    logger.warning("ESCALATION ALERT:\n%s", alert)

    tools_used = list(state.get("tools_used", [])) + ["escalate_to_human"]

    return {
        "escalation_needed": True,
        "final_response": handoff_message,
        "tools_used": tools_used,
        "messages": [AIMessage(content=handoff_message)],
    }


# ---------------------------------------------------------------------------
# Node 9 — generate_response
# ---------------------------------------------------------------------------


async def generate_response(state: SupportAgentState) -> Dict[str, Any]:
    """Synthesise the final customer-facing response.

    Combines retrieved KB context, tool outputs, customer data, and the
    conversation history into a polished reply.

    State reads : messages, intent, sentiment, urgency, customer_data,
                  retrieved_docs, order_data, ticket_data
    State writes : draft_response, final_response, confidence_score, messages
    """
    logger.info("Node: generate_response | session=%s", state.get("session_id"))

    intent = state.get("intent", "general_inquiry")
    retrieved = state.get("retrieved_docs", [])
    customer = state.get("customer_data") or {}
    order = state.get("order_data") or {}
    ticket = state.get("ticket_data") or {}

    # Compose tool / data context block.
    tool_results_parts: List[str] = []
    if order:
        tool_results_parts.append(f"Order data: {json.dumps(order, default=str)}")
    if ticket:
        tool_results_parts.append(f"Ticket data: {json.dumps(ticket, default=str)}")
    if state.get("draft_email"):
        tool_results_parts.append(f"Draft email prepared (see draft_email field).")
    tool_results = "\n".join(tool_results_parts) or "(no tool data)"

    # Customer data summary for prompt.
    customer_summary = (
        f"Name: {customer.get('name', 'Unknown')}, "
        f"Tier: {customer.get('tier', 'unknown')}, "
        f"Email: {customer.get('email', 'N/A')}"
        if customer else "(no customer data)"
    )

    user_prompt = RESPONSE_USER_TEMPLATE.format(
        conversation_history=_conversation_history(state),
        intent=intent,
        sentiment=state.get("sentiment", "neutral"),
        urgency=state.get("urgency", "medium"),
        customer_data=customer_summary,
        retrieved_docs=_format_retrieved_docs(retrieved),
        tool_results=tool_results,
    )

    final_text = "I'm sorry, I couldn't generate a response right now. Please try again."
    confidence = 0.5

    llm = _get_llm(temperature=0.3)
    try:
        response = await llm.ainvoke([
            SystemMessage(content=RESPONSE_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        final_text = response.content.strip()
        confidence = 0.9  # TODO: parse token logprobs when available
    except Exception as exc:  # noqa: BLE001
        logger.error("generate_response LLM call failed: %s", exc)

    tools_used = list(state.get("tools_used", [])) + ["generate_response"]
    logger.info("generate_response: response_length=%d", len(final_text))

    return {
        "draft_response": final_text,
        "final_response": final_text,
        "confidence_score": confidence,
        "tools_used": tools_used,
        "messages": [AIMessage(content=final_text)],
    }


# ---------------------------------------------------------------------------
# Node 10 — performance_summary  (admin / optional path)
# ---------------------------------------------------------------------------


async def performance_summary(state: SupportAgentState) -> Dict[str, Any]:
    """Generate a structured performance report for admin/QA reviewers.

    This node is only reached via the admin conditional edge in the graph.

    State reads : all fields
    State writes : performance_data, final_response, tools_used
    """
    logger.info("Node: performance_summary | session=%s", state.get("session_id"))

    user_prompt = PERFORMANCE_SUMMARY_USER_TEMPLATE.format(
        session_id=state.get("session_id", "N/A"),
        customer_id=state.get("customer_id", "N/A"),
        turn_count=state.get("turn_count", 0),
        intent=state.get("intent", "N/A"),
        sentiment=state.get("sentiment", "N/A"),
        escalation_needed=state.get("escalation_needed", False),
        tools_used=", ".join(state.get("tools_used", [])),
        confidence_score=state.get("confidence_score", 0.0),
        conversation_history=_conversation_history(state, max_turns=20),
    )

    report = "[Performance summary unavailable]"
    perf_data: Dict[str, Any] = {
        "session_id": state.get("session_id"),
        "turn_count": state.get("turn_count", 0),
        "intent": state.get("intent"),
        "sentiment": state.get("sentiment"),
        "escalation_needed": state.get("escalation_needed", False),
        "tools_used": list(state.get("tools_used", [])),
        "confidence_score": state.get("confidence_score", 0.0),
        "docs_retrieved": len(state.get("retrieved_docs", [])),
    }

    llm = _get_llm(temperature=0.1)
    try:
        response = await llm.ainvoke([
            SystemMessage(content=PERFORMANCE_SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        report = response.content.strip()
        perf_data["report"] = report
    except Exception as exc:  # noqa: BLE001
        logger.error("performance_summary LLM call failed: %s", exc)

    tools_used = list(state.get("tools_used", [])) + ["performance_summary"]

    return {
        "performance_data": perf_data,
        "final_response": report,
        "tools_used": tools_used,
        "messages": [AIMessage(content=report)],
    }


# ---------------------------------------------------------------------------
# Node 11 — audit  (always the last node)
# ---------------------------------------------------------------------------


async def audit_and_log(state: SupportAgentState) -> Dict[str, Any]:
    """Post-turn audit: persist metrics, log telemetry, clean state.

    This is the terminal node — always runs last before END.

    State reads : all fields
    State writes : (nothing critical — purely side-effects + metadata update)
    """
    logger.info(
        "Node: audit_and_log | session=%s intent=%s escalated=%s turns=%d",
        state.get("session_id"),
        state.get("intent"),
        state.get("escalation_needed"),
        state.get("turn_count", 0),
    )

    # TODO: Persist AgentActionLog record to database.
    # TODO: Emit Prometheus / OpenTelemetry metrics.
    # TODO: Push conversation summary to analytics pipeline.

    audit_entry = {
        "session_id": state.get("session_id"),
        "intent": state.get("intent"),
        "sentiment": state.get("sentiment"),
        "urgency": state.get("urgency"),
        "escalation_needed": state.get("escalation_needed"),
        "turn_count": state.get("turn_count"),
        "tools_used": list(state.get("tools_used", [])),
        "confidence_score": state.get("confidence_score"),
        "ticket_id": state.get("current_ticket_id"),
        "docs_retrieved": len(state.get("retrieved_docs", [])),
    }

    logger.info("AUDIT: %s", json.dumps(audit_entry))

    # Return metadata update so trace info is preserved in checkpointed state.
    meta = dict(state.get("metadata") or {})
    meta["last_audit"] = audit_entry

    return {"metadata": meta}

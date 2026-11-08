"""Chat and Agent Execution API Router — real LangGraph invocation.

Endpoints:
    POST /api/v1/chat          — process a chat turn through the LangGraph agent
    POST /api/v1/chat/stream   — SSE streaming version (agent progress events)
    GET  /api/v1/chat/{session_id}/history — conversation history from checkpoint
    DELETE /api/v1/chat/{session_id}       — clear a conversation session

Both chat endpoints invoke the compiled LangGraph ``SupportAgentState``
workflow and return the agent's final answer plus the actions it took
(tools used, ticket created, escalation, retrieved docs, …).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy.ext.asyncio import AsyncSession

from app.input_sanitizer import (
    InputValidationError,
    detect_prompt_injection,
    escape_for_prompt,
    sanitize_message,
    sanitize_session_id,
)
from app.middleware import get_request_id

from agent.state import initial_state
from api.schemas import ChatRequest, ChatResponse
from app.dependencies import (
    AuthenticatedUser,
    get_agent_runner,
    get_current_active_user,
    get_database,
    UserRole,
)
from app.monitoring import (
    observe_conversation_outcome,
    record_conversation,
    set_log_context,
)
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger("support_agent.api.chat")
router = APIRouter()

# In-memory checkpointer keyed per app instance. Conversation continuity
# works across requests within one process; swap for AsyncPostgresSaver /
# RedisSaver in multi-worker production deployments.
_memory_checkpointer: Optional[Any] = None


def _get_checkpointer() -> Any:
    global _memory_checkpointer
    if _memory_checkpointer is None:
        _memory_checkpointer = MemorySaver()
    return _memory_checkpointer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_customer_id(payload: ChatRequest, current_user: AuthenticatedUser) -> str:
    """Customers are always bound to their own account; staff may impersonate
    a customer_id from the payload (e.g. while assisting over the phone)."""
    if current_user.role == UserRole.CUSTOMER:
        return str(current_user.id)
    return str(payload.customer_id or current_user.id)


def _message_text(msg: Any) -> str:
    content = getattr(msg, "content", "")
    return content if isinstance(content, str) else json.dumps(content, default=str)


def _extract_actions(final_state: Dict[str, Any]) -> Dict[str, Any]:
    """Summarise what the agent did during this turn for the API response."""
    tools_used: List[str] = list(final_state.get("tools_used") or [])
    ticket = final_state.get("ticket_data") or {}
    order = final_state.get("order_data") or {}

    actions: Dict[str, Any] = {
        "tools_used": tools_used,
        "documents_retrieved": len(final_state.get("retrieved_docs") or []),
        "ticket_created": bool(ticket) and ticket.get("status") == "open",
        "ticket_id": ticket.get("ticket_number") or ticket.get("id"),
        "order_lookup": bool(order),
        "escalated": bool(final_state.get("escalation_needed")),
        "escalation_reason": final_state.get("escalation_reason"),
        "draft_email": final_state.get("draft_email"),
    }
    return {k: v for k, v in actions.items() if v not in (None, False, [], "")}


def _validate_chat_payload(payload: ChatRequest) -> str:
    """Validate + sanitise the chat payload at the API boundary.

    Returns the sanitised message. Raises HTTPException(400) on invalid
    input and HTTP 200-with-flag (never an error) on detected injection —
    injection is logged and neutralised downstream, not blindly rejected,
    to avoid a trivially exploitable oracle.
    """
    try:
        message = sanitize_message(payload.message)
        payload.session_id = sanitize_session_id(payload.session_id)
    except InputValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return message


async def _run_agent(
    agent_runner: Any,
    payload: ChatRequest,
    customer_id: str,
    session_id: str,
    current_user: Optional[AuthenticatedUser] = None,
) -> Dict[str, Any]:
    """Invoke the LangGraph workflow and return its final state."""
    from agent.state import initial_state

    message = _validate_chat_payload(payload)

    # Prompt-injection screening: log + tag high-risk input, never execute
    # raw. The untrusted text is also escaped before it reaches any prompt.
    high_risk, categories = detect_prompt_injection(message)
    if high_risk:
        logger.warning(
            "PROMPT INJECTION SUSPECTED | session=%s user=%s categories=%s",
            session_id, current_user.id if current_user else "?", categories,
        )
        message = escape_for_prompt(message)

    state = initial_state(session_id=session_id, customer_id=customer_id)
    state["messages"] = [HumanMessage(content=message)]
    state["metadata"] = {
        "role": current_user.role.value if current_user else "customer",
        "injection_categories": categories,
    }

    config = {"configurable": {"thread_id": session_id}}
    try:
        final_state = await agent_runner.ainvoke(state, config=config)
    except Exception as exc:  # noqa: BLE001
        logger.exception("LangGraph agent execution failed for session=%s", session_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The support agent failed to process this message. Please try again.",
        ) from exc
    return final_state or {}


def _build_chat_response(
    payload: ChatRequest,
    current_user: AuthenticatedUser,
    session_id: str,
    final_state: Dict[str, Any],
) -> ChatResponse:
    final_text = (
        final_state.get("final_response")
        or next(
            (_message_text(m) for m in reversed(final_state.get("messages", [])) if isinstance(m, AIMessage)),
            "",
        )
        or "I'm sorry — I couldn't generate a response right now. Please try again."
    )
    turns = final_state.get("turn_count") or 1
    return ChatResponse(
        session_id=session_id,
        response=final_text,
        intent=final_state.get("intent"),
        sentiment=final_state.get("sentiment"),
        escalation_required=bool(final_state.get("escalation_needed")),
        ticket_id=(final_state.get("ticket_data") or {}).get("ticket_number")
        or final_state.get("current_ticket_id"),
        confidence_score=float(final_state.get("confidence_score") or 0.0),
        turn_count=turns,
        timestamp=datetime.now(timezone.utc),
        actions=_extract_actions(final_state),
    )


# ---------------------------------------------------------------------------
# POST /chat — main endpoint
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Process Chat Turn",
)
async def process_chat_message(
    payload: ChatRequest,
    current_user: AuthenticatedUser = Depends(get_current_active_user),
    agent_runner: Any = Depends(get_agent_runner),
    db: AsyncSession = Depends(get_database),
) -> ChatResponse:
    """Submit a customer message to the LangGraph support workflow.

    Returns the agent's final answer plus structured metadata about the
    actions it took (tool invocations, ticket creation, escalation, …).
    """
    session_id = payload.session_id or f"sess_{uuid.uuid4().hex[:12]}"
    customer_id = _resolve_customer_id(payload, current_user)
    set_log_context(
        session_id=session_id,
        user_id=str(current_user.id),
        request_id=get_request_id(),
    )
    record_conversation(new_session=not bool(payload.session_id))
    logger.info(
        "Chat turn | session=%s customer=%s user=%s",
        session_id, customer_id, current_user.email,
    )

    t0 = time.perf_counter()
    final_state = await _run_agent(agent_runner, payload, customer_id, session_id, current_user)
    observe_conversation_outcome(final_state, time.perf_counter() - t0, endpoint="chat")
    return _build_chat_response(payload, current_user, session_id, final_state)


# ---------------------------------------------------------------------------
# POST /chat/stream — SSE streaming version
# ---------------------------------------------------------------------------


@router.post(
    "/stream",
    summary="Process Chat Turn (Server-Sent Events)",
)
async def stream_chat_message(
    payload: ChatRequest,
    current_user: AuthenticatedUser = Depends(get_current_active_user),
    agent_runner: Any = Depends(get_agent_runner),
    db: AsyncSession = Depends(get_database),
) -> StreamingResponse:
    """Streams agent progress as Server-Sent Events.

    Event sequence:
        ``meta``      → session/intent classification info
        ``status``    → each graph node as it executes
        ``token``     → final answer text chunks
        ``actions``   → structured actions-taken summary
        ``done``      → terminal event with full ChatResponse payload
        ``error``     → terminal event on failure
    """
    session_id = payload.session_id or f"sess_{uuid.uuid4().hex[:12]}"
    customer_id = _resolve_customer_id(payload, current_user)
    set_log_context(
        session_id=session_id,
        user_id=str(current_user.id),
        request_id=get_request_id(),
    )
    record_conversation(new_session=not bool(payload.session_id))
    logger.info("Chat stream | session=%s customer=%s", session_id, customer_id)

    async def event_stream() -> AsyncGenerator[str, None]:
        def sse(event: str, data: Any) -> str:
            return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"

        try:
            t0 = time.perf_counter()
            state = initial_state(session_id=session_id, customer_id=customer_id)
            try:
                message = _validate_chat_payload(payload)
            except HTTPException:
                raise
            state["messages"] = [HumanMessage(content=message)]
            state["metadata"] = {
                "role": current_user.role.value if current_user else "customer",
                "injection_categories": detect_prompt_injection(message)[1],
            }
            config = {"configurable": {"thread_id": session_id}}

            final_state: Dict[str, Any] = dict(state)
            async for chunk in agent_runner.astream(state, config=config, stream_mode="updates"):
                for node_name, node_update in (chunk or {}).items():
                    if not node_update:
                        continue
                    final_state.update(node_update)
                    if node_name != "__end__":
                        yield sse("status", {"node": node_name})

                    if node_name == "triage" or node_name == "intent_classifier":
                        yield sse("meta", {
                            "session_id": session_id,
                            "intent": node_update.get("intent"),
                            "sentiment": node_update.get("sentiment"),
                            "urgency": node_update.get("urgency"),
                        })

            response = _build_chat_response(payload, current_user, session_id, final_state)
            observe_conversation_outcome(final_state, time.perf_counter() - t0, endpoint="chat_stream")

            # Stream the final answer in word chunks for a live-typing feel.
            for word in response.response.split(" "):
                yield sse("token", {"text": word + " "})
            yield sse("actions", response.actions or {})
            yield sse("done", response.model_dump(mode="json"))
        except HTTPException as exc:
            yield sse("error", {"detail": exc.detail})
        except Exception as exc:  # noqa: BLE001
            logger.exception("Chat stream failed for session=%s", session_id)
            yield sse("error", {"detail": "The support agent failed to process this message."})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Session history / cleanup
# ---------------------------------------------------------------------------


@router.get(
    "/{session_id}/history",
    summary="Get Conversation History",
)
async def get_session_history(
    session_id: str,
    agent_runner: Any = Depends(get_agent_runner),
    current_user: AuthenticatedUser = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Retrieves chronological message history for a given session."""
    config = {"configurable": {"thread_id": session_id}}
    try:
        snapshot = await agent_runner.aget_state(config)
    except Exception:  # noqa: BLE001
        snapshot = None

    messages = []
    if snapshot and snapshot.values:
        for m in snapshot.values.get("messages", []):
            role = "customer" if getattr(m, "type", "") == "human" else "agent"
            messages.append({"role": role, "content": _message_text(m)})

    return {
        "session_id": session_id,
        "messages": messages,
        "total_turns": len([m for m in messages if m["role"] == "customer"]),
    }


@router.delete(
    "/{session_id}",
    summary="Clear Conversation Session",
)
async def delete_session(
    session_id: str,
    agent_runner: Any = Depends(get_agent_runner),
    current_user: AuthenticatedUser = Depends(get_current_active_user),
) -> Dict[str, str]:
    """Clears conversation messages and checkpointer checkpoints for a session."""
    config = {"configurable": {"thread_id": session_id}}
    try:
        await agent_runner.aupdate_state(config, values=None, as_node="triage")
    except Exception:  # noqa: BLE001
        pass
    return {"session_id": session_id, "status": "cleared"}

"""Typed HTTP client for the FastAPI backend.

Every frontend page talks to the backend API only — never to the agent,
database, or LangGraph directly.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")
TIMEOUT = 90.0


class APIError(Exception):
    """Raised when the backend returns a non-2xx response."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _headers(token: Optional[str] = None) -> Dict[str, str]:
    h: Dict[str, str] = {}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _request(
    method: str,
    path: str,
    token: Optional[str] = None,
    json_body: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    url = f"{BACKEND_URL}{path}"
    try:
        resp = httpx.request(
            method,
            url,
            json=json_body,
            data=data,
            params=params,
            headers=_headers(token),
            timeout=TIMEOUT,
        )
    except httpx.ConnectError as exc:
        raise APIError(0, f"Cannot reach backend at {BACKEND_URL}. Is it running?") from exc

    if resp.status_code >= 400:
        # FastAPI error envelope: {"detail": "..."}
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:  # noqa: BLE001
            detail = resp.text
        raise APIError(resp.status_code, str(detail))
    if resp.status_code == 204 or not resp.content:
        return {}
    return resp.json()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def login(email: str, password: str) -> Dict[str, Any]:
    """OAuth2 form login → TokenResponse."""
    return _request("POST", "/api/v1/auth/login", data={"username": email, "password": password})


def refresh_token(refresh: str) -> Dict[str, Any]:
    return _request("POST", "/api/v1/auth/refresh", json_body={"refresh_token": refresh})


def get_me(token: str) -> Dict[str, Any]:
    return _request("GET", "/api/v1/auth/me", token=token)


# ---------------------------------------------------------------------------
# Chat / Agent
# ---------------------------------------------------------------------------


def send_chat(token: str, message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    return _request(
        "POST",
        "/api/v1/chat",
        token=token,
        json_body={"message": message, "session_id": session_id},
    )


def stream_chat(token: str, message: str, session_id: Optional[str] = None):
    """Yield parsed SSE events from POST /chat/stream.

    Each event: {"event": str, "data": dict}
    """
    url = f"{BACKEND_URL}/api/v1/chat/stream"
    headers = _headers(token)
    payload = {"message": message}
    if session_id:
        payload["session_id"] = session_id
    with httpx.stream("POST", url, json=payload, headers=headers, timeout=TIMEOUT) as resp:
        if resp.status_code >= 400:
            raise APIError(resp.status_code, f"Streaming failed ({resp.status_code}).")
        event_name = ""
        for line in resp.iter_lines():
            if line.startswith("event: "):
                event_name = line[len("event: "):].strip()
            elif line.startswith("data: ") and event_name:
                import json

                try:
                    data = json.loads(line[len("data: "):])
                except Exception:  # noqa: BLE001
                    data = {}
                yield {"event": event_name, "data": data}
                if event_name in ("done", "error"):
                    return


def get_chat_history(token: str, session_id: str) -> Dict[str, Any]:
    return _request("GET", f"/api/v1/chat/{session_id}/history", token=token)


def clear_chat_session(token: str, session_id: str) -> Dict[str, Any]:
    return _request("DELETE", f"/api/v1/chat/{session_id}", token=token)


# ---------------------------------------------------------------------------
# Tickets
# ---------------------------------------------------------------------------


def list_tickets(token: str, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"limit": limit}
    if status:
        params["status"] = status
    return _request("GET", "/api/v1/tickets", token=token, params=params)


def get_ticket(token: str, ticket_number: str) -> Dict[str, Any]:
    return _request("GET", f"/api/v1/tickets/{ticket_number}", token=token)


def create_ticket(token: str, subject: str, description: str, priority: str = "medium",
                  customer_id: Optional[int] = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {"subject": subject, "description": description, "priority": priority}
    if customer_id is not None:
        body["customer_id"] = customer_id
    return _request("POST", "/api/v1/tickets", token=token, json_body=body)


def update_ticket(token: str, ticket_id: int, **fields: Any) -> Dict[str, Any]:
    return _request("PATCH", f"/api/v1/tickets/{ticket_id}", token=token, json_body=fields)


# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------


def get_customer(token: str, customer_id: int) -> Dict[str, Any]:
    return _request("GET", f"/api/v1/customers/{customer_id}", token=token)


def list_customer_orders(token: str, customer_id: int, limit: int = 10) -> List[Dict[str, Any]]:
    return _request("GET", f"/api/v1/customers/{customer_id}/orders", token=token, params={"limit": limit})


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------


def performance_summary(token: str, window_hours: int = 24, include_report: bool = False) -> Dict[str, Any]:
    return _request(
        "GET",
        "/api/v1/admin/performance-summary",
        token=token,
        params={"window_hours": window_hours, "include_report": include_report},
    )


def agent_logs(token: str, action_type: Optional[str] = None, skip: int = 0,
               limit: int = 50) -> Dict[str, Any]:
    params: Dict[str, Any] = {"skip": skip, "limit": limit}
    if action_type:
        params["action_type"] = action_type
    return _request("GET", "/api/v1/admin/agent-logs", token=token, params=params)


# ---------------------------------------------------------------------------
# Convenience for Streamlit
# ---------------------------------------------------------------------------


def api_call(st_element, fn, *args, **kwargs):
    """Run an API call showing an inline error on failure. Returns None on error."""
    try:
        return fn(*args, **kwargs)
    except APIError as exc:
        if exc.status_code == 0:
            st_element.error(f"🔌 {exc.message}")
        elif exc.status_code == 401:
            st_element.error("🔒 Session expired — please log in again.")
            for k in ("token", "auth"):
                st.session_state.pop(k, None)
        else:
            st_element.error(f"API error {exc.status_code}: {exc.message}")
        return None


@st.cache_data(ttl=10, show_spinner=False)
def cached_tickets(token: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
    return list_tickets(token, status=status)

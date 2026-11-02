"""Shared session-state helpers: authentication, token cache, demo login."""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from frontend.api_client import api_call, get_customer, get_me, list_customer_orders, login

DEMO_ACCOUNTS = [
    {"label": "👤 Jane Doe — Customer", "email": "jane@example.com", "password": "pw123456"},
    {"label": "🧑‍💻 Sam Rivera — Support Agent", "email": "staff@example.com", "password": "staffpw123"},
    {"label": "🛡️ Ada Quinn — Admin", "email": "admin@example.com", "password": "adminpw123"},
]


def is_logged_in() -> bool:
    return bool(st.session_state.get("token"))


def current_auth() -> Dict[str, Any]:
    """Return {'token', 'me', 'role'} — empty dicts when logged out."""
    return {
        "token": st.session_state.get("token"),
        "me": st.session_state.get("me") or {},
        "role": st.session_state.get("role", ""),
    }


def do_login(email: str, password: str) -> bool:
    """Authenticate against the backend; populate session state. True on success."""
    placeholder = st.empty()
    tokens = api_call(placeholder, login, email, password)
    placeholder.empty()
    if not tokens:
        return False

    st.session_state["token"] = tokens["access_token"]
    st.session_state["refresh_token"] = tokens.get("refresh_token")
    st.session_state["role"] = tokens.get("role", "customer")

    me = api_call(placeholder, get_me, tokens["access_token"]) or {}
    placeholder.empty()
    st.session_state["me"] = me

    # Customers: cache own profile + orders for the sidebar.
    if tokens.get("role") == "customer":
        profile = api_call(placeholder, get_customer, tokens["access_token"], me.get("user_id"))
        placeholder.empty()
        st.session_state["customer_profile"] = profile or {}
        st.session_state["orders"] = api_call(
            placeholder, list_customer_orders, tokens["access_token"], me.get("user_id")
        )
        placeholder.empty()

    # Fresh chat session per login
    st.session_state.pop("chat_session_id", None)
    st.session_state.pop("messages", None)
    return True


def logout() -> None:
    for key in ("token", "refresh_token", "me", "role", "customer_profile", "orders"):
        st.session_state.pop(key, None)


def render_login_form(compact: bool = False) -> None:
    """Demo login: one-click role buttons + manual email/password."""
    if not compact:
        st.markdown("### 🔐 Sign in")
        st.caption("Demo mode — pick an account or enter credentials manually.")

    col_a, col_b = st.columns([1, 1])
    picked = None
    for i, acct in enumerate(DEMO_ACCOUNTS):
        col = [col_a, col_b][i % 2]
        with col:
            if st.button(acct["label"], use_container_width=True, key=f"demo_{i}"):
                picked = acct

    st.divider()
    with st.form("login_form"):
        email = st.text_input("Email", value="jane@example.com")
        password = st.text_input("Password", value="pw123456", type="password")
        submitted = st.form_submit_button("Sign in", use_container_width=True, type="primary")

    if picked:
        if do_login(picked["email"], picked["password"]):
            st.rerun()
    elif submitted:
        if do_login(email.strip(), password):
            st.rerun()
        else:
            st.error("Invalid email or password.")

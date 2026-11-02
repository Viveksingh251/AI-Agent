"""Page 1 — Customer Chat: talk to the LangGraph agent via the FastAPI backend."""

from __future__ import annotations

import uuid
from typing import Any, Dict

import streamlit as st

from frontend.api_client import APIError, api_call, send_chat, stream_chat
from frontend.auth import current_auth, do_login, is_logged_in, render_login_form
from frontend.components import badges, escalation_banner, tool_steps_expander


def render_sidebar() -> None:
    with st.sidebar:
        profile = st.session_state.get("customer_profile") or {}
        me = st.session_state.get("me") or {}

        st.markdown("### 👤 Your profile")
        st.markdown(f"**{profile.get('name', me.get('name', 'Guest'))}**")
        st.caption(f"`{profile.get('email', me.get('email', ''))}`")
        st.markdown(badges(profile.get("tier"), me.get("role")), unsafe_allow_html=True)

        orders = st.session_state.get("orders") or []
        st.divider()
        st.markdown("### 📦 Recent orders")
        if orders:
            for o in orders[:4]:
                st.markdown(
                    f"`{o.get('order_number')}` — {badge(o.get('status'))} "
                    f"&nbsp; ${o.get('total_amount')}",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No orders yet.")

        st.divider()
        if st.button("🧹 New conversation", use_container_width=True):
            st.session_state.pop("chat_session_id", None)
            st.session_state.pop("messages", None)
            st.rerun()
        if st.button("🚪 Log out", use_container_width=True):
            from frontend.auth import logout

            logout()
            st.rerun()


def render_login_gate() -> None:
    st.title("💬 Customer Support Chat")
    st.caption("Sign in to start chatting with your AI support assistant.")
    render_login_form()


def _append_message(role: str, content: str, **extra: Any) -> None:
    msg: Dict[str, Any] = {"role": role, "content": content}
    msg.update(extra)
    st.session_state.setdefault("messages", []).append(msg)


def render_message(msg: Dict[str, Any]) -> None:
    with st.chat_message("assistant" if msg["role"] == "assistant" else "user",
                         avatar="🤖" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])

        # Escalation notice — prominent banner inside the assistant bubble
        if msg.get("escalated"):
            escalation_banner(
                msg.get("escalation_reason", "Requested human review"),
                msg.get("ticket_id"),
            )

        # Created ticket chip
        if msg.get("ticket_created") and msg.get("ticket_id"):
            st.success(f"🎫 Support ticket `{msg['ticket_id']}` was created for you.")

        # Collapsible intermediate steps
        if msg["role"] == "assistant" and (msg.get("actions") or msg.get("steps")):
            steps = msg.get("steps") or []
            actions = msg.get("actions") or {}
            if steps:
                _render_live_steps(steps, ticket_created=actions.get("ticket_created"),
                                   ticket_id=actions.get("ticket_id"))
            else:
                tool_steps_expander([], actions)


def _render_live_steps(steps, ticket_created=False, ticket_id=None) -> None:
    with st.expander("🔧 Agent reasoning — expand to see tool calls", expanded=False):
        for s in steps:
            st.markdown(f"- `{s}`")
        if ticket_created and ticket_id:
            st.success(f"🎫 Ticket `{ticket_id}` created")


def main() -> None:
    if not is_logged_in():
        render_login_gate()
        return

    role = current_auth()["role"]
    if role != "customer":
        st.info("You are signed in as staff — switch to the **Agent Dashboard** or **Admin** page, "
                "or log in as a customer to test the chat experience.")
        if st.button("Log in as a customer instead"):
            from frontend.auth import logout

            logout()
            st.rerun()
        return

    render_sidebar()
    st.title("💬 Customer Support Chat")
    st.caption("Ask about orders, returns, billing — the agent can search the knowledge base, "
               "look up your data, open tickets, and escalate when needed.")
    st.divider()

    # ---- chat history (st.session_state) ----
    if "messages" not in st.session_state:
        _append_message(
            "assistant",
            "Hi! 👋 I'm your AI support assistant. How can I help you today?\n\n"
            "You can ask things like *\"Where is my order ORD-1001?\"* or "
            "*\"What's your return policy?\"*",
        )

    for msg in st.session_state["messages"]:
        render_message(msg)

    # ---- input ----
    if prompt := st.chat_input("Type your message…"):
        _append_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            status_box = st.empty()
            steps: list = []
            actions: Dict[str, Any] = {}
            answer = ""
            error = None

            try:
                # Stream the agent's progress via SSE (falls back silently on error)
                for event in stream_chat(
                    current_auth()["token"],
                    prompt,
                    st.session_state.get("chat_session_id"),
                ):
                    ev, data = event["event"], event["data"]
                    if ev == "meta":
                        st.session_state["chat_session_id"] = data.get("session_id")
                        sentiment = data.get("sentiment")
                        if sentiment:
                            status_box.markdown(
                                f"{badge('neutral').replace('neutral', '')}"
                                f"*Classifying intent:* `{data.get('intent')}` · sentiment: {sentiment}",
                                unsafe_allow_html=True,
                            )
                    elif ev == "status":
                        node = data.get("node", "")
                        if node not in ("__end__",):
                            steps.append(node)
                            status_box.markdown(f"⚙️ *Working…* (`{node}`)")
                    elif ev == "token":
                        answer += data.get("text", "")
                        placeholder.markdown(answer + "▌")
                    elif ev == "actions":
                        actions = data or {}
                    elif ev == "done":
                        full = data or {}
                        answer = full.get("response") or answer
                        actions = full.get("actions") or actions
                    elif ev == "error":
                        error = data.get("detail", "Agent error.")
            except APIError as exc:
                error = exc.message
            except Exception as exc:  # noqa: BLE001
                error = f"Connection problem: {exc}"

            status_box.empty()
            if error and not answer:
                placeholder.error(error)
                _append_message("assistant", f"⚠️ {error}")
            else:
                placeholder.markdown(answer or "(no response)")
                if actions.get("escalated"):
                    escalation_banner(
                        actions.get("escalation_reason") or "Requested human review",
                        actions.get("ticket_id"),
                    )
                if actions.get("ticket_created") and actions.get("ticket_id"):
                    st.success(f"🎫 Support ticket `{actions['ticket_id']}` was created for you.")
                if steps or actions:
                    tool_steps_expander(steps, actions)

                _append_message(
                    "assistant",
                    answer or "(no response)",
                    escalated=actions.get("escalated"),
                    escalation_reason=actions.get("escalation_reason"),
                    ticket_created=actions.get("ticket_created"),
                    ticket_id=actions.get("ticket_id"),
                    actions=actions,
                    steps=steps,
                )


main()

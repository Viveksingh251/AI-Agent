"""Page 2 — Support Agent Dashboard: open tickets, take over escalations,
review & send drafted emails. Staff roles only."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from frontend.api_client import (
    APIError,
    api_call,
    get_customer,
    get_ticket,
    list_tickets,
    send_chat,
    update_ticket,
)
from frontend.auth import current_auth, is_logged_in, logout, render_login_form
from frontend.components import badge, badges, metric_row, page_header

OPEN_STATUSES = ["open", "in_progress", "escalated"]


def _staff_gate() -> bool:
    if not is_logged_in():
        render_login_gate_staff()
        return False
    if current_auth()["role"] == "customer":
        st.warning("This dashboard is for support staff. You are logged in as a customer.")
        if st.button("Switch account"):
            logout()
            st.rerun()
        return False
    return True


def render_login_gate_staff() -> None:
    page_header("🧑‍💻 Agent Dashboard", "Sign in with a support-agent or admin account.")
    render_login_form()


def _take_over(ticket: Dict[str, Any]) -> None:
    """'Take over' = assign via status bump to in_progress + agent chat nudge."""
    token = current_auth()["token"]
    ticket_id = ticket.get("id")
    api_call(
        st.empty(),
        update_ticket,
        token,
        ticket_id,
        status="in_progress",
    )
    st.toast(f"✅ Took over ticket {ticket.get('ticket_number')}", icon="✅")


def _close_ticket(ticket: Dict[str, Any]) -> None:
    token = current_auth()["token"]
    api_call(st.empty(), update_ticket, token, ticket.get("id"), status="resolved")


def _draft_email_for(ticket: Dict[str, Any], tone: str) -> Optional[str]:
    """Ask the agent (through the backend chat endpoint) for a follow-up draft."""
    token = current_auth()["token"]
    resp = api_call(
        st.empty(),
        send_chat,
        token,
        f"Draft a follow-up email for ticket {ticket.get('ticket_number')} "
        f"about '{ticket.get('subject')}' with a {tone} tone.",
    )
    if not resp:
        return None
    # The agent surfaces drafted emails in its actions block
    return (resp.get("actions") or {}).get("draft_email") or resp.get("response")


def _ticket_detail_panel(ticket: Dict[str, Any]) -> None:
    """Expanded detail view with actions for one ticket."""
    token = current_auth()["token"]
    detail = api_call(st.empty(), get_ticket, token, ticket.get("ticket_number") or str(ticket.get("id")))
    if detail is None:
        return

    st.markdown(f"#### {detail.get('subject')}")
    st.markdown(badges(detail.get("status"), detail.get("priority")), unsafe_allow_html=True)
    st.write(detail.get("description"))
    if detail.get("customer_id"):
        cust = api_call(st.empty(), get_customer, token, detail["customer_id"])
        if cust:
            st.caption(f"Customer: **{cust.get('name')}** ({cust.get('email')}) · tier {cust.get('tier')}")

    st.divider()
    c1, c2, c3 = st.columns(3)
    if detail.get("status") == "escalated":
        if c1.button("🙋 Take over ticket", key=f"take_{detail['id']}", type="primary"):
            _take_over(detail)
            st.rerun()
    if detail.get("status") in ("open", "in_progress", "escalated"):
        if c2.button("✅ Mark resolved", key=f"res_{detail['id']}"):
            _close_ticket(detail)
            st.rerun()

    # ---- drafted email approve / send flow ----
    st.markdown("##### ✉️ Follow-up email")
    tone = st.selectbox("Tone", ["professional", "friendly", "apologetic"],
                        key=f"tone_{detail['id']}", horizontal=True)
    draft_key = f"draft_{detail['id']}"
    if st.button("🪄 Generate draft", key=f"gen_{detail['id']}"):
        with st.spinner("Agent is drafting…"):
            draft = _draft_email_for(detail, tone)
        if draft:
            st.session_state[draft_key] = draft
            st.rerun()

    if draft := st.session_state.get(draft_key):
        edited = st.text_area("Email draft", value=draft, height=260, key=f"edit_{detail['id']}")
        b1, b2 = st.columns(2)
        if b1.button("✔️ Approve & send", key=f"send_{detail['id']}", type="primary"):
            # Sending = record the approved note on the ticket via staff PATCH +
            # inform the agent chat that the email was approved.
            api_call(
                st.empty(),
                send_chat,
                token,
                f"The follow-up email for ticket {detail.get('ticket_number')} was approved "
                f"and sent to the customer.",
            )
            api_call(st.empty(), update_ticket, token, detail["id"], status="resolved")
            st.session_state.pop(draft_key, None)
            st.toast("📨 Email sent & ticket resolved", icon="📨")
            st.rerun()
        if b2.button("🗑️ Discard draft", key=f"dis_{detail['id']}"):
            st.session_state.pop(draft_key, None)
            st.rerun()


def main() -> None:
    if not _staff_gate():
        return

    page_header(
        "🧑‍💻 Agent Dashboard",
        "Triage open tickets, take over escalations, and review agent-drafted emails.",
    )

    token = current_auth()["token"]

    # ---- load tickets ----
    all_tickets: List[Dict[str, Any]] = []
    for status_filter in OPEN_STATUSES:
        batch = api_call(st.empty(), list_tickets, token, status=status_filter)
        if batch:
            all_tickets.extend(batch)

    escalated = [t for t in all_tickets if t.get("status") == "escalated"]
    in_progress = [t for t in all_tickets if t.get("status") == "in_progress"]
    open_t = [t for t in all_tickets if t.get("status") == "open"]

    metric_row({
        "🔥 Escalated": len(escalated),
        "⚙️ In progress": len(in_progress),
        "📬 Open": len(open_t),
    })
    st.divider()

    # ---- escalated first: takeover queue ----
    st.markdown("### 🚨 Escalated — needs takeover")
    if not escalated:
        st.success("No escalated tickets. Nice and quiet. 🎉")
    for t in escalated:
        with st.container(border=True):
            c1, c2, c3 = st.columns([5, 2])
            with c1:
                st.markdown(f"**{t.get('ticket_number')}** — {t.get('subject')}")
            with c2:
                st.markdown(badges(t.get("status"), t.get("priority")), unsafe_allow_html=True)
            with c3:
                if st.button("🙋 Take over", key=f"q_take_{t['id']}", type="primary",
                             use_container_width=True):
                    _take_over(t)
                    st.rerun()

    # ---- open / in-progress ----
    st.markdown("### 📋 Queue")
    for status_group, icon in ((open_t, "📬"), (in_progress, "⚙️")):
        for t in status_group:
            with st.container(border=True):
                head = st.columns([5, 2, 2, 1])
                with head[0]:
                    st.markdown(f"{icon} **{t.get('ticket_number')}** — {t.get('subject')}")
                with head[1]:
                    st.markdown(badge(t.get("status")), unsafe_allow_html=True)
                with head[2]:
                    st.markdown(badge(t.get("priority")), unsafe_allow_html=True)
                with head[3]:
                    if st.button("Open", key=f"open_{t['id']}"):
                        st.session_state["selected_ticket"] = t.get("id")
                        st.session_state["selected_ticket_number"] = t.get("ticket_number")

    # ---- detail / action panel ----
    selected_id = st.session_state.get("selected_ticket")
    if selected_id:
        st.divider()
        selected = next((t for t in all_tickets if t.get("id") == selected_id), None)
        if selected:
            with st.container(border=True):
                _ticket_detail_panel(selected)
            if st.button("← Close panel"):
                st.session_state.pop("selected_ticket", None)
                st.session_state.pop("selected_ticket_number", None)
                st.rerun()


main()

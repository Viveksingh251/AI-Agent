"""Page 3 — Admin: performance summary + agent action audit logs."""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from frontend.api_client import agent_logs, api_call, performance_summary
from frontend.auth import current_auth, is_logged_in, logout, render_login_form
from frontend.components import badge, badges, metric_row, page_header


def _admin_gate() -> bool:
    if not is_logged_in():
        page_header("🛡️ Admin", "Sign in with an admin account.")
        render_login_form()
        return False
    role = current_auth()["role"]
    if role not in ("admin", "support_agent"):
        st.warning("This page is restricted to admins and support agents.")
        if st.button("Switch account"):
            logout()
            st.rerun()
        return False
    return True


def _render_performance(token: str) -> None:
    st.markdown("### 📊 Performance summary")

    c1, c2 = st.columns([1, 3])
    window = c1.selectbox("Window", [1, 6, 24, 72, 168], index=2, format_func=lambda h: f"Last {h}h")
    include_report = c2.checkbox("Include LLM narrative report", value=False)

    summary: Dict[str, Any] = api_call(
        st.empty(), performance_summary, token, window, include_report
    ) or {}

    metric_row({
        "🤖 Agent actions": summary.get("total_actions", 0),
        "✅ Success rate": summary.get("success_rate", 0.0),
        "🎫 Tickets created": summary.get("tickets_created", 0),
        "📬 Open tickets": summary.get("open_tickets", 0),
        "🚨 Escalations": summary.get("escalations", 0),
        "❌ Failed actions": summary.get("failed_actions", 0),
    })

    by_type = summary.get("actions_by_type") or {}
    if by_type:
        st.markdown("##### Actions by type")
        max_count = max(by_type.values()) or 1
        for action_type, count in sorted(by_type.items(), key=lambda kv: -kv[1]):
            pct = int(count / max_count * 100)
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"`{action_type}`")
                st.progress(pct / 100)
            with col_b:
                st.metric("", count, label_visibility="collapsed")

    if include_report and summary.get("narrative_report"):
        st.markdown("##### 🧠 Narrative report")
        st.info(summary["narrative_report"])
    elif include_report:
        st.caption("LLM narrative unavailable (OpenAI not configured on the backend).")


def _render_logs(token: str) -> None:
    st.markdown("### 🧾 Agent action logs")

    f1, f2, f3 = st.columns([2, 2, 1])
    action_type = f1.text_input("Filter by action type", placeholder="e.g. create_support_ticket")
    success_filter = f2.selectbox("Outcome", ["all", "success only", "failures only"])
    limit = f3.selectbox("Page size", [25, 50, 100], index=1)

    params = {"limit": limit}
    if action_type.strip():
        params["action_type"] = action_type.strip()
    if success_filter == "success only":
        params["success"] = True
    elif success_filter == "failures only":
        params["success"] = False

    page = st.session_state.get("logs_skip", 0)
    data: Dict[str, Any] = api_call(
        st.empty(), agent_logs, token, params.get("action_type"), page, params["limit"],
        **({"success": params["success"]} if "success" in params else {}),
    ) or {}

    logs = data.get("logs") or []
    if not logs:
        st.info("No log entries match these filters.")
        return

    st.caption(f"Showing {len(logs)} of {data.get('total', len(logs))} entries")

    for log in logs:
        icon = "✅" if log.get("success") else "❌"
        with st.container(border=True):
            head = st.columns([4, 2, 3, 2])
            with head[0]:
                st.markdown(f"{icon} `{log.get('action_type')}`")
            with head[1]:
                st.markdown(badge("true" if log.get("success") else "false"), unsafe_allow_html=True)
            with head[2]:
                cust = log.get("customer_id")
                st.caption(f"customer: {cust if cust is not None else '—'}")
            with head[3]:
                st.caption(str(log.get("created_at", ""))[:19].replace("T", " "))
            details = log.get("details") or {}
            if details:
                with st.expander("Details"):
                    st.json(details, expanded=False)

    # pagination
    nav1, nav2, nav3 = st.columns([1, 1, 4])
    total = data.get("total", 0)
    if page > 0 and nav1.button("← Prev"):
        st.session_state["logs_skip"] = max(0, page - limit)
        st.rerun()
    if page + limit < total and nav2.button("Next →"):
        st.session_state["logs_skip"] = page + limit
        st.rerun()


def main() -> None:
    if not _admin_gate():
        return

    token = current_auth()["token"]
    role = current_auth()["role"]
    with st.sidebar:
        st.markdown("### 🛡️ Admin console")
        st.markdown(badges(role), unsafe_allow_html=True)
        if st.button("🚪 Log out", use_container_width=True):
            logout()
            st.rerun()

    page_header("🛡️ Admin Console", "Platform observability for the AI support agent.")

    tab_summary, tab_logs = st.tabs(["📊 Performance", "🧾 Agent logs"])
    with tab_summary:
        _render_performance(token)
    with tab_logs:
        _render_logs(token)


main()

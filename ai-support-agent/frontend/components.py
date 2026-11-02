"""Streamlit UI reusable presentation components — badges, cards, headers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Status badge rendering (pure CSS-in-markdown pills)
# ---------------------------------------------------------------------------

_BADGE_STYLES: Dict[str, str] = {
    # tickets
    "open": "background:#e3f2fd;color:#1565c0;",
    "in_progress": "background:#fff8e1;color:#a67c00;",
    "resolved": "background:#e8f5e9;color:#2e7d32;",
    "escalated": "background:#ffebee;color:#c62828;",
    "closed": "background:#eceff1;color:#546e7a;",
    # priorities
    "low": "background:#eceff1;color:#546e7a;",
    "medium": "background:#fff8e1;color:#a67c00;",
    "high": "background:#ffe0b2;color:#e65100;",
    "critical": "background:#ffebee;color:#b71c1c;",
    # sentiment / misc
    "positive": "background:#e8f5e9;color:#2e7d32;",
    "neutral": "background:#eceff1;color:#546e7a;",
    "frustrated": "background:#fff8e1;color:#a67c00;",
    "angry": "background:#ffebee;color:#b71c1c;",
    "very_frustrated": "background:#ffebee;color:#b71c1c;",
    # roles
    "customer": "background:#e3f2fd;color:#1565c0;",
    "support_agent": "background:#ede7f6;color:#5e35b1;",
    "admin": "background:#e8eaf6;color:#283593;",
    "true": "background:#ffebee;color:#b71c1c;",
    "false": "background:#e8f5e9;color:#2e7d32;",
}

_BADGE_LABELS = {
    "in_progress": "in progress",
    "very_frustrated": "very frustrated",
}


def badge(value: Optional[str]) -> str:
    """Render a coloured status pill as inline HTML for st.markdown."""
    if value is None:
        return ""
    key = str(value).lower()
    style = _BADGE_STYLES.get(key, "background:#eceff1;color:#37474f;")
    label = _BADGE_LABELS.get(key, key.replace("_", " "))
    return (
        f'<span style="{style}padding:2px 10px;border-radius:999px;'
        f'font-size:0.75rem;font-weight:600;white-space:nowrap;">{label}</span>'
    )


def badges(*values: Optional[str]) -> str:
    return "&nbsp; ".join(badge(v) for v in values if v is not None)


# ---------------------------------------------------------------------------
# Layout components
# ---------------------------------------------------------------------------


def configure_page() -> None:
    st.set_page_config(
        page_title="AI Customer Support Portal",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def page_header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)
    st.divider()


def ticket_card(ticket: Dict[str, Any], extra_columns=None) -> Optional[bool]:
    """Render one ticket as a styled row. Returns action result if clicked."""
    cols = extra_columns or [4, 2, 2]
    c1, c2, c3, c4, c5 = st.columns(cols)
    with c1:
        st.markdown(
            f"**{ticket.get('ticket_number', ticket.get('id', '?'))}** — "
            f"{ticket.get('subject', 'Untitled')}",
            unsafe_allow_html=False,
        )
    with c2:
        st.markdown(badge(ticket.get("status")), unsafe_allow_html=True)
    with c3:
        st.markdown(badge(ticket.get("priority")), unsafe_allow_html=True)
    with c4:
        st.caption(str(ticket.get("created_at", ""))[:16])
    with c5:
        return True
    return None


def metric_row(metrics: Dict[str, Any]) -> None:
    """Render a row of st.metric cards from a dict."""
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        with col:
            if isinstance(value, float) and 0 <= value <= 1 and label.lower().endswith("rate"):
                st.metric(label, f"{value:.0%}")
            else:
                st.metric(label, value)


def escalation_banner(reason: str, ticket_ref: Optional[str] = None) -> None:
    """Prominent banner shown when a conversation escalated to a human."""
    st.error(
        f"🚨 **Escalated to a human specialist**\n\n"
        f"**Reason:** {reason}\n\n"
        f"**Ticket:** `{ticket_ref or 'pending assignment'}` — a senior agent has been notified.",
        icon="🚨",
    )


def tool_steps_expander(steps: List[Dict[str, Any]], actions: Dict[str, Any]) -> None:
    """Collapsible panel showing the agent's intermediate work."""
    tools = actions.get("tools_used", []) or []
    label_bits = []
    if tools:
        label_bits.append(f"{len(tools)} tool calls")
    if actions.get("documents_retrieved"):
        label_bits.append(f"{actions['documents_retrieved']} docs retrieved")
    if actions.get("order_lookup"):
        label_bits.append("order lookup")
    if not label_bits:
        label_bits.append("no intermediate actions")

    with st.expander(f"🔧 Agent reasoning — {', '.join(label_bits)}", expanded=False):
        if tools:
            st.markdown("**Tools invoked (in order):**")
            for t in tools:
                st.markdown(f"- `{t}`")
        if actions.get("ticket_created"):
            st.success(f"🎫 Created ticket `{actions.get('ticket_id')}`")
        if actions.get("order_lookup"):
            st.info("📦 Looked up an order for this customer")
        if actions.get("draft_email"):
            st.markdown("**Draft email prepared:**")
            st.text_area("Draft", actions["draft_email"], height=200, disabled=True,
                         label_visibility="collapsed")
        if not tools and not actions:
            st.caption("The agent answered directly from the knowledge base.")

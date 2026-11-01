"""Streamlit UI Reusable Presentation Components."""

from typing import Any, Dict, Optional
import streamlit as st


def render_header() -> None:
    """Renders the top application branding header and status indicators."""
    st.set_page_config(
        page_title="AI Customer Support Portal",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("🤖 AI Customer Support Portal")
    st.caption("Powered by LangGraph, Hybrid RAG & Real-Time Ticketing")


def render_customer_sidebar(customer_info: Dict[str, Any]) -> None:
    """Renders customer account info, active tier, and quick actions in the sidebar.
    
    TODO:
        - Populate live tickets and order history from backend API.
    """
    with st.sidebar:
        st.subheader("👤 Customer Profile")
        st.write(f"**Name:** {customer_info.get('name', 'Jane Doe')}")
        st.write(f"**Customer ID:** `{customer_info.get('customer_id', 'CUST-10492')}`")
        st.write(f"**Tier:** :star: {customer_info.get('tier', 'Gold Member')}")
        st.divider()

        st.subheader("🎫 Active Support Tickets")
        st.info("No open urgent incidents.")
        # TODO: Add dynamic ticket cards with status badges
        st.divider()

        if st.button("🧹 Clear Conversation", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()


def render_escalation_alert(reason: str, ticket_id: Optional[str] = None) -> None:
    """Displays a prominent banner alerting user when conversation has transferred to a human."""
    st.warning(
        f"⚠️ **Escalated to Human Specialist**\n\n"
        f"**Reason:** {reason}\n\n"
        f"**Ticket Reference:** `{ticket_id or 'Pending Assignment'}`\n\n"
        f"A senior support engineer has been notified."
    )


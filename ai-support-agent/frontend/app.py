"""Streamlit Customer Support Frontend Application Entry Point."""

import os
import uuid
import httpx
import streamlit as st

from frontend.components import (
    render_customer_sidebar,
    render_escalation_alert,
    render_header,
)

BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")


def initialize_session_state() -> None:
    """Initializes persistent chat session state and identifiers."""
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = f"session_{uuid.uuid4().hex[:8]}"

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Hello! I am your AI Support Assistant. How can I help you today?",
            }
        ]

    if "customer_profile" not in st.session_state:
        st.session_state["customer_profile"] = {
            "name": "Jane Doe",
            "customer_id": "CUST-10492",
            "tier": "Gold Member",
        }


def main():
    """Main Streamlit execution loop."""
    render_header()
    initialize_session_state()

    # Render customer profile sidebar
    render_customer_sidebar(st.session_state["customer_profile"])

    # Render conversation message history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("escalated"):
                render_escalation_alert(
                    reason=msg.get("escalation_reason", "Complex request requiring supervisor review"),
                    ticket_id=msg.get("ticket_id"),
                )

    # User chat input widget
    if prompt := st.chat_input("Describe your issue or question..."):
        # Display user message immediately
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Call FastAPI backend agent endpoint
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("AI Agent is reasoning and searching knowledge base..."):
                try:
                    payload = {
                        "message": prompt,
                        "session_id": st.session_state["session_id"],
                        "customer_id": st.session_state["customer_profile"]["customer_id"],
                    }

                    # TODO: Switch to SSE streaming response when enabled
                    response = httpx.post(
                        f"{BACKEND_API_URL}/api/v1/chat",
                        json=payload,
                        timeout=30.0,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        reply_text = data.get("response", "No response content received.")
                        message_placeholder.markdown(reply_text)

                        st.session_state["messages"].append(
                            {
                                "role": "assistant",
                                "content": reply_text,
                                "escalated": data.get("escalation_required", False),
                                "ticket_id": data.get("ticket_id"),
                            }
                        )
                    else:
                        error_msg = f"API Error ({response.status_code}): {response.text}"
                        message_placeholder.error(error_msg)

                except Exception as exc:
                    message_placeholder.error(f"Failed to connect to backend server: {str(exc)}")


if __name__ == "__main__":
    main()


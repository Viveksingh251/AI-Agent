"""AI Customer Support Portal — Streamlit entry point.

Run with:
    streamlit run frontend/app.py

Multi-page navigation lives in frontend/pages/ (Streamlit auto-discovers):
    1_💬_Chat.py            — customer chat with the LangGraph agent
    2_🧑‍💻_Agent_Dashboard.py — staff triage / takeover / email approval
    3_🛡️_Admin.py           — performance summary + agent action logs
"""

from __future__ import annotations

import streamlit as st

from frontend.api_client import BACKEND_URL
from frontend.auth import current_auth, do_login, is_logged_in, render_login_form
from frontend.components import badge, badges, configure_page

configure_page()


# ---------------------------------------------------------------------------
# Home page: landing + quick login state
# ---------------------------------------------------------------------------

st.title("🤖 AI Customer Support Portal")
st.caption("Powered by LangGraph agents, RAG knowledge search, and a FastAPI backend.")

# Health dot for the backend
try:
    import httpx

    resp = httpx.get(f"{BACKEND_URL}/health", timeout=3.0)
    ok = resp.status_code == 200
except Exception:  # noqa: BLE001
    ok = False

col_dot, col_txt, _ = st.columns([0.06, 0.5, 0.44])
with col_dot:
    st.markdown(
        f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;'
        f'background:{ "#2e7d32" if ok else "#c62828"};"></span>',
        unsafe_allow_html=True,
    )
with col_txt:
    st.caption(f"Backend: {BACKEND_URL} — {'connected' if ok else 'offline'}")

st.divider()

# ---------------------------------------------------------------------------
# Navigation cards
# ---------------------------------------------------------------------------

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("#### 💬 Customer Chat")
    st.caption("Chat with the AI agent. It searches the knowledge base, looks up your "
               "orders, opens tickets, and escalates to humans when needed.")
    st.page_link("pages/1_💬_Chat.py", label="Open chat →", icon="💬")

with c2:
    st.markdown("#### 🧑‍💻 Agent Dashboard")
    st.caption("For support staff: triage open tickets, take over escalations, "
               "review & send drafted follow-up emails.")
    st.page_link("pages/2_🧑‍💻_Agent_Dashboard.py", label="Open dashboard →", icon="🧑‍💻")

with c3:
    st.markdown("#### 🛡️ Admin Console")
    st.caption("Platform observability: agent performance metrics and the full "
               "action audit trail.")
    st.page_link("pages/3_🛡️_Admin.py", label="Open console →", icon="🛡️")

st.divider()

# ---------------------------------------------------------------------------
# Quick sign-in box (shared across pages via st.session_state)
# ---------------------------------------------------------------------------

if is_logged_in():
    auth = current_auth()
    st.markdown("### 👤 Current session")
    a1, a2 = st.columns([3, 1])
    with a1:
        st.markdown(f"**{auth['me'].get('name', '?')}** — `{auth['me'].get('email', '')}`")
        st.markdown(badges(auth["me"].get("role")), unsafe_allow_html=True)
    with a2:
        from frontend.auth import logout

        if st.button("🚪 Log out", use_container_width=True):
            logout()
            st.rerun()
else:
    st.markdown("### 🔐 Quick sign-in (demo)")
    render_login_form(compact=True)

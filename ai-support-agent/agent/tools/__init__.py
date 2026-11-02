"""Customer Support Agent Tools Package.

Public API — the eight LangChain tools the graph's tool-calling LLM can bind:

- search_knowledge_base
- get_customer_info
- get_order_status
- create_support_ticket
- update_ticket_status
- draft_followup_email
- escalate_ticket
- get_ticket_history

Plus the registry (`execute_tools` node, `get_all_tools`) and the shared
authorization / audit-log helpers in ``base``.
"""

from agent.tools.base import audit_log, db_session, deny, is_authorized
from agent.tools.customer_tools import get_customer_info, get_order_status
from agent.tools.kb_tools import search_knowledge_base
from agent.tools.registry import (
    ALL_TOOLS,
    execute_tools,
    get_all_tools,
    register_all,
)
from agent.tools.ticket_tools import (
    create_support_ticket,
    draft_followup_email,
    escalate_ticket,
    get_ticket_history,
    update_ticket_status,
)

__all__ = [
    # Tools
    "search_knowledge_base",
    "get_customer_info",
    "get_order_status",
    "create_support_ticket",
    "update_ticket_status",
    "draft_followup_email",
    "escalate_ticket",
    "get_ticket_history",
    # Registry
    "ALL_TOOLS",
    "register_all",
    "get_all_tools",
    "execute_tools",
    # Shared helpers
    "audit_log",
    "db_session",
    "deny",
    "is_authorized",
]

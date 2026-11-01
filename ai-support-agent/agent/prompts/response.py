"""Response generation prompts — final answer synthesis."""

from agent.prompts.system import RESPONSE_STYLE_GUIDE, SYSTEM_PERSONA

# ---------------------------------------------------------------------------
# Main response generation prompt
# ---------------------------------------------------------------------------

RESPONSE_SYSTEM_PROMPT = f"""\
{SYSTEM_PERSONA}

{RESPONSE_STYLE_GUIDE}
"""

RESPONSE_USER_TEMPLATE = """\
## Conversation so far
{conversation_history}

## Classified intent
Intent : {intent}
Sentiment: {sentiment}
Urgency : {urgency}

## Customer profile (may be empty)
{customer_data}

## Knowledge-base context retrieved
{retrieved_docs}

## Tool / database results
{tool_results}

## Task
Generate the final customer-facing response.  Follow the formatting rules in
the system prompt.  If knowledge-base context or tool results are present, use
them — do not fabricate information.  If you cannot answer confidently, say so
and offer to escalate.
"""

# ---------------------------------------------------------------------------
# Ticket & data lookup prompt (used when building the response after tools)
# ---------------------------------------------------------------------------

ORDER_RESPONSE_TEMPLATE = """\
The customer asked about their order.  Here is the data retrieved:

Order details:
{order_data}

Customer tier: {tier}

Write a helpful 2–3 sentence response confirming the order status, providing
the tracking number if available, and giving an ETA if present.  If the order
is delayed, acknowledge any inconvenience.
"""

TICKET_RESPONSE_TEMPLATE = """\
A support ticket has just been {action} for this customer.

Ticket details:
{ticket_data}

Write a warm, brief confirmation message (2–3 sentences) that:
1. Confirms the ticket {action} (include the ticket ID).
2. Sets expectation on response time.
3. Invites the customer to reply to the ticket for updates.
"""

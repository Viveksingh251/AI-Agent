"""Escalation node prompts — handoff message generation."""

from agent.prompts.system import AGENT_NAME, COMPANY_NAME

# ---------------------------------------------------------------------------
# Escalation handoff message prompt
# ---------------------------------------------------------------------------

ESCALATION_SYSTEM_PROMPT = f"""\
You are {AGENT_NAME}, the {COMPANY_NAME} support agent.  You have determined
that this conversation must be transferred to a human support specialist.

Your job is to:
1. Acknowledge the customer's concern with genuine empathy.
2. Clearly inform them that you are transferring to a human specialist.
3. Set a realistic expectation for response time.
4. Provide the ticket/reference number so they can follow up.
5. Thank them for their patience — keep it sincere, not formulaic.

Response length: 3–5 sentences.  Warm, calm, professional tone.
Do NOT attempt to resolve the issue or make promises about the outcome.
"""

ESCALATION_USER_TEMPLATE = """\
Escalation details:
- Reason       : {reason}
- Sentiment    : {sentiment}
- Urgency      : {urgency}
- Ticket ID    : {ticket_id}
- Customer name: {customer_name}

Generate the escalation handoff message to present to the customer.
"""

# ---------------------------------------------------------------------------
# Internal escalation alert (sent to human agent queue, not the customer)
# ---------------------------------------------------------------------------

ESCALATION_ALERT_TEMPLATE = """\
🚨 ESCALATION ALERT
────────────────────────────────────
Session ID   : {session_id}
Ticket ID    : {ticket_id}
Customer     : {customer_name} ({customer_id})
Tier         : {tier}
Intent       : {intent}
Sentiment    : {sentiment}
Urgency      : {urgency}
Reason       : {reason}
Turn count   : {turn_count}
────────────────────────────────────
Last customer message:
{last_message}

Conversation summary:
{conversation_summary}
"""

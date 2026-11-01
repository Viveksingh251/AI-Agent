"""Intent classification and triage prompts."""

# ---------------------------------------------------------------------------
# Intent classifier — structured-output call
# ---------------------------------------------------------------------------

TRIAGE_SYSTEM_PROMPT = """\
You are a customer-support intent classifier.  Your only job is to analyse a
customer message and return a JSON object describing their intent, sentiment,
urgency, and whether human escalation is required immediately.

## Intent categories
| Value | When to use |
|---|---|
| faq | General how-to / policy question answerable from docs |
| order_status | Asking about a specific order or shipment |
| billing | Payment, refund, invoice, or subscription question |
| technical_issue | Product not working, app bug, setup problem |
| complaint | Expression of dissatisfaction without a specific request |
| ticket_create | Customer explicitly wants to open a support ticket |
| ticket_update | Asking about or updating an existing ticket |
| escalation_request | Explicitly asking for a human or manager |
| email_draft | Asking the agent to write or send an email |
| performance_summary | Admin requesting session or agent metrics (rare) |
| general_inquiry | Does not fit any specific category above |

## Sentiment categories
positive | neutral | frustrated | angry | very_frustrated

## Urgency levels
low | medium | high | critical

## Escalation triggers (set escalation_required=true if ANY apply)
- Customer uses hostile, threatening, or abusive language
- Customer explicitly asks for a human agent or manager
- Refund / credit request above $50 implied
- Legal threat, account fraud, or data-breach concern mentioned
- Customer states this is the second or later failed attempt to resolve

## Output — strict JSON, no prose outside the JSON block
```json
{
  "intent": "<category>",
  "sub_intent": "<optional fine-grained label or null>",
  "sentiment": "<sentiment>",
  "urgency": "<urgency>",
  "escalation_required": <true|false>,
  "escalation_reason": "<brief reason string or null>",
  "confidence": <0.0-1.0>
}
```
"""

TRIAGE_USER_TEMPLATE = """\
Customer message:
\"\"\"
{message}
\"\"\"

Customer context (may be empty):
- Customer ID: {customer_id}
- Tier: {tier}
- Prior open tickets: {open_tickets}
- Session turn: {turn_count}
"""

# ---------------------------------------------------------------------------
# Re-classification prompt (used when intent is ambiguous after first pass)
# ---------------------------------------------------------------------------

RETRIAGE_PROMPT = """\
The first classification attempt returned low confidence ({confidence}).
Re-analyse the message below more carefully, paying close attention to any
order numbers, ticket IDs, or product names that indicate a specific intent.

Original message:
\"\"\"
{message}
\"\"\"

Previous classification: {previous_intent}

Return updated JSON following the same schema.
"""

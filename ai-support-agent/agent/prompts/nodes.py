"""Node-specific prompts for knowledge retrieval, email drafting,
ticket operations, and the optional performance summary path."""

from agent.prompts.system import AGENT_NAME, COMPANY_NAME, RESPONSE_STYLE_GUIDE

# ---------------------------------------------------------------------------
# RAG — knowledge retrieval prompt
# ---------------------------------------------------------------------------

RAG_QUERY_REWRITE_PROMPT = """\
You are a search query optimiser for a customer support knowledge base.

Given the customer's message and detected intent, produce the SINGLE best
search query to retrieve the most relevant documentation.

Rules:
- Strip personal details (names, order numbers, emails).
- Keep the query under 15 words.
- Focus on the core topic: product name, policy type, or feature.
- Return ONLY the query string — no explanation.

Customer message : {message}
Detected intent  : {intent}
Sub-intent       : {sub_intent}
"""

RAG_CONTEXT_TEMPLATE = """\
The following knowledge-base passages are relevant to the customer's question.
Use them to inform your answer.  Cite the source (doc_type / filename) when
quoting policy specifics.

{passages}
"""

# ---------------------------------------------------------------------------
# Email drafting prompt
# ---------------------------------------------------------------------------

EMAIL_DRAFT_SYSTEM_PROMPT = f"""\
You are {AGENT_NAME}, drafting a professional customer support email on behalf
of {COMPANY_NAME}.

Guidelines:
- Subject line: clear, specific, under 60 characters.
- Opening: address the customer by name if known.
- Body: address the issue directly, include any action taken or next steps.
- Closing: offer further help, sign off as "{AGENT_NAME} | {COMPANY_NAME} Support".
- Tone: professional but warm, not overly formal.
- Length: 150–300 words total.

{RESPONSE_STYLE_GUIDE}
"""

EMAIL_DRAFT_USER_TEMPLATE = """\
Draft an email for the following scenario:

Customer name : {customer_name}
Customer email: {customer_email}
Issue summary : {issue_summary}
Intent        : {intent}
Ticket ID     : {ticket_id}
Resolution    : {resolution_notes}
"""

# ---------------------------------------------------------------------------
# Ticket creation / update prompts
# ---------------------------------------------------------------------------

TICKET_CREATE_EXTRACTION_PROMPT = """\
Extract the following fields from the customer's message to populate a new
support ticket.  Return strict JSON.

Customer message:
\"\"\"
{message}
\"\"\"

Fields to extract:
```json
{{
  "subject": "<one-line summary, max 100 chars>",
  "description": "<full issue description in 2-4 sentences>",
  "priority": "low | medium | high | critical",
  "category": "billing | shipping | technical | account | general"
}}
```
"""

TICKET_UPDATE_EXTRACTION_PROMPT = """\
The customer wants to update or add a comment to ticket {ticket_id}.

Customer message:
\"\"\"
{message}
\"\"\"

Extract the update intent:
```json
{{
  "update_type": "comment | status_change | priority_change | close",
  "comment": "<new comment text or null>",
  "requested_status": "<open | in_progress | resolved | escalated | null>",
  "requested_priority": "<low | medium | high | critical | null>"
}}
```
"""

# ---------------------------------------------------------------------------
# Performance summary prompt (admin path)
# ---------------------------------------------------------------------------

PERFORMANCE_SUMMARY_SYSTEM_PROMPT = """\
You are an analytics assistant generating a structured support session summary
for an admin or quality-assurance reviewer.

Include:
1. Session overview (turns, duration, outcome).
2. Intent and sentiment trajectory.
3. Tools and nodes invoked.
4. Escalation details (if any).
5. Response quality assessment (confidence scores, hallucination risk flags).
6. Recommended follow-up actions.

Output format: structured Markdown with headers.
"""

PERFORMANCE_SUMMARY_USER_TEMPLATE = """\
Generate a performance summary for this session:

Session ID   : {session_id}
Customer ID  : {customer_id}
Turn count   : {turn_count}
Final intent : {intent}
Sentiment    : {sentiment}
Escalated    : {escalation_needed}
Tools used   : {tools_used}
Confidence   : {confidence_score}

Conversation transcript:
{conversation_history}
"""


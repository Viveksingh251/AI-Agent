# Adding a New Tool to the Agent

This guide walks through adding a working tool end-to-end, using a real example: `get_warranty_status`.

Tools live in `agent/tools/` and are registered in `agent/tools/registry.py`. Every tool follows one contract so the graph, authorization, audit trail, and observability work uniformly.

---

## Tool contract (read this first)

1. **Signature** — decorated with `@tool` (from `langchain_core.tools`); the docstring is what the LLM sees, so describe parameters clearly.
2. **Return** — always a dict with `success: bool`:
   ```python
   {"success": True, "warranty": {...}}
   {"success": False, "error": "Human-readable message for the customer."}
   ```
   Never raise for expected failures and never leak raw exceptions to the LLM.
3. **Authorization** — accept `requesting_customer_id` and `role` arguments (injected server-side — the LLM can never set them) and enforce them with `is_authorized()` / `deny()`.
4. **Audit** — call `audit_log()` (best-effort; never let it fail the tool).
5. **Errors** — wrap unexpected exceptions with `tool_error()`.

---

## Step 1 — Write the tool

Create or extend a module in `agent/tools/`. For a warranty tool, `agent/tools/customer_tools.py` is a good home:

```python
from typing import Annotated, Any, Dict, Optional

from langchain_core.tools import tool
from pydantic import Field

from agent.tools.base import audit_log, db_session, deny, is_authorized, tool_error


@tool
async def get_warranty_status(
    order_number: Annotated[str, Field(description="Order number, e.g. ORD-10234")],
    requesting_customer_id: Annotated[Optional[str], Field(description="Injected server-side; do not set")] = None,
    role: Annotated[str, Field(description="Injected server-side; do not set")] = "customer",
) -> Dict[str, Any]:
    """Look up the warranty status of a customer's order.

    Args:
        order_number: The order number to check, e.g. ORD-10234.
    """
    action = "get_warranty_status"
    try:
        async with db_session() as session:
            order = await get_order_by_number(session, order_number)  # your CRUD call

            if order is None:
                return {"success": False, "error": f"Order '{order_number}' was not found."}

            # --- Authorization: ownership or staff ---
            if not is_authorized(requesting_customer_id, str(order.customer_id), role):
                return deny(action)

            warranty = {
                "order_number": order.order_number,
                "covered": True,
                "expires_at": "2027-01-15",
            }

        await audit_log(action, {"order_number": order_number}, customer_id=requesting_customer_id)
        return {"success": True, "warranty": warranty}

    except Exception as exc:  # noqa: BLE001
        return tool_error(action, exc)
```

Key points:

- `db_session()` handles commit/rollback for you.
- `deny(action)` returns the canonical `{"success": False, "authorized": False, "error": ...}` payload and logs the denial.
- `tool_error(action, exc)` logs the exception and returns a safe customer-facing message.

## Step 2 — Register it

In `agent/tools/registry.py`:

```python
def register_all() -> Dict[str, BaseTool]:
    if ALL_TOOLS:
        return ALL_TOOLS

    from agent.tools.customer_tools import get_customer_info, get_order_status, get_warranty_status
    ...
    for t in (
        search_knowledge_base,
        get_customer_info,
        get_order_status,
        get_warranty_status,          # ← add here
        create_support_ticket,
        ...
    ):
        ALL_TOOLS[t.name] = t
```

`register_all()` is idempotent and is also what binds the tool list to the LLM (`get_agent_tools()` in `graph.py`).

## Step 3 — Make the agent use it

Two independent mechanisms; use either or both:

### a) Tool-calling LLM (automatic)
If the LLM supports tool calling, the `execute_tools` node picks up `tool_calls` by name from the registry — no extra work. Ensure the docstring/args are clear enough for the model to call it correctly.

### b) Intent fallback map (no tool-calling LLM)
Add the intent → tool mapping in `_FALLBACK_MAP` (registry.py) and, if the intent is new, teach routing:

1. Add the intent constant in `agent/state.py` (e.g. `INTENT_WARRANTY = "warranty"`), and add it to `ALL_INTENTS` and `TOOL_INTENTS`.
2. Mention it in the triage prompt (`agent/prompts/triage.py`) so the classifier can produce it.
3. Route it in `agent/graph.py::route_after_triage` (add to the `execute_tools` branch) — and add it to `TOOL_INTENTS` handling if needed.
4. Add a fallback entry:

```python
_FALLBACK_MAP: Dict[str, List[str]] = {
    ...
    "warranty": ["get_warranty_status", "get_order_status"],
}
```

If the tool needs arguments extracted from the user's message (order number, ticket id), add a branch in `_intent_fallback_calls()` that regex-extracts them, following the `get_order_status` example.

### c) Promote results into graph state (optional)
If downstream nodes (response generation) should see the result, add promotion in `execute_tools`:

```python
if name == "get_warranty_status" and payload.get("success"):
    state_update["warranty_data"] = payload.get("warranty")
```

(Add `warranty_data: Optional[Dict[str, Any]]` to `SupportAgentState` in `agent/state.py` first.)

## Step 4 — Test it

Add tests in `tests/test_tools.py` following the existing patterns:

```python
import pytest
from agent.tools.customer_tools import get_warranty_status


@pytest.mark.asyncio
async def test_warranty_denied_for_other_customer():
    result = await get_warranty_status.ainvoke({
        "order_number": "ORD-10234",
        "requesting_customer_id": "99",   # not the owner
        "role": "customer",
    })
    assert result["success"] is False
    assert result.get("authorized") is False


@pytest.mark.asyncio
async def test_warranty_allowed_for_owner(monkeypatch):
    ...  # mock the CRUD layer, assert success payload


@pytest.mark.asyncio
async def test_warranty_staff_bypass():
    result = await get_warranty_status.ainvoke({
        "order_number": "ORD-10234",
        "requesting_customer_id": "99",
        "role": "admin",
    })
    assert result["success"] is True
```

Cover at minimum: **owner allowed**, **other customer denied**, **staff allowed**, **not-found / DB failure** paths.

Run:

```bash
pytest tests/test_tools.py -v
```

## Step 5 — Observability comes for free

Once registered, the tool automatically gets:

- an OpenTelemetry span `tool.<name>` per invocation,
- Prometheus counters `support_agent_tool_usage_total{tool_name="<name>", status=...}` and histogram `support_agent_tool_duration_seconds`,
- an audit row in `agent_action_logs` (visible via `GET /api/v1/admin/agent-logs`).

No extra code needed — this is handled by `execute_tools` and `app/monitoring.py`.

---

## Checklist

- [ ] Tool function with `@tool`, clear docstring & typed args
- [ ] Returns `{"success": bool, ...}`; uses `tool_error()` for exceptions
- [ ] Enforces `is_authorized()` / returns `deny()` when unauthorized
- [ ] Calls `audit_log()`
- [ ] Registered in `register_all()`
- [ ] Routing wired (tool-calling docstring quality and/or `_FALLBACK_MAP` + intent)
- [ ] State promotion added if the response node needs the data
- [ ] Tests: allowed / denied / staff / failure

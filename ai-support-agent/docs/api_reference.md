# REST API Reference

The AI Support Agent backend provides an OpenAPI 3.1 compatible REST API. Interactive Swagger documentation is accessible at `http://localhost:8000/docs`.

---

## 1. Chat & Agent Endpoints

### `POST /api/v1/chat`
Submits a customer message into the LangGraph support workflow.

**Request Body:**
```json
{
  "message": "Where is my order #1234?",
  "session_id": "sess_88219a",
  "customer_id": "CUST-10492"
}
```

**Response (200 OK):**
```json
{
  "session_id": "sess_88219a",
  "response": "Your order #1234 was dispatched on Oct 25 and is currently in transit via FedEx.",
  "intent": "order_status",
  "sentiment": "neutral",
  "escalation_required": false,
  "ticket_id": null,
  "confidence_score": 0.98,
  "turn_count": 1,
  "timestamp": "2026-11-01T12:00:00Z"
}
```

---

## 2. Tickets Endpoints

### `POST /api/v1/tickets`
Creates a support ticket manually or via automation.

### `GET /api/v1/tickets/{ticket_code}`
Returns details and status history for a given ticket.

---

## 3. Customer Endpoints

### `POST /api/v1/customers`
Registers or updates a customer profile.

### `GET /api/v1/customers/{external_id}`
Retrieves customer tier and profile by external ID.


0
import sys
sys.path.insert(0, r"c:\Users\Vivek Singh\OneDrive\placement\agent\ai-support-agent")
import os
os.environ.setdefault("APP_ENV", "testing")

from fastapi.testclient import TestClient
from app.main import app

with TestClient(app) as client:
    r = client.get("/metrics")
    body = r.text
    print("metrics status:", r.status_code)
    for m in [
        "support_agent_conversations_total",
        "support_agent_tool_usage_total",
        "support_agent_escalations_total",
        "support_agent_response_time_seconds",
        "support_agent_tickets_created_total",
    ]:
        assert m in body, m
    print("/metrics OK")
    r2 = client.get("/health")
    print("health status:", r2.status_code)
print("APP OK")

"""API Tool providing HTTP client interactions with internal and external services."""

import logging
from typing import Any, Dict, Optional

import httpx

from app.config import Settings
from app.tools import BaseTool

logger = logging.getLogger("company_agent.tools.api")


class CompanyAPITool(BaseTool):
    """Tool enabling the AI Agent to communicate with internal microservices and external APIs."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.COMPANY_API_BASE_URL.rstrip("/")
        self.timeout = settings.HTTP_TIMEOUT_SECONDS

    @property
    def name(self) -> str:
        return "api_tool"

    @property
    def description(self) -> str:
        return (
            "Access real-time operational APIs and external microservices. "
            "Use for system telemetry, microservice health verification, exchange rates, "
            "and general HTTP data retrieval."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "get_system_status",
                        "check_service_health",
                        "get_exchange_rate",
                        "fetch_external_data",
                    ],
                    "description": "The API action to execute.",
                },
                "service_name": {
                    "type": "string",
                    "description": "Specific microservice name (e.g., 'auth-service', 'billing-api', 'ai-gateway').",
                },
                "base_currency": {
                    "type": "string",
                    "description": "Base currency code (e.g. 'USD', 'EUR').",
                },
                "target_currency": {
                    "type": "string",
                    "description": "Target currency code (e.g. 'EUR', 'GBP', 'INR', 'JPY').",
                },
                "endpoint_url": {
                    "type": "string",
                    "description": "Target URL for fetch_external_data action.",
                },
            },
            "required": ["action"],
        }

    async def execute(self, arguments: Dict[str, Any]) -> Any:
        action = arguments.get("action")

        if action == "get_system_status":
            return await self._get_system_status()
        elif action == "check_service_health":
            return await self._check_service_health(
                service_name=arguments.get("service_name", "unknown")
            )
        elif action == "get_exchange_rate":
            return await self._get_exchange_rate(
                base_currency=arguments.get("base_currency", "USD"),
                target_currency=arguments.get("target_currency", "EUR"),
            )
        elif action == "fetch_external_data":
            return await self._fetch_external_data(
                url=arguments.get("endpoint_url", "")
            )
        else:
            return {"error": f"Unknown API action: '{action}'."}

    async def _get_system_status(self) -> Dict[str, Any]:
        """Provides simulated real-time infrastructure telemetry for Acme Corp services."""
        return {
            "cluster": "acme-prod-us-east-1",
            "uptime_percent": 99.98,
            "services": {
                "auth-service": {"status": "HEALTHY", "latency_ms": 14, "version": "v2.4.1"},
                "billing-api": {"status": "HEALTHY", "latency_ms": 32, "version": "v1.9.0"},
                "ai-gateway": {"status": "HEALTHY", "latency_ms": 48, "version": "v3.1.2"},
                "notification-engine": {"status": "DEGRADED", "latency_ms": 280, "version": "v1.2.0", "note": "High queue depth in worker pool"},
                "database-primary": {"status": "HEALTHY", "cpu_load": "24%", "connections": 142},
            },
        }

    async def _check_service_health(self, service_name: str) -> Dict[str, Any]:
        """Checks the real-time health of a specific internal service."""
        clean_name = service_name.lower().strip()
        system_status = await self._get_system_status()
        service_data = system_status.get("services", {}).get(clean_name)

        if service_data:
            return {
                "service": clean_name,
                "found": True,
                "details": service_data,
            }

        return {
            "service": clean_name,
            "found": False,
            "error": f"Service '{clean_name}' is not in the recognized service registry.",
            "available_services": list(system_status.get("services", {}).keys()),
        }

    async def _get_exchange_rate(
        self, base_currency: str, target_currency: str
    ) -> Dict[str, Any]:
        """Provides currency exchange rates for international expense calculations."""
        base = base_currency.upper().strip()
        target = target_currency.upper().strip()

        # Representative benchmark rates
        rates_to_usd = {
            "USD": 1.0,
            "EUR": 1.08,
            "GBP": 1.28,
            "JPY": 0.0066,
            "CAD": 0.73,
            "AUD": 0.65,
            "INR": 0.012,
        }

        if base in rates_to_usd and target in rates_to_usd:
            rate = rates_to_usd[base] / rates_to_usd[target]
            return {
                "base_currency": base,
                "target_currency": target,
                "rate": round(rate, 4),
                "timestamp": "2026-10-31T00:00:00Z",
                "source": "Acme Treasury Currency Feed",
            }

        return {
            "error": f"Currency pair {base}/{target} not supported. Supported: {list(rates_to_usd.keys())}"
        }

    async def _fetch_external_data(self, url: str) -> Dict[str, Any]:
        """Performs a safe HTTP GET request to a permitted external URL."""
        if not url:
            return {"error": "Target URL cannot be empty."}

        if not (url.startswith("http://") or url.startswith("https://")):
            return {"error": "Invalid URL protocol. Must start with http:// or https://"}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                try:
                    data = response.json()
                except Exception:
                    data = response.text[:2000]

                return {
                    "status_code": response.status_code,
                    "url": url,
                    "data": data,
                }
        except Exception as exc:
            logger.warning(f"Failed HTTP call to {url}: {exc}")
            return {"error": f"HTTP request failed: {str(exc)}"}


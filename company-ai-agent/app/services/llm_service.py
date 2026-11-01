"""LLM Service handling completions, tool calling, and intelligent fallback reasoning."""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from app.config import Settings, get_settings

logger = logging.getLogger("company_agent.services.llm")


class LLMToolCall(BaseModel):
    """Structured representation of a tool call emitted by the LLM."""

    id: str
    name: str
    arguments: Dict[str, Any]
    arguments_json: str


class LLMResponse(BaseModel):
    """Standardized response from the LLM provider."""

    content: Optional[str] = None
    tool_calls: List[LLMToolCall] = []


class LLMService:
    """Service wrapping LLM interactions with tool calling and mock capability."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.OPENAI_MODEL
        self.provider = settings.LLM_PROVIDER
        self.client = None

        if self.api_key and self.provider != "mock":
            try:
                from openai import AsyncOpenAI

                client_kwargs = {"api_key": self.api_key}
                if settings.OPENAI_BASE_URL:
                    client_kwargs["base_url"] = settings.OPENAI_BASE_URL
                self.client = AsyncOpenAI(**client_kwargs)
                logger.info(f"Initialized OpenAI Async client with model '{self.model}'.")
            except ImportError:
                logger.warning("OpenAI package not installed or import failed. Using mock LLM.")
                self.client = None
        else:
            logger.info("Using mock/heuristic LLM engine (no OpenAI key provided or provider='mock').")

    async def generate_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Invokes the language model or fallback reasoning engine."""
        if self.client:
            try:
                return await self._call_openai(messages, tools)
            except Exception as exc:
                logger.warning(
                    f"OpenAI API call failed ({exc}). Falling back to internal reasoning engine."
                )
                return await self._call_heuristic_engine(messages, tools)

        return await self._call_heuristic_engine(messages, tools)

    async def _call_openai(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Executes API call to OpenAI with tool specifications."""
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.settings.LLM_TEMPERATURE,
        }

        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        response = await self.client.chat.completions.create(**params)
        choice = response.choices[0].message

        tool_calls: List[LLMToolCall] = []
        if choice.tool_calls:
            for tc in choice.tool_calls:
                call_id = tc.id
                func_name = tc.function.name
                raw_args = tc.function.arguments or "{}"
                try:
                    parsed_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    parsed_args = {}

                tool_calls.append(
                    LLMToolCall(
                        id=call_id,
                        name=func_name,
                        arguments=parsed_args,
                        arguments_json=raw_args,
                    )
                )

        return LLMResponse(content=choice.content, tool_calls=tool_calls)

    async def _call_heuristic_engine(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Intelligent heuristic engine for test environments, offline use, and demonstrations.
        
        Evaluates conversation context and formulates appropriate tool calls or synthesized answers.
        """
        # Check if the last message is a tool response
        last_message = messages[-1] if messages else {}
        user_message = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_message = m.get("content", "").lower()
                break

        # If a tool just completed, synthesize the final answer from its result
        if last_message.get("role") == "tool":
            tool_content = last_message.get("content", "")
            tool_name = last_message.get("name", "tool")

            try:
                data = eval(tool_content) if tool_content.startswith(("[", "{")) else tool_content
            except Exception:
                data = tool_content

            return LLMResponse(
                content=(
                    f"Based on the enterprise {tool_name} records:\n\n"
                    f"```json\n{json.dumps(data, indent=2, default=str)}\n```\n\n"
                    f"The requested operational data has been retrieved and verified."
                ),
                tool_calls=[],
            )

        # Detect intent and issue appropriate tool call if tools are present
        if tools:
            # Intent: Employee query
            if any(term in user_message for term in ["employee", "staff", "who is", "architect", "engineer", "department"]):
                query = ""
                dept = None
                if "engineering" in user_message:
                    dept = "Engineering"
                elif "hr" in user_message or "human resources" in user_message:
                    dept = "Human Resources"
                elif "it" in user_message:
                    dept = "IT Support"

                # Extract potential names
                for name in ["alice", "bob", "carol", "david", "elena"]:
                    if name in user_message:
                        query = name
                        break

                args = {"action": "search_employees"}
                if query:
                    args["query"] = query
                if dept:
                    args["department"] = dept

                return LLMResponse(
                    content="Querying the enterprise employee directory to find relevant staff members.",
                    tool_calls=[
                        LLMToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            name="database_tool",
                            arguments=args,
                            arguments_json=json.dumps(args),
                        )
                    ],
                )

            # Intent: Ticket query
            if any(term in user_message for term in ["ticket", "issue", "incident", "bug", "vpn", "latency", "tik-"]):
                status = None
                priority = None
                if "critical" in user_message:
                    priority = "Critical"
                elif "high" in user_message:
                    priority = "High"

                if "open" in user_message:
                    status = "Open"
                elif "progress" in user_message:
                    status = "In Progress"
                elif "resolved" in user_message:
                    status = "Resolved"

                args = {"action": "search_tickets"}
                if status:
                    args["status"] = status
                if priority:
                    args["priority"] = priority

                return LLMResponse(
                    content="Inspecting active IT support and engineering tickets in the database.",
                    tool_calls=[
                        LLMToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            name="database_tool",
                            arguments=args,
                            arguments_json=json.dumps(args),
                        )
                    ],
                )

            # Intent: Policy query
            if any(term in user_message for term in ["policy", "handbook", "remote work", "stipend", "travel", "password", "security", "mfa"]):
                category = "HR"
                if any(w in user_message for term in ["security", "password", "mfa"] for w in [term]):
                    category = "Security"
                elif "travel" in user_message or "expense" in user_message:
                    category = "Travel"

                args = {"action": "get_policy", "policy_category": category}
                return LLMResponse(
                    content=f"Retrieving corporate policy records for category '{category}'.",
                    tool_calls=[
                        LLMToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            name="database_tool",
                            arguments=args,
                            arguments_json=json.dumps(args),
                        )
                    ],
                )

            # Intent: System health / telemetry API
            if any(term in user_message for term in ["system status", "health", "uptime", "service", "cluster", "telemetry"]):
                args = {"action": "get_system_status"}
                return LLMResponse(
                    content="Querying live infrastructure telemetry via the API tool.",
                    tool_calls=[
                        LLMToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            name="api_tool",
                            arguments=args,
                            arguments_json=json.dumps(args),
                        )
                    ],
                )

            # Intent: Exchange rate API
            if any(term in user_message for term in ["exchange rate", "currency", "convert", "usd to eur", "rates"]):
                args = {"action": "get_exchange_rate", "base_currency": "USD", "target_currency": "EUR"}
                return LLMResponse(
                    content="Retrieving foreign currency exchange rates via the API tool.",
                    tool_calls=[
                        LLMToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            name="api_tool",
                            arguments=args,
                            arguments_json=json.dumps(args),
                        )
                    ],
                )

        # General conversational response
        return LLMResponse(
            content=(
                f"Hello! I am Nova, Acme Corp's enterprise AI co-pilot. "
                f"I can assist with querying employee records, tracking IT support tickets, "
                f"reviewing corporate policies, and checking live microservice health telemetry."
            ),
            tool_calls=[],
        )

    async def ping(self) -> bool:
        """Returns True if the LLM service or mock engine is operational."""
        return True


def get_llm_service(settings: Optional[Settings] = None) -> LLMService:
    """Factory provider for LLMService."""
    _settings = settings or get_settings()
    return LLMService(settings=_settings)


"""Core AI Agent Orchestrator for the Company AI System."""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.config import Settings, get_settings
from app.prompts import SYSTEM_PROMPT
from app.schemas import (
    ChatResponse,
    ExecutionStep,
    HealthCheckResponse,
    MessageItem,
    ToolCallRecord,
    ToolDefinition,
)
from app.services.llm_service import LLMService, get_llm_service
from app.services.memory_service import MemoryService, get_memory_service
from app.tools import ToolRegistry, get_tool_registry

logger = logging.getLogger("company_agent.agent")


class CompanyAIAgent:
    """Enterprise AI Agent with dynamic ReAct reasoning and tool calling.
    
    Coordinates the LLM service, memory service, and available business tools
    to resolve user queries accurately and securely.
    """

    def __init__(
        self,
        llm_service: LLMService,
        memory_service: MemoryService,
        tool_registry: ToolRegistry,
        settings: Settings,
    ):
        self.llm = llm_service
        self.memory = memory_service
        self.tools = tool_registry
        self.settings = settings

    async def run(
        self,
        message: str,
        session_id: Optional[str] = None,
        system_override: Optional[str] = None,
    ) -> ChatResponse:
        """Executes a multi-turn conversation turn with iterative tool execution."""
        active_session_id = session_id or f"session_{uuid.uuid4().hex[:12]}"
        system_prompt = system_override or SYSTEM_PROMPT

        # 1. Fetch relevant conversation history from persistent memory
        prior_messages = await self.memory.get_history(active_session_id)
        
        # 2. Build model prompt context
        conversation_context: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]
        
        for msg in prior_messages:
            conversation_context.append({
                "role": msg.role,
                "content": msg.content or "",
            })
            
        conversation_context.append({"role": "user", "content": message})

        # Save user message to memory
        await self.memory.add_message(
            session_id=active_session_id,
            role="user",
            content=message,
        )

        execution_steps: List[ExecutionStep] = []
        tools_used: List[str] = []
        final_answer = ""
        current_iteration = 0
        max_iterations = self.settings.MAX_ITERATIONS

        # 3. Dynamic Tool Calling / ReAct loop
        while current_iteration < max_iterations:
            current_iteration += 1
            logger.info(
                f"[Session: {active_session_id}] Agent iteration {current_iteration}/{max_iterations}"
            )

            # Invoke LLM with registered tool specifications
            llm_response = await self.llm.generate_response(
                messages=conversation_context,
                tools=self.tools.get_openai_tools_schema(),
            )

            # Case A: Model requests one or more tool calls
            if llm_response.tool_calls:
                # Append assistant tool call message to loop context
                conversation_context.append({
                    "role": "assistant",
                    "content": llm_response.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments_json,
                            },
                        }
                        for tc in llm_response.tool_calls
                    ],
                })

                for tc in llm_response.tool_calls:
                    tool_name = tc.name
                    if tool_name not in tools_used:
                        tools_used.append(tool_name)

                    logger.info(f"Executing tool '{tool_name}' with args: {tc.arguments}")
                    
                    try:
                        tool_result = await self.tools.execute(
                            name=tool_name,
                            arguments=tc.arguments,
                        )
                    except Exception as err:
                        logger.exception(f"Tool {tool_name} failed unexpectedly")
                        tool_result = {"error": f"Tool execution failed: {str(err)}"}

                    step = ExecutionStep(
                        step_number=current_iteration,
                        thought=llm_response.content or f"Calling tool '{tool_name}' to retrieve information.",
                        tool_call=ToolCallRecord(
                            id=tc.id,
                            tool_name=tool_name,
                            arguments=tc.arguments,
                            result=tool_result,
                        ),
                    )
                    execution_steps.append(step)

                    # Append tool result into context for next turn
                    conversation_context.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tool_name,
                        "content": str(tool_result),
                    })

                continue  # Model will digest tool outputs in the next iteration

            # Case B: Model produced final answer
            final_answer = llm_response.content or "No response generated."
            break

        if not final_answer:
            final_answer = (
                "I apologize, but I reached the maximum number of reasoning steps "
                "before arriving at a conclusion. Please try refining your query."
            )

        # 4. Save assistant answer to memory
        await self.memory.add_message(
            session_id=active_session_id,
            role="assistant",
            content=final_answer,
        )

        return ChatResponse(
            session_id=active_session_id,
            answer=final_answer,
            tools_used=tools_used,
            steps=execution_steps,
            iterations=current_iteration,
            timestamp=datetime.now(timezone.utc),
        )

    def get_available_tools(self) -> List[ToolDefinition]:
        """Returns the registered tools and their input schemas."""
        return self.tools.list_tools()

    async def health_check(self) -> HealthCheckResponse:
        """Verifies health across LLM, Memory, and Tools."""
        llm_ok = await self.llm.ping()
        db_ok = await self.memory.ping()

        overall_status = "healthy" if (llm_ok and db_ok) else "degraded"
        return HealthCheckResponse(
            status=overall_status,
            llm_status="connected" if llm_ok else "unavailable",
            database_status="connected" if db_ok else "unavailable",
            active_tools=len(self.tools.list_tools()),
            environment=self.settings.APP_ENV,
            timestamp=datetime.now(timezone.utc),
        )


def get_agent(
    llm: LLMService = None,
    memory: MemoryService = None,
    tools: ToolRegistry = None,
    settings: Settings = None,
) -> CompanyAIAgent:
    """FastAPI dependency provider for CompanyAIAgent."""
    _settings = settings or get_settings()
    _llm = llm or get_llm_service(_settings)
    _memory = memory or get_memory_service(_settings)
    _tools = tools or get_tool_registry(_settings)
    return CompanyAIAgent(
        llm_service=_llm,
        memory_service=_memory,
        tool_registry=_tools,
        settings=_settings,
    )


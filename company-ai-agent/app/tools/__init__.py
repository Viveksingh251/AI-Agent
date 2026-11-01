"""Tool Registration and Base Tool Definitions for Company AI Agent."""

import abc
import logging
from typing import Any, Dict, List, Optional

from app.config import Settings, get_settings
from app.schemas import ToolDefinition

logger = logging.getLogger("company_agent.tools")


class BaseTool(abc.ABC):
    """Abstract Base Class for all tools executable by the Company AI Agent."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique identifier name for the tool."""
        pass

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Detailed description explaining when and how to use the tool."""
        pass

    @property
    @abc.abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """JSON Schema dictionary describing required and optional parameters."""
        pass

    @abc.abstractmethod
    async def execute(self, arguments: Dict[str, Any]) -> Any:
        """Asynchronously executes the tool with parsed input arguments."""
        pass

    def to_openai_schema(self) -> Dict[str, Any]:
        """Converts tool specification to OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    def to_tool_definition(self) -> ToolDefinition:
        """Returns Pydantic ToolDefinition representation."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters_schema,
        )


class ToolRegistry:
    """Registry managing available agent tools and execution dispatching."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Registers a tool in the active agent toolkit."""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get(self, name: str) -> Optional[BaseTool]:
        """Fetches a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        """Returns definitions of all registered tools."""
        return [tool.to_tool_definition() for tool in self._tools.values()]

    def get_openai_tools_schema(self) -> List[Dict[str, Any]]:
        """Returns all tools formatted for OpenAI function calling."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    async def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Locates and executes a tool safely."""
        tool = self.get(name)
        if not tool:
            return {"error": f"Tool '{name}' not found in registry."}

        try:
            return await tool.execute(arguments)
        except Exception as exc:
            logger.exception(f"Unhandled error while executing tool '{name}'")
            return {"error": f"Execution error in '{name}': {str(exc)}"}


_registry_instance: Optional[ToolRegistry] = None


def get_tool_registry(settings: Optional[Settings] = None) -> ToolRegistry:
    """Singleton provider for ToolRegistry with default tools loaded."""
    global _registry_instance
    if _registry_instance is None:
        _settings = settings or get_settings()
        registry = ToolRegistry()

        # Import tools lazily to avoid circular imports
        from app.tools.api_tool import CompanyAPITool
        from app.tools.database_tool import DatabaseTool

        registry.register(DatabaseTool(settings=_settings))
        registry.register(CompanyAPITool(settings=_settings))
        _registry_instance = registry

    return _registry_instance


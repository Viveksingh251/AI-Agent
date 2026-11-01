"""Unit and Integration Tests for Company AI Agent and Tools."""

import pytest
import pytest_asyncio

from app.agent import CompanyAIAgent
from app.config import Settings
from app.models import init_db_and_seed
from app.services.llm_service import LLMService
from app.services.memory_service import MemoryService
from app.tools import ToolRegistry
from app.tools.api_tool import CompanyAPITool
from app.tools.database_tool import DatabaseTool


@pytest_asyncio.fixture(autouse=True)
async def setup_test_database():
    """Ensure database schema is created and seeded before each test."""
    await init_db_and_seed()


@pytest.fixture
def test_settings():
    return Settings(
        APP_ENV="testing",
        LLM_PROVIDER="mock",
        DATABASE_URL="sqlite+aiosqlite:///:memory:",
    )


@pytest.fixture
def agent_components(test_settings):
    llm = LLMService(settings=test_settings)
    memory = MemoryService(settings=test_settings)
    registry = ToolRegistry()
    registry.register(DatabaseTool(settings=test_settings))
    registry.register(CompanyAPITool(settings=test_settings))

    agent = CompanyAIAgent(
        llm_service=llm,
        memory_service=memory,
        tool_registry=registry,
        settings=test_settings,
    )
    return {
        "agent": agent,
        "llm": llm,
        "memory": memory,
        "registry": registry,
        "settings": test_settings,
    }


@pytest.mark.asyncio
async def test_agent_initialization(agent_components):
    """Test that agent initializes with all expected tools and properties."""
    agent: CompanyAIAgent = agent_components["agent"]
    tools = agent.get_available_tools()
    tool_names = [t.name for t in tools]

    assert "database_tool" in tool_names
    assert "api_tool" in tool_names
    assert len(tools) >= 2


@pytest.mark.asyncio
async def test_database_tool_search_employees(agent_components):
    """Test searching employees via DatabaseTool."""
    registry: ToolRegistry = agent_components["registry"]
    result = await registry.execute(
        "database_tool",
        {"action": "search_employees", "department": "Engineering"},
    )

    assert isinstance(result, list)
    assert len(result) >= 1
    assert any(emp["name"] == "Alice Johnson" for emp in result)


@pytest.mark.asyncio
async def test_database_tool_security_guardrail(agent_components):
    """Test SQL injection and destructive query prevention in DatabaseTool."""
    registry: ToolRegistry = agent_components["registry"]
    malicious_query = "DROP TABLE employees;"
    result = await registry.execute(
        "database_tool",
        {"action": "execute_read_query", "sql_query": malicious_query},
    )

    assert "error" in result
    assert "Security Violation" in result["error"]


@pytest.mark.asyncio
async def test_api_tool_system_status(agent_components):
    """Test retrieving infrastructure telemetry via CompanyAPITool."""
    registry: ToolRegistry = agent_components["registry"]
    result = await registry.execute("api_tool", {"action": "get_system_status"})

    assert "cluster" in result
    assert "services" in result
    assert "auth-service" in result["services"]


@pytest.mark.asyncio
async def test_api_tool_exchange_rate(agent_components):
    """Test currency exchange rate calculations via CompanyAPITool."""
    registry: ToolRegistry = agent_components["registry"]
    result = await registry.execute(
        "api_tool",
        {"action": "get_exchange_rate", "base_currency": "USD", "target_currency": "EUR"},
    )

    assert result["base_currency"] == "USD"
    assert result["target_currency"] == "EUR"
    assert "rate" in result
    assert result["rate"] > 0


@pytest.mark.asyncio
async def test_agent_run_employee_query(agent_components):
    """Test end-to-end agent reasoning and tool dispatch for employee inquiry."""
    agent: CompanyAIAgent = agent_components["agent"]
    response = await agent.run(
        message="Who is the lead cloud architect in engineering?",
        session_id="test_session_emp_01",
    )

    assert response.session_id == "test_session_emp_01"
    assert len(response.steps) >= 1
    assert "database_tool" in response.tools_used
    assert "Alice Johnson" in response.answer or "lead cloud architect" in response.answer.lower()


@pytest.mark.asyncio
async def test_agent_memory_persistence(agent_components):
    """Test that multi-turn interactions retain session history in memory."""
    agent: CompanyAIAgent = agent_components["agent"]
    session_id = "test_memory_thread_99"

    # Turn 1
    resp1 = await agent.run(
        message="What is the policy for remote work and home office stipend?",
        session_id=session_id,
    )
    assert resp1.session_id == session_id

    # Turn 2
    resp2 = await agent.run(
        message="Are there any critical tickets currently open?",
        session_id=session_id,
    )
    assert resp2.session_id == session_id

    # Verify history
    memory: MemoryService = agent_components["memory"]
    history = await memory.get_history(session_id)
    assert len(history) == 4  # 2 user messages + 2 assistant messages


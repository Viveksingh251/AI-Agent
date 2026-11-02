"""Shared pytest fixtures: async DB, factories, API client, mocked LLM.

Environment is configured BEFORE any app import so
``app.config.get_settings`` picks up the test SQLite database.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# Project root on sys.path + test environment
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./_pytest_main.db"
os.environ.setdefault("SECRET_KEY", "pytest-secret-key-not-for-production")
os.environ.setdefault("ALLOWED_ORIGINS", '["*"]')

from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine  # noqa: E402

from app.security import create_access_token  # noqa: E402
from database import crud  # noqa: E402
from database.models import Base, CustomerTier, EmployeeRole, OrderStatus, TicketPriority  # noqa: E402

DB_FILE = PROJECT_ROOT / "_pytest_main.db"


# ---------------------------------------------------------------------------
# Event loop for pytest-asyncio
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db_engine():
    """Fresh file-backed SQLite database per test (clean, isolated)."""
    if DB_FILE.exists():
        DB_FILE.unlink()
    engine = create_async_engine("sqlite+aiosqlite:///_pytest_main.db", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()
    if DB_FILE.exists():
        DB_FILE.unlink()


@pytest_asyncio.fixture
async def db_session(db_engine) -> AsyncIterator[AsyncSession]:
    """Session bound to the isolated test engine, committed per-test."""
    maker = async_sessionmaker(bind=db_engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        yield session
        await session.commit()


# ---------------------------------------------------------------------------
# Factory fixtures — customers, employees, orders, tickets
# ---------------------------------------------------------------------------


class Factory:
    """Creates DB rows within the current test session."""

    def __init__(self, session: AsyncSession, counter: Dict[str, int]):
        self._session = session
        self._counter = counter

    async def customer(self, *, email: Optional[str] = None, name: str = "Test Customer",
                       password: str = "Password123!", tier: Any = CustomerTier.FREE,
                       **kwargs: Any) -> Any:
        n = self._counter["customer"]
        self._counter["customer"] += 1
        return await crud.create_customer(
            self._session,
            email=email or f"cust{n}.test@example.com",
            name=name,
            password=password,
            tier=tier,
            **kwargs,
        )

    async def employee(self, *, role: Any = EmployeeRole.SUPPORT, email: Optional[str] = None,
                       password: str = "StaffPass123!", **kwargs: Any) -> Any:
        n = self._counter["employee"]
        self._counter["employee"] += 1
        return await crud.create_employee(
            self._session,
            email=email or f"staff{n}.test@support.example.com",
            name=f"Staff {n}",
            password=password,
            role=role,
            **kwargs,
        )

    async def order(self, customer_id: int, *, order_number: Optional[str] = None,
                    total: str = "99.99", status: Any = OrderStatus.PROCESSING,
                    **kwargs: Any) -> Any:
        n = self._counter["order"]
        self._counter["order"] += 1
        return await crud.create_order(
            self._session,
            customer_id=customer_id,
            order_number=order_number or f"ORD-TEST-{n:04d}",
            total_amount=total,
            items=[{"sku": f"SKU-{n}", "name": "Test Widget", "qty": 1}],
            status=status,
            **kwargs,
        )

    async def ticket(self, customer_id: int, *, subject: str = "Test issue",
                     description: str = "A test ticket description",
                     priority: Any = TicketPriority.MEDIUM,
                     status: str = "open", ticket_number: Optional[str] = None, **kwargs: Any) -> Any:
        n = self._counter["ticket"]
        self._counter["ticket"] += 1
        return await crud.create_support_ticket(
            self._session,
            ticket_number=ticket_number or f"TCK-TEST-{n:04d}",
            customer_id=customer_id,
            subject=subject,
            description=description,
            priority=priority,
            status=status,
            created_by_agent=True,
            **kwargs,
        )

    async def flush(self) -> None:
        await self._session.commit()


@pytest_asyncio.fixture
async def factory(db_session: AsyncSession) -> AsyncIterator[Factory]:
    """Object-mother style factory for seeding test data."""
    yield Factory(db_session, {"customer": 0, "employee": 0, "order": 0, "ticket": 0})
    await db_session.commit()


# ---------------------------------------------------------------------------
# Common seeded fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def customer(factory: Factory):
    """A customer with one order and one open ticket; commits and returns dict."""
    c = await factory.customer(email="jane.suite@example.com", name="Jane Suite")
    o = await factory.order(c.id, order_number="ORD-SUITE-0001")
    t = await factory.ticket(c.id, subject="Broken charger", description="Charger sparks on plug-in")
    await factory.flush()
    return {"customer": c, "order": o, "ticket": t}


@pytest_asyncio.fixture
async def other_customer(factory: Factory):
    """A second customer — used for cross-tenant authorization tests."""
    c = await factory.customer(email="mallory.suite@example.com", name="Mallory Rival")
    o = await factory.order(c.id, order_number="ORD-SUITE-0002")
    t = await factory.ticket(c.id, subject="Other person's issue", description="Not Jane's ticket")
    await factory.flush()
    return {"customer": c, "order": o, "ticket": t}


@pytest_asyncio.fixture
async def staff(factory: Factory):
    e = await factory.employee(role=EmployeeRole.SUPPORT)
    await factory.flush()
    return e


@pytest_asyncio.fixture
async def admin(factory: Factory):
    e = await factory.employee(role=EmployeeRole.ADMIN)
    await factory.flush()
    return e


# ---------------------------------------------------------------------------
# Auth token helpers
# ---------------------------------------------------------------------------


def token_for(user: Any, role: str, user_type: str = "customer") -> str:
    return create_access_token({
        "sub": user.email,
        "user_id": user.id,
        "name": user.name,
        "role": role,
        "user_type": user_type,
    })


@pytest.fixture
def customer_headers(customer) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token_for(customer['customer'], 'customer')}"}


@pytest.fixture
def other_customer_headers(other_customer) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token_for(other_customer['customer'], 'customer')}"}


@pytest.fixture
def staff_headers(staff) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token_for(staff, 'support_agent', user_type='employee')}"}


@pytest.fixture
def admin_headers(admin) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token_for(admin, 'admin', user_type='employee')}"}


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def api_client() -> AsyncIterator[AsyncClient]:
    """Async TestClient for the FastAPI app, with lifespan executed."""
    from app.main import app

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


# ---------------------------------------------------------------------------
# Mocked LLM for LangGraph integration tests
# ---------------------------------------------------------------------------


class FakeLLM:
    """Deterministic scriptable LLM replacement.

    ``responses`` maps a substring of the prompt → reply text. Unmatched
    prompts get ``default``. Records every prompt for assertions.
    """

    def __init__(self, responses: Optional[Dict[str, str]] = None, default: str = "ok"):
        self.responses = responses or {}
        self.default = default
        self.prompts: list[str] = []

    def _reply(self, prompt: str) -> str:
        self.prompts.append(prompt)
        for needle, reply in self.responses.items():
            if needle in prompt:
                return reply
        return self.default

    async def ainvoke(self, messages: Any, **_: Any) -> Any:
        from langchain_core.messages import AIMessage

        prompt = "\n".join(
            str(getattr(m, "content", m)) for m in (messages if isinstance(messages, list) else [messages])
        )
        return AIMessage(content=self._reply(prompt))

    def invoke(self, messages: Any, **_: Any) -> Any:
        from langchain_core.messages import AIMessage

        prompt = "\n".join(
            str(getattr(m, "content", m)) for m in (messages if isinstance(messages, list) else [messages])
        )
        return AIMessage(content=self._reply(prompt))


@pytest.fixture
def agent_graph():
    """Fresh compiled LangGraph instance with an isolated in-memory checkpointer."""
    from langgraph.checkpoint.memory import MemorySaver

    from agent.graph import create_support_graph

    return create_support_graph(MemorySaver())


TRIAGE_JSON = (
    '{"intent": "faq", "sub_intent": null, "sentiment": "neutral", '
    '"urgency": "medium", "escalation_required": false, '
    '"escalation_reason": null, "confidence": 0.95}'
)
ESCALATE_JSON = (
    '{"intent": "complaint", "sub_intent": null, "sentiment": "very_frustrated", '
    '"urgency": "critical", "escalation_required": true, '
    '"escalation_reason": "Furious customer", "confidence": 0.9}'
)


@pytest.fixture
def mock_llm(monkeypatch):
    """Patch ``agent.nodes`` LLM factory with a scriptable FakeLLM.

    Usage: ``mock_llm.responses["TICKET_CREATE"] = '{...}'``
    """
    fake = FakeLLM(default=TRIAGE_JSON)
    monkeypatch.setattr("agent.nodes._OPENAI_AVAILABLE", True)
    monkeypatch.setattr("agent.nodes._get_llm", lambda **_: fake)
    return fake


# ---------------------------------------------------------------------------
# Fake vector store for RAG tests
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_vectorstore(monkeypatch):
    """Patch ``rag.vectorstore.FAISSVectorStore`` with a deterministic in-memory store."""
    from langchain_core.documents import Document

    class FakeStore:
        def __init__(self, *a, **k):
            self.docs = [
                (Document(page_content="Returns are accepted within 30 days of delivery with original packaging.",
                          metadata={"source": "returns.md", "doc_type": "policy", "category": "returns"}), 0.92),
                (Document(page_content="Standard shipping takes 3-5 business days.",
                          metadata={"source": "shipping.md", "doc_type": "faq", "category": "shipping"}), 0.85),
                (Document(page_content="The keyboard supports Bluetooth 5.2 and USB-C.",
                          metadata={"source": "specs.md", "doc_type": "product", "category": "specs"}), 0.40),
            ]

        def similarity_search_with_score(self, query, k=6, filter=None):
            return self.docs[:k]

        def get_stats(self):
            return {"n_docs": len(self.docs)}

        def as_retriever(self, search_kwargs=None):
            return self

    monkeypatch.setattr("rag.vectorstore.FAISSVectorStore", FakeStore)
    return FakeStore

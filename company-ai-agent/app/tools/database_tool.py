"""Database Tool providing safe querying of company relational database."""

import logging
import re
from typing import Any, Dict, List, Optional

from sqlalchemy import or_, select, text

from app.config import Settings
from app.models import EmployeeRecord, PolicyRecord, TicketRecord, async_session_factory
from app.tools import BaseTool

logger = logging.getLogger("company_agent.tools.database")


class DatabaseTool(BaseTool):
    """Tool enabling the AI Agent to inspect employees, tickets, and corporate policies."""

    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def name(self) -> str:
        return "database_tool"

    @property
    def description(self) -> str:
        return (
            "Safely query Acme Corp's enterprise database. Supports searching employees, "
            "inspecting IT support tickets, retrieving official company policies, and running "
            "read-only SQL queries."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "search_employees",
                        "search_tickets",
                        "get_policy",
                        "execute_read_query",
                    ],
                    "description": "The specific database operation to perform.",
                },
                "query": {
                    "type": "string",
                    "description": "General search keyword (name, ticket description, or policy topic).",
                },
                "department": {
                    "type": "string",
                    "description": "Department filter for employee searches (e.g. 'Engineering', 'HR').",
                },
                "status": {
                    "type": "string",
                    "description": "Status filter for tickets (e.g. 'Open', 'In Progress', 'Resolved').",
                },
                "priority": {
                    "type": "string",
                    "description": "Priority filter for tickets (e.g. 'Critical', 'High', 'Medium', 'Low').",
                },
                "policy_category": {
                    "type": "string",
                    "description": "Category for policies ('HR', 'Security', 'Travel', 'IT').",
                },
                "sql_query": {
                    "type": "string",
                    "description": "Raw SELECT SQL query if custom aggregation or joining is needed (read-only).",
                },
            },
            "required": ["action"],
        }

    async def execute(self, arguments: Dict[str, Any]) -> Any:
        action = arguments.get("action")

        if action == "search_employees":
            return await self._search_employees(
                query=arguments.get("query"),
                department=arguments.get("department"),
            )
        elif action == "search_tickets":
            return await self._search_tickets(
                query=arguments.get("query"),
                status=arguments.get("status"),
                priority=arguments.get("priority"),
            )
        elif action == "get_policy":
            return await self._get_policy(
                category=arguments.get("policy_category"),
                query=arguments.get("query"),
            )
        elif action == "execute_read_query":
            return await self._execute_read_query(sql_query=arguments.get("sql_query", ""))
        else:
            return {"error": f"Unknown database action: '{action}'."}

    async def _search_employees(
        self, query: Optional[str] = None, department: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        async with async_session_factory() as session:
            stmt = select(EmployeeRecord)
            if department:
                stmt = stmt.where(EmployeeRecord.department.ilike(f"%{department}%"))
            if query:
                stmt = stmt.where(
                    or_(
                        EmployeeRecord.name.ilike(f"%{query}%"),
                        EmployeeRecord.role.ilike(f"%{query}%"),
                        EmployeeRecord.employee_id.ilike(f"%{query}%"),
                        EmployeeRecord.email.ilike(f"%{query}%"),
                    )
                )
            result = await session.execute(stmt)
            employees = result.scalars().all()
            return [
                {
                    "employee_id": emp.employee_id,
                    "name": emp.name,
                    "department": emp.department,
                    "role": emp.role,
                    "email": emp.email,
                    "status": emp.status,
                }
                for emp in employees
            ]

    async def _search_tickets(
        self,
        query: Optional[str] = None,
        status: Optional[str] = None,
        priority: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        async with async_session_factory() as session:
            stmt = select(TicketRecord)
            if status:
                stmt = stmt.where(TicketRecord.status.ilike(f"%{status}%"))
            if priority:
                stmt = stmt.where(TicketRecord.priority.ilike(f"%{priority}%"))
            if query:
                stmt = stmt.where(
                    or_(
                        TicketRecord.ticket_code.ilike(f"%{query}%"),
                        TicketRecord.title.ilike(f"%{query}%"),
                        TicketRecord.description.ilike(f"%{query}%"),
                        TicketRecord.assigned_to.ilike(f"%{query}%"),
                    )
                )
            result = await session.execute(stmt)
            tickets = result.scalars().all()
            return [
                {
                    "ticket_code": t.ticket_code,
                    "title": t.title,
                    "description": t.description,
                    "status": t.status,
                    "priority": t.priority,
                    "assigned_to": t.assigned_to,
                    "created_at": t.created_at.isoformat() if t.created_at else None,
                }
                for t in tickets
            ]

    async def _get_policy(
        self, category: Optional[str] = None, query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        async with async_session_factory() as session:
            stmt = select(PolicyRecord)
            if category:
                stmt = stmt.where(PolicyRecord.category.ilike(f"%{category}%"))
            if query:
                stmt = stmt.where(
                    or_(
                        PolicyRecord.title.ilike(f"%{query}%"),
                        PolicyRecord.content.ilike(f"%{query}%"),
                        PolicyRecord.policy_code.ilike(f"%{query}%"),
                    )
                )
            result = await session.execute(stmt)
            policies = result.scalars().all()
            return [
                {
                    "policy_code": p.policy_code,
                    "category": p.category,
                    "title": p.title,
                    "content": p.content,
                    "effective_date": p.effective_date,
                }
                for p in policies
            ]

    async def _execute_read_query(self, sql_query: str) -> Dict[str, Any]:
        """Safely executes custom SQL SELECT queries with strict security verification."""
        clean_sql = sql_query.strip()
        if not clean_sql:
            return {"error": "SQL query cannot be empty."}

        # Security check: Read-only verification
        forbidden_patterns = [
            r"\bDROP\b",
            r"\bDELETE\b",
            r"\bINSERT\b",
            r"\bUPDATE\b",
            r"\bALTER\b",
            r"\bTRUNCATE\b",
            r"\bCREATE\b",
            r"\bGRANT\b",
            r"\bREVOKE\b",
            r"\bEXEC\b",
            r"\bEXECUTE\b",
        ]
        for pattern in forbidden_patterns:
            if re.search(pattern, clean_sql, re.IGNORECASE):
                logger.warning(f"Forbidden SQL statement detected: {clean_sql}")
                return {
                    "error": "Security Violation: Only read-only SELECT queries are allowed."
                }

        if not re.match(r"^(SELECT|WITH)\b", clean_sql, re.IGNORECASE):
            return {"error": "Query must start with SELECT or WITH statement."}

        async with async_session_factory() as session:
            try:
                result = await session.execute(text(clean_sql))
                rows = result.fetchall()
                keys = list(result.keys())
                return {
                    "columns": keys,
                    "rows": [dict(zip(keys, row)) for row in rows],
                    "count": len(rows),
                }
            except Exception as e:
                logger.warning(f"Failed to execute read query: {e}")
                return {"error": f"SQL execution error: {str(e)}"}


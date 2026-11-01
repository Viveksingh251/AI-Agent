"""Conversation Memory Service managing session history and persistence."""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import delete, func, select

from app.config import Settings, get_settings
from app.models import MessageRecord, SessionRecord, async_session_factory
from app.schemas import MessageItem, SessionSummary

logger = logging.getLogger("company_agent.services.memory")


class MemoryService:
    """Manages persistent turn-by-turn conversation memory for multi-user sessions."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.max_messages = settings.MAX_HISTORY_MESSAGES

    async def get_history(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[MessageItem]:
        """Retrieves chronological messages for an active session."""
        max_limit = limit or self.max_messages
        async with async_session_factory() as session:
            stmt = (
                select(MessageRecord)
                .where(MessageRecord.session_id == session_id)
                .order_by(MessageRecord.id.desc())
                .limit(max_limit)
            )
            result = await session.execute(stmt)
            messages = result.scalars().all()

            # Reverse to chronological order
            chronological = list(reversed(messages))
            return [
                MessageItem(
                    id=m.id,
                    role=m.role,
                    content=m.content,
                    tool_calls_json=m.tool_calls_json,
                    timestamp=m.timestamp,
                )
                for m in chronological
            ]

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: Optional[str] = None,
        tool_calls_json: Optional[str] = None,
    ) -> MessageItem:
        """Appends a new turn message to the session."""
        async with async_session_factory() as session:
            # Ensure parent session record exists
            result = await session.execute(
                select(SessionRecord).where(SessionRecord.id == session_id)
            )
            sess = result.scalar_one_or_none()

            now = datetime.now(timezone.utc)
            if not sess:
                sess = SessionRecord(
                    id=session_id,
                    title=(content[:60] + "...") if content else "Session",
                    created_at=now,
                    updated_at=now,
                )
                session.add(sess)
            else:
                sess.updated_at = now

            message = MessageRecord(
                session_id=session_id,
                role=role,
                content=content,
                tool_calls_json=tool_calls_json,
                timestamp=now,
            )
            session.add(message)
            await session.commit()
            await session.refresh(message)

            return MessageItem(
                id=message.id,
                role=message.role,
                content=message.content,
                tool_calls_json=message.tool_calls_json,
                timestamp=message.timestamp,
            )

    async def list_sessions(self, limit: int = 50) -> List[SessionSummary]:
        """Lists recent conversation sessions with message counts."""
        async with async_session_factory() as session:
            stmt = (
                select(
                    SessionRecord,
                    func.count(MessageRecord.id).label("message_count"),
                )
                .outerjoin(MessageRecord, SessionRecord.id == MessageRecord.session_id)
                .group_by(SessionRecord.id)
                .order_by(SessionRecord.updated_at.desc())
                .limit(limit)
            )
            results = await session.execute(stmt)
            rows = results.all()

            summaries = []
            for sess_record, count in rows:
                summaries.append(
                    SessionSummary(
                        session_id=sess_record.id,
                        title=sess_record.title,
                        message_count=count,
                        created_at=sess_record.created_at,
                        updated_at=sess_record.updated_at,
                    )
                )
            return summaries

    async def clear_session(self, session_id: str) -> int:
        """Removes all stored messages for a specific session."""
        async with async_session_factory() as session:
            stmt = delete(MessageRecord).where(MessageRecord.session_id == session_id)
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount

    async def ping(self) -> bool:
        """Validates database connectivity."""
        try:
            async with async_session_factory() as session:
                await session.execute(select(1))
            return True
        except Exception as exc:
            logger.error(f"Database ping failed: {exc}")
            return False


def get_memory_service(settings: Optional[Settings] = None) -> MemoryService:
    """Factory provider for MemoryService."""
    _settings = settings or get_settings()
    return MemoryService(settings=_settings)


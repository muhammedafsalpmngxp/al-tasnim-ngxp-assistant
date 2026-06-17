import time
import uuid
import logging
from typing import Dict, Optional

from .models import ConversationState, Intent

logger = logging.getLogger(__name__)

# Sessions inactive longer than this (seconds) are cleaned up automatically
_SESSION_TTL_SEC = 3600  # 1 hour


class ConversationManager:
    def __init__(self, max_attempts: int = 3, session_ttl: int = _SESSION_TTL_SEC):
        self.max_attempts = max_attempts
        self._session_ttl = session_ttl
        self._sessions: Dict[str, ConversationState] = {}
        self._last_active: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        self._cleanup_expired()
        if session_id and session_id in self._sessions:
            self._touch(session_id)
            return session_id
        new_id = session_id or str(uuid.uuid4())
        self._sessions[new_id] = ConversationState(session_id=new_id)
        self._touch(new_id)
        return new_id

    def get_session(self, session_id: str) -> Optional[ConversationState]:
        session = self._sessions.get(session_id)
        if session:
            self._touch(session_id)
        return session

    def reset_session(self, session_id: str) -> None:
        """Fully remove a session (called by /reset endpoint)."""
        self._sessions.pop(session_id, None)
        self._last_active.pop(session_id, None)

    # ------------------------------------------------------------------
    # Query history
    # ------------------------------------------------------------------

    def add_query(self, session_id: str, query: str) -> None:
        session = self.get_session(session_id)
        if session is None:
            session_id = self.get_or_create_session(session_id)
            session = self.get_session(session_id)
        session.query_history.append(query)
        if len(session.query_history) > 10:
            session.query_history.pop(0)

    # ------------------------------------------------------------------
    # Clarification state
    # ------------------------------------------------------------------

    def needs_clarification(self, session_id: str) -> bool:
        session = self.get_session(session_id)
        return bool(session and session.pending_clarification)

    def set_clarification(
        self,
        session_id: str,
        question: str,
        original_query: str,
        partial_intent: Optional[Intent] = None,
    ) -> None:
        session = self.get_session(session_id)
        if session:
            session.pending_clarification = True
            session.clarification_asked = question
            session.original_query = original_query
            session.previous_intent = partial_intent
            session.attempts += 1

    def get_clarification_context(self, session_id: str) -> Optional[str]:
        session = self.get_session(session_id)
        if session and session.clarification_asked:
            return (
                f"Original question: {session.original_query}\n"
                f"Assistant asked: {session.clarification_asked}\n"
                f"User's clarification: "
            )
        return None

    def clear_clarification(self, session_id: str) -> None:
        session = self.get_session(session_id)
        if session:
            session.pending_clarification = False
            session.clarification_asked = None
            session.original_query = None
            session.attempts = 0

    def can_continue(self, session_id: str) -> bool:
        session = self.get_session(session_id)
        return session.attempts < self.max_attempts if session else True

    def get_remaining_attempts(self, session_id: str) -> int:
        session = self.get_session(session_id)
        return (self.max_attempts - session.attempts) if session else self.max_attempts

    # ------------------------------------------------------------------
    # TTL cleanup — called automatically on every session access
    # ------------------------------------------------------------------

    def _touch(self, session_id: str) -> None:
        self._last_active[session_id] = time.time()

    def _cleanup_expired(self) -> None:
        now = time.time()
        expired = [
            sid
            for sid, last in self._last_active.items()
            if now - last > self._session_ttl
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._last_active.pop(sid, None)
        if expired:
            logger.debug("Cleaned up %d expired sessions", len(expired))

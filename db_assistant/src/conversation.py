from typing import Dict, Optional
from .models import ConversationState, Intent
import uuid

class ConversationManager:
    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts
        self._sessions: Dict[str, ConversationState] = {}
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        if session_id and session_id in self._sessions:
            return session_id
        new_id = session_id or str(uuid.uuid4())
        self._sessions[new_id] = ConversationState(session_id=new_id)
        return new_id
    
    def get_session(self, session_id: str) -> Optional[ConversationState]:
        return self._sessions.get(session_id)
    
    def add_query(self, session_id: str, query: str):
        session = self.get_session(session_id)
        if session:
            session.query_history.append(query)
            if len(session.query_history) > 10:
                session.query_history.pop(0)
    
    def needs_clarification(self, session_id: str) -> bool:
        session = self.get_session(session_id)
        return session and session.pending_clarification
    
    def set_clarification(self, session_id: str, question: str, original_query: str, partial_intent: Optional[Intent] = None):
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
            context = f"Original question: {session.original_query}\n"
            context += f"Assistant asked: {session.clarification_asked}\n"
            context += f"User's clarification: "
            return context
        return None
    
    def clear_clarification(self, session_id: str):
        session = self.get_session(session_id)
        if session:
            session.pending_clarification = False
            session.clarification_asked = None
            session.original_query = None
            session.attempts = 0
    
    def can_continue(self, session_id: str) -> bool:
        session = self.get_session(session_id)
        if session:
            return session.attempts < self.max_attempts
        return True
    
    def get_remaining_attempts(self, session_id: str) -> int:
        session = self.get_session(session_id)
        if session:
            return self.max_attempts - session.attempts
        return self.max_attempts

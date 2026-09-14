"""Session manager for multi-turn conversational context and audit persistence.
"""
from typing import Dict, List, Optional
import json
from pathlib import Path
from app.core.config import SESSIONS_DIR
from app.core.models import ChatSessionState, ChatTurn, QueryFilter


class SessionManager:
    """Manages chat session lifecycle and persistence."""

    def __init__(self, sessions_dir: Path = SESSIONS_DIR):
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_path(self, session_id: str) -> Path:
        safe_id = "".join(c for c in session_id if c.isalnum() or c in ("-", "_"))
        return self.sessions_dir / f"{safe_id}.json"

    def get_or_create_session(self, session_id: str) -> ChatSessionState:
        path = self._get_session_path(session_id)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return ChatSessionState(**data)
            except Exception:
                pass
        # Create fresh session
        session = ChatSessionState(session_id=session_id)
        self.save_session(session)
        return session

    def save_session(self, session: ChatSessionState) -> None:
        path = self._get_session_path(session.session_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(session.model_dump(), f, indent=2)

    def append_turn(
        self,
        session_id: str,
        turn: ChatTurn,
        new_active_dataset: Optional[str] = None,
        new_filters: Optional[List[QueryFilter]] = None,
    ) -> ChatSessionState:
        session = self.get_or_create_session(session_id)
        session.history.append(turn)

        if new_active_dataset:
            session.active_dataset = new_active_dataset

        if new_filters is not None:
            session.accumulated_filters = new_filters

        self.save_session(session)
        return session

    def clear_session(self, session_id: str) -> None:
        path = self._get_session_path(session_id)
        if path.exists():
            path.unlink()

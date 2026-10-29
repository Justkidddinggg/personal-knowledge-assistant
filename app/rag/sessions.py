"""In-memory conversation sessions (lost on restart; single process only)."""
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Turn:
    question: str
    answer: str


@dataclass
class _Session:
    scope: frozenset[str] | None  # which documents this conversation is about
    turns: list[Turn] = field(default_factory=list)


class SessionStore:
    def __init__(self, max_turns: int = 4, max_sessions: int = 100) -> None:
        self._max_turns = max_turns
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, _Session] = OrderedDict()
        self._lock = threading.Lock()  # FastAPI runs sync endpoints in a thread pool

    def start_or_continue(
        self, session_id: str | None, doc_ids: list[str] | None
    ) -> tuple[str, list[Turn]]:
        """Return (session_id, previous turns). Unknown ids start a fresh session."""
        scope = None if doc_ids is None else frozenset(doc_ids)
        with self._lock:
            if session_id is not None and session_id in self._sessions:
                sid = session_id
                session = self._sessions[sid]
                if session.scope != scope:  # different documents: old context is irrelevant
                    session.scope = scope
                    session.turns.clear()
            else:
                sid = uuid.uuid4().hex
                session = _Session(scope)
                self._sessions[sid] = session
                while len(self._sessions) > self._max_sessions:
                    self._sessions.popitem(last=False)  # drop the least recently used
            self._sessions.move_to_end(sid)
            return sid, list(session.turns)

    def add_turn(self, session_id: str, turn: Turn) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:  # evicted while the answer was being generated
                return
            session.turns.append(turn)
            del session.turns[: -self._max_turns]  # keep only the most recent turns
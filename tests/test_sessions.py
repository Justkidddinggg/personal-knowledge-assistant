from app.rag.sessions import SessionStore, Turn


def test_new_session_has_id_and_empty_history() -> None:
    sid, turns = SessionStore().start_or_continue(None, None)
    assert isinstance(sid, str) and sid
    assert turns == []


def test_turns_are_trimmed_to_max() -> None:
    store = SessionStore(max_turns=2)
    sid, _ = store.start_or_continue(None, None)
    for i in (1, 2, 3):
        store.add_turn(sid, Turn(f"q{i}", f"a{i}"))
    _, turns = store.start_or_continue(sid, None)
    assert [t.question for t in turns] == ["q2", "q3"]


def test_least_recently_used_session_is_evicted() -> None:
    store = SessionStore(max_sessions=2)
    s1, _ = store.start_or_continue(None, None)
    s2, _ = store.start_or_continue(None, None)
    s3, _ = store.start_or_continue(None, None)
    assert store.start_or_continue(s3, None)[0] == s3  # still known
    assert store.start_or_continue(s1, None)[0] != s1  # evicted -> fresh session


def test_changing_document_scope_clears_history() -> None:
    store = SessionStore()
    sid, _ = store.start_or_continue(None, ["d1"])
    store.add_turn(sid, Turn("q", "a"))
    _, turns = store.start_or_continue(sid, ["d2"])
    assert turns == []


def test_same_scope_in_different_order_keeps_history() -> None:
    store = SessionStore()
    sid, _ = store.start_or_continue(None, ["d1", "d2"])
    store.add_turn(sid, Turn("q", "a"))
    _, turns = store.start_or_continue(sid, ["d2", "d1"])
    assert len(turns) == 1
"""Tests del TTL y la limpieza de sesiones inactivas."""

from __future__ import annotations

import time

from app.sessions import session_store


def test_evict_inactive_borra_sesiones_antiguas() -> None:
    session_store.reset()
    s1 = session_store.create()
    s2 = session_store.create()
    s1.last_activity_at = time.time() - 10_000
    s2.last_activity_at = time.time()

    removed = session_store.evict_inactive(ttl_seconds=3600)
    assert removed == 1
    assert session_store.get(s2.session_id) is not None
    assert session_store.get(s1.session_id) is None


def test_evict_inactive_no_borra_cuando_ttl_largo() -> None:
    session_store.reset()
    s = session_store.create()
    s.last_activity_at = time.time() - 60

    removed = session_store.evict_inactive(ttl_seconds=3600)
    assert removed == 0
    assert session_store.get(s.session_id) is not None


def test_touch_actualiza_last_activity() -> None:
    session_store.reset()
    s = session_store.create()
    s.last_activity_at = 0.0
    fetched = session_store.get(s.session_id)
    assert fetched is not None
    assert fetched.last_activity_at > 0

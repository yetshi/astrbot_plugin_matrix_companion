"""Persona-scoped JSON state store.

Each persona owns one file, so swapping the Matrix account or the platform
instance never touches the simulation itself. This module has no AstrBot
imports and is unit-testable on its own.
"""

from __future__ import annotations

import json
import os
import threading
import unicodedata
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
AUDIT_LIMIT = 200


def safe_stem(persona_id: str) -> str:
    """Turn a persona id into a filesystem-safe, reversible stem.

    Args:
        persona_id: Raw persona identifier.

    Returns:
        A stem that percent-encodes unsafe characters.
    """
    text = unicodedata.normalize("NFC", str(persona_id or "")).strip()
    out: list[str] = []
    for character in text:
        if character in '\\/:*?"<>|' or unicodedata.category(character).startswith("C"):
            out.extend(f"%{byte:02X}" for byte in character.encode("utf-8"))
        else:
            out.append(character)
    return "".join(out)[:120] or "default"


def empty_state(persona_id: str) -> dict[str, Any]:
    """Return a fresh state document for one persona.

    Args:
        persona_id: Owner persona id.

    Returns:
        A new state mapping.
    """
    return {
        "schema": SCHEMA_VERSION,
        "persona_id": persona_id,
        "life": {
            "date": "",
            "mood": "",
            "energy": 70,
            "status": "",
            "plan": [],
            "generated_at": 0.0,
            "sleeping": False,
            "segment_key": "",
            # Conversation feedback: pending lines wait for the next
            # reconciliation pass, log keeps what actually happened.
            "pending_chat": [],
            "log": [],
            "reconciled_at": 0.0,
            # A reconciled status stays visible for a while before the plan
            # takes over again.
            "manual_status": "",
            "manual_until": 0.0,
        },
        "proactive": {
            "day": "",
            "sent_today": 0,
            "last_sent_at": 0.0,
            "last_text": "",
            "unanswered": 0,
        },
        "inbound": {"last_at": 0.0, "last_text": ""},
        # What the user said about their own state (sleeping / away / back).
        "user": {
            "sleeping": False,
            "sleeping_since": 0.0,
            "away": "",
            "away_since": 0.0,
            "last_text": "",
            "updated_at": 0.0,
        },
        "audit": [],
    }


def _deep_merge(defaults: dict[str, Any], loaded: Any) -> dict[str, Any]:
    """Merge stored values over defaults without dropping new keys."""
    if not isinstance(loaded, dict):
        return defaults
    result = deepcopy(defaults)
    for key, value in loaded.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def append_audit(state: dict[str, Any], entry: dict[str, Any]) -> None:
    """Append one audit record, trimming the oldest entries."""
    audit = state.get("audit")
    if not isinstance(audit, list):
        audit = []
        state["audit"] = audit
    audit.append(entry)
    if len(audit) > AUDIT_LIMIT:
        del audit[: len(audit) - AUDIT_LIMIT]


def _append_capped(state: dict[str, Any], key: str, entry: Any, limit: int) -> list:
    items = state.get(key)
    if not isinstance(items, list):
        items = []
        state[key] = items
    items.append(entry)
    if len(items) > limit:
        del items[: len(items) - limit]
    return items


def append_pending_chat(
    life: dict[str, Any], text: str, at: float, limit: int = 20
) -> None:
    """Queue one inbound line for the next life reconciliation.

    Args:
        life: The persona's ``life`` mapping.
        text: Inbound message text.
        at: Unix timestamp of the message.
        limit: Maximum queued lines.
    """
    body = str(text or "").strip()[:400]
    if not body:
        return
    _append_capped(life, "pending_chat", {"at": at, "text": body}, limit)


def append_life_log(
    life: dict[str, Any], text: str, at: float, limit: int = 20
) -> None:
    """Record one thing that actually happened to the persona.

    Args:
        life: The persona's ``life`` mapping.
        text: Short event description.
        at: Unix timestamp.
        limit: Maximum retained entries.
    """
    body = str(text or "").strip()[:200]
    if not body:
        return
    _append_capped(life, "log", {"at": at, "text": body}, limit)


class PersonaStore:
    """Thread-safe JSON store keyed by persona id."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._lock = threading.RLock()
        self._cache: dict[str, dict[str, Any]] = {}

    def path_for(self, persona_id: str) -> Path:
        """Return the state file path for a persona."""
        return self.root / f"{safe_stem(persona_id)}.json"

    def load(self, persona_id: str) -> dict[str, Any]:
        """Load (and cache) one persona state."""
        with self._lock:
            cached = self._cache.get(persona_id)
            if cached is not None:
                return cached
            state = empty_state(persona_id)
            path = self.path_for(persona_id)
            if path.is_file():
                try:
                    loaded = json.loads(path.read_text(encoding="utf-8"))
                    state = _deep_merge(state, loaded)
                except Exception:
                    state = empty_state(persona_id)
                    state["load_error"] = True
            state["persona_id"] = persona_id
            self._cache[persona_id] = state
            return state

    def save(self, persona_id: str, state: dict[str, Any] | None = None) -> None:
        """Persist one persona state atomically.

        Args:
            persona_id: Owner persona id.
            state: State to persist; defaults to the cached document.

        Raises:
            OSError: If the file cannot be written.
        """
        with self._lock:
            payload = state if state is not None else self._cache.get(persona_id)
            if payload is None:
                return
            self.root.mkdir(parents=True, exist_ok=True)
            path = self.path_for(persona_id)
            temporary = path.with_suffix(".json.tmp")
            data = json.dumps(payload, ensure_ascii=False, indent=2)
            temporary.write_text(data, encoding="utf-8")
            os.replace(temporary, path)
            self._cache[persona_id] = payload

    def cached(self) -> dict[str, dict[str, Any]]:
        """Return a shallow copy of the cached persona ids."""
        with self._lock:
            return dict(self._cache)


def ensure_proactive_day(state: dict[str, Any], day_key: str) -> None:
    """Reset the daily proactive counters when the calendar day changes."""
    proactive = state.setdefault("proactive", {})
    if proactive.get("day") != day_key:
        proactive["day"] = day_key
        proactive["sent_today"] = 0


def now_iso(now: datetime) -> str:
    """Format a timestamp for audit records."""
    return now.strftime("%Y-%m-%dT%H:%M:%S")

"""User-side state cues extracted from inbound messages.

The persona has its own simulated life, but it must still react to what the
user actually says: going to sleep, heading out, coming back. The rules are
deliberately conservative keyword matches so the behaviour is predictable and
costs no model call.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

SLEEP_CUES = (
    "去睡了",
    "睡觉了",
    "我睡了",
    "要睡了",
    "准备睡",
    "晚安",
    "困了",
    "去睡",
    "睡觉去",
    "熄灯",
    "躺下了",
)
WAKE_CUES = ("醒了", "睡醒", "起床", "早安", "早上好")
AWAY_CUES = (
    "出门",
    "上班",
    "上课",
    "开会",
    "在忙",
    "忙了",
    "吃饭",
    "洗澡",
    "开车",
    "在路上",
    "先不聊",
    "下了",
)
BACK_CUES = ("回来了", "到家", "下班", "吃完了", "开完会", "忙完了", "我回来了")
# "睡不着" and friends read as "still awake", so the sleep cue must not fire.
SLEEP_NEGATIONS = ("睡不着", "不想睡", "还没睡", "不睡", "睡不了")

SLEEPING_TIMEOUT_HOURS = 12


@dataclass(frozen=True)
class Cue:
    """One interpreted user cue."""

    kind: str
    text: str


def detect_cue(text: str) -> Cue:
    """Classify one inbound message into a user-state cue.

    Args:
        text: Inbound message text.

    Returns:
        A cue whose ``kind`` is ``sleep``, ``wake``, ``away``, ``back`` or
        ``none``.
    """
    body = str(text or "").strip()
    if not body:
        return Cue("none", "")
    if any(token in body for token in SLEEP_NEGATIONS):
        return Cue("none", body)
    for token in WAKE_CUES:
        if token in body:
            return Cue("wake", body)
    for token in SLEEP_CUES:
        if token in body:
            return Cue("sleep", body)
    for token in BACK_CUES:
        if token in body:
            return Cue("back", body)
    for token in AWAY_CUES:
        if token in body:
            return Cue("away", body)
    return Cue("none", body)


def apply_cue(user: dict[str, Any], cue: Cue, text: str, now: datetime) -> None:
    """Update the stored user state from one cue.

    Any ordinary message also means the user is around again, so a plain
    message clears both ``sleeping`` and ``away``.

    Args:
        user: The persona's ``user`` mapping, mutated in place.
        cue: Result of :func:`detect_cue`.
        text: Raw message text.
        now: Current local time.
    """
    stamp = now.timestamp()
    body = str(text or "").strip()[:200]
    if cue.kind == "sleep":
        user["sleeping"] = True
        user["sleeping_since"] = stamp
    elif cue.kind == "wake":
        user["sleeping"] = False
        user["sleeping_since"] = 0.0
        user["away"] = ""
        user["away_since"] = 0.0
    elif cue.kind == "away":
        user["away"] = body[:60]
        user["away_since"] = stamp
        user["sleeping"] = False
        user["sleeping_since"] = 0.0
    elif cue.kind == "back":
        user["away"] = ""
        user["away_since"] = 0.0
        user["sleeping"] = False
        user["sleeping_since"] = 0.0
    else:
        # Any other message proves the user is awake and reachable.
        user["sleeping"] = False
        user["sleeping_since"] = 0.0
        user["away"] = ""
        user["away_since"] = 0.0
    user["last_text"] = body
    user["updated_at"] = stamp


def is_user_sleeping(
    user: Any, now: datetime, *, timeout_hours: float = SLEEPING_TIMEOUT_HOURS
) -> bool:
    """Return whether the user is still considered asleep.

    Args:
        user: The persona's ``user`` mapping.
        now: Current local time.
        timeout_hours: Safety timeout so a stale flag cannot silence the bot.

    Returns:
        True when the sleep flag is set and still fresh.
    """
    if not isinstance(user, dict) or not user.get("sleeping"):
        return False
    since = float(user.get("sleeping_since") or 0.0)
    if not since:
        return True
    return (now.timestamp() - since) < timeout_hours * 3600


def user_summary(user: Any, now: datetime) -> str:
    """Render a short description of the user's known state.

    Args:
        user: The persona's ``user`` mapping.
        now: Current local time.

    Returns:
        One line, or ``""`` when nothing is known.
    """
    if not isinstance(user, dict):
        return ""
    parts: list[str] = []
    last_text = str(user.get("last_text") or "").strip()
    if last_text:
        parts.append(f"对方最后说：{last_text[:80]}")
    if is_user_sleeping(user, now):
        parts.append("对方说过去睡了，可能还在睡")
    away = str(user.get("away") or "").strip()
    if away:
        parts.append(f"对方可能在外面：{away[:40]}")
    return "；".join(parts)

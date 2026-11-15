"""User-side state cues extracted from inbound messages.

The persona has its own simulated life, but it must still react to what the
user actually says: going to sleep, heading out, coming back. The rules are
deliberately conservative keyword matches so the behaviour is predictable and
costs no model call.
"""

from __future__ import annotations

import re
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
PLACE_FRESH_HOURS = 8

# Longest first so "图书馆" wins over "书".
_PLACE_NOUNS = (
    "健身房",
    "图书馆",
    "咖啡厅",
    "办公室",
    "超市",
    "医院",
    "宿舍",
    "学校",
    "公司",
    "车站",
    "机场",
    "地铁",
    "路上",
    "车上",
    "外面",
    "家里",
    "家",
)
_PLACE_MARKERS = ("在", "到", "回", "去", "出门", "到了")
_PLACE_PATTERN = re.compile(
    r"我在\s*([\u4e00-\u9fa5A-Za-z0-9]{1,10}?)(?:呢|啊|了|，|。|！|$)"
)
_PLACE_STOPWORDS = {
    "你",
    "这",
    "那",
    "想",
    "说",
    "等",
    "忙",
    "睡",
    "看",
    "做",
    "吃",
    "开",
    "上班",
    "上课",
    "开会",
    "洗澡",
    "加班",
    "发呆",
    "生气",
}


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


def detect_place(text: str) -> str | None:
    """Extract a place from an explicit location statement.

    Only statements with a locative marker are accepted, so "我喜欢家" or
    "我在开会" cannot turn into a place.

    Args:
        text: Inbound message text.

    Returns:
        A short place label, or ``None`` when the message carries none.
    """
    body = str(text or "").strip()
    if not body:
        return None
    if "到家" in body or "回家" in body or "回来了" in body:
        return "家"
    if any(marker in body for marker in _PLACE_MARKERS):
        for noun in _PLACE_NOUNS:
            if noun in body:
                return noun
    match = _PLACE_PATTERN.search(body)
    if match:
        candidate = match.group(1).strip()
        if candidate and candidate not in _PLACE_STOPWORDS and len(candidate) >= 2:
            return candidate
    return None


def apply_place(user: dict[str, Any], place: str, now: datetime) -> None:
    """Record the place the user said they are at.

    Args:
        user: The persona's ``user`` mapping, mutated in place.
        place: Short place label.
        now: Current local time.
    """
    value = str(place or "").strip()[:30]
    if not value:
        return
    user["place"] = value
    user["place_updated_at"] = now.timestamp()


def current_place(
    user: Any, now: datetime, *, fresh_hours: float = PLACE_FRESH_HOURS
) -> str:
    """Return the stored place while it is still fresh.

    Args:
        user: The persona's ``user`` mapping.
        now: Current local time.
        fresh_hours: How long a recorded place stays usable.

    Returns:
        The place label, or ``""`` when unknown or stale.
    """
    if not isinstance(user, dict):
        return ""
    place = str(user.get("place") or "").strip()
    if not place:
        return ""
    updated = float(user.get("place_updated_at") or 0.0)
    if updated and (now.timestamp() - updated) > fresh_hours * 3600:
        return ""
    return place


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
    place = current_place(user, now)
    if place:
        parts.append(f"对方可能在：{place}")
    return "；".join(parts)

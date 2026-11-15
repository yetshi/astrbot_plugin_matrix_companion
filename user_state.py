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


PROFILE_SCALAR_FIELDS = frozenset(
    {
        "display_name",
        "address_as",
        "city",
        "timezone",
        "occupation",
        "usual_sleep",
        "notes",
    }
)
PROFILE_LIST_FIELDS = frozenset({"interests", "values", "dislikes"})
PROFILE_DATED_FIELDS = frozenset({"important_dates", "health"})
# Never stored, even when the user says them.
PROFILE_FORBIDDEN = frozenset(
    {
        "politics",
        "political",
        "religion",
        "religious",
        "faith",
        "income",
        "salary",
        "money",
        "finance",
        "wealth",
        "address_detail",
        "home_address",
        "id_card",
        "bank",
        "password",
    }
)
PROFILE_LIST_LIMIT = 20
HEALTH_KEYWORDS = (
    "感冒",
    "发烧",
    "咳嗽",
    "头疼",
    "头痛",
    "胃",
    "疼",
    "难受",
    "不舒服",
    "医院",
    "药",
    "睡不好",
    "失眠",
    "过敏",
    "受伤",
    "月经",
    "姨妈",
    "生理期",
)
HEALTH_TTL_DAYS = 14
HEALTH_FRESH_HOURS = 48


def _ensure_profile(user: dict[str, Any]) -> dict[str, Any]:
    profile = user.get("profile")
    if not isinstance(profile, dict):
        profile = {}
        user["profile"] = profile
    for key in PROFILE_SCALAR_FIELDS:
        profile.setdefault(key, "")
    for key in PROFILE_LIST_FIELDS:
        profile.setdefault(key, [])
    for key in PROFILE_DATED_FIELDS:
        profile.setdefault(key, [])
    profile.setdefault("updated_at", 0.0)
    return profile


def _health_started(entry: dict[str, Any]) -> datetime | None:
    raw = str(entry.get("since") or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def health_note(user: Any, now: datetime) -> str:
    """Describe the user's ongoing health only when it is relevant.

    A steady-state condition older than the fresh window is never injected
    unless the user just brought it up, so the character cannot ask
    "feeling better?" out of nowhere.

    Args:
        user: The persona's ``user`` mapping.
        now: Current local time.

    Returns:
        A prompt line, or ``""`` when nothing relevant is known.
    """
    if not isinstance(user, dict):
        return ""
    profile = user.get("profile") if isinstance(user.get("profile"), dict) else {}
    entries = profile.get("health") if isinstance(profile.get("health"), list) else []
    active: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("status") != "ongoing":
            continue
        started = _health_started(entry)
        if started is not None and (now - started).days > HEALTH_TTL_DAYS:
            continue
        active.append(entry)
    if not active:
        return ""
    last_text = str(user.get("last_text") or "")
    relevant = any(token in last_text for token in HEALTH_KEYWORDS)
    newest = active[-1]
    text = str(newest.get("text") or "").strip()
    since = str(newest.get("since") or "").strip()
    if not text:
        return ""
    if relevant:
        return f"对方最近{text}（{since}），可以顺带问一句好点没有。"
    started = _health_started(newest)
    if (
        started is not None
        and (now - started).total_seconds() <= HEALTH_FRESH_HOURS * 3600
    ):
        return f"对方最近{text}（{since}）。只有话题相关时才关心，不要无缘无故提。"
    return ""


def apply_profile_updates(
    user: dict[str, Any],
    updates: Any,
    now: datetime,
    *,
    list_limit: int = PROFILE_LIST_LIMIT,
) -> dict[str, list]:
    """Validate and apply model-proposed profile facts.

    Args:
        user: The persona's ``user`` mapping, mutated in place.
        updates: List of ``{"key", "value", "source"}`` mappings.
        now: Current local time.
        list_limit: Maximum entries per list field.

    Returns:
        ``{"applied": [key, ...], "rejected": [(key, reason), ...]}``.
    """
    profile = _ensure_profile(user)
    applied: list[str] = []
    rejected: list[tuple[str, str]] = []
    for item in updates or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip().lower()
        value = str(item.get("value") or "").strip()
        source = str(item.get("source") or "").strip()
        if not key:
            continue
        if key in PROFILE_FORBIDDEN:
            rejected.append((key, "forbidden"))
            continue
        if not value:
            rejected.append((key, "empty"))
            continue
        if not source:
            rejected.append((key, "no_source"))
            continue

        if key in PROFILE_SCALAR_FIELDS:
            profile[key] = value[:80]
        elif key in PROFILE_LIST_FIELDS:
            items = profile.get(key) if isinstance(profile.get(key), list) else []
            trimmed = value[:60]
            if trimmed not in items:
                items.append(trimmed)
                profile[key] = items[-list_limit:]
        elif key in PROFILE_DATED_FIELDS:
            entries = profile.get(key) if isinstance(profile.get(key), list) else []
            trimmed = value[:60]
            existing = next(
                (
                    entry
                    for entry in entries
                    if isinstance(entry, dict) and entry.get("text") == trimmed
                ),
                None,
            )
            if key == "health":
                if existing is None:
                    entries.append(
                        {
                            "text": trimmed,
                            "since": now.date().isoformat(),
                            "status": "ongoing",
                            "source": source[:120],
                        }
                    )
                else:
                    existing["status"] = "ongoing"
                    existing["source"] = source[:120]
            elif existing is None:
                entries.append(
                    {
                        "text": trimmed,
                        "source": source[:120],
                        "added_at": now.timestamp(),
                    }
                )
            profile[key] = entries[-list_limit:]
        else:
            rejected.append((key, "unknown_key"))
            continue
        applied.append(key)

    if applied:
        profile["updated_at"] = now.timestamp()
    return {"applied": applied, "rejected": rejected}


def profile_view(user: Any) -> dict[str, Any]:
    """Return a read-only profile view for the panel.

    Args:
        user: The persona's ``user`` mapping.

    Returns:
        A JSON-serializable view; ``has_content`` is False when nothing was learned.
    """
    if not isinstance(user, dict):
        return {"has_content": False}
    profile = user.get("profile") if isinstance(user.get("profile"), dict) else {}
    health = [
        {
            "text": str(entry.get("text") or ""),
            "since": str(entry.get("since") or ""),
            "status": str(entry.get("status") or ""),
        }
        for entry in (profile.get("health") or [])
        if isinstance(entry, dict) and entry.get("text")
    ]
    view = {
        "display_name": str(profile.get("display_name") or ""),
        "address_as": str(profile.get("address_as") or ""),
        "city": str(profile.get("city") or ""),
        "timezone": str(profile.get("timezone") or ""),
        "occupation": str(profile.get("occupation") or ""),
        "usual_sleep": str(profile.get("usual_sleep") or ""),
        "notes": str(profile.get("notes") or ""),
        "interests": [str(item) for item in (profile.get("interests") or [])],
        "values": [str(item) for item in (profile.get("values") or [])],
        "dislikes": [str(item) for item in (profile.get("dislikes") or [])],
        "important_dates": [
            str(item.get("text") or "")
            for item in (profile.get("important_dates") or [])
            if isinstance(item, dict) and item.get("text")
        ],
        "health": health,
        "updated_at": float(profile.get("updated_at") or 0.0),
    }
    scalars = (
        view["display_name"],
        view["address_as"],
        view["city"],
        view["timezone"],
        view["occupation"],
        view["usual_sleep"],
        view["notes"],
    )
    has_content = any(scalars) or any(
        view[key]
        for key in (
            "interests",
            "values",
            "dislikes",
            "important_dates",
            "health",
        )
    )
    view["has_content"] = bool(has_content)
    return view


def clear_profile(user: dict[str, Any]) -> bool:
    """Drop every learned profile fact, keeping transient session state.

    Args:
        user: The persona's ``user`` mapping, mutated in place.

    Returns:
        True when there was something to clear.
    """
    if not isinstance(user, dict):
        return False
    profile = user.get("profile")
    had_content = False
    if isinstance(profile, dict):
        had_content = bool(
            any(str(profile.get(key) or "").strip() for key in PROFILE_SCALAR_FIELDS)
            or any(profile.get(key) for key in PROFILE_LIST_FIELDS)
            or any(profile.get(key) for key in PROFILE_DATED_FIELDS)
        )
    else:
        profile = {}
        user["profile"] = profile
    for key in PROFILE_SCALAR_FIELDS:
        profile[key] = ""
    for key in PROFILE_LIST_FIELDS | PROFILE_DATED_FIELDS:
        profile[key] = []
    profile["updated_at"] = 0.0
    user["observations"] = []
    return had_content


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

    profile = user.get("profile") if isinstance(user.get("profile"), dict) else {}
    name = str(profile.get("address_as") or profile.get("display_name") or "").strip()
    if name:
        parts.append(f"对方希望你称他：{name[:20]}")
    facts: list[str] = []
    for key, label in (
        ("city", "城市"),
        ("usual_sleep", "作息"),
        ("occupation", "职业"),
    ):
        value = str(profile.get(key) or "").strip()
        if value:
            facts.append(f"{label}：{value[:30]}")
    if facts:
        parts.append("；".join(facts))
    values = [
        str(item).strip()[:40]
        for item in (profile.get("values") or [])
        if str(item).strip()
    ][:2]
    if values:
        parts.append("在意：" + "、".join(values))
    dislikes = [
        str(item).strip()[:40]
        for item in (profile.get("dislikes") or [])
        if str(item).strip()
    ][:2]
    if dislikes:
        parts.append("不喜欢：" + "、".join(dislikes))
    health = health_note(user, now)
    if health:
        parts.append(health.rstrip("。"))

    if is_user_sleeping(user, now):
        parts.append("对方说过去睡了，可能还在睡")
    away = str(user.get("away") or "").strip()
    if away:
        parts.append(f"对方可能在外面：{away[:40]}")
    place = current_place(user, now)
    if place:
        parts.append(f"对方可能在：{place}")
    return "；".join(parts)

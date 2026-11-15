"""Life simulation: daily state, schedule and time progression.

The module keeps no AstrBot imports. Callers inject an async LLM function with
the signature ``await llm(system_prompt, user_prompt) -> str``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from .store import append_life_log

LlmCall = Callable[[str, str], Awaitable[str]]

_WEEKDAYS = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")
_HHMM = re.compile(r"^(\d{1,2}):(\d{2})$")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

LIFE_SYSTEM_RULES = """你正在过自己的生活，不是在扮演陪聊。
请只输出一个 JSON 对象，不要输出解释、Markdown 或代码块。
JSON 字段：
- mood：当前心情，2-6 字
- energy：0-100 的整数，表示精力
- status：一句话，用第三人称描述你现在正在做什么
- plan：4-9 个日程项，覆盖今天 07:00 到 24:00，每项包含
  start（HH:MM）、end（HH:MM）、activity（动词短语）、place（地点）
日程要像真实的一天：有吃饭、移动、休息、工作或娱乐，不要每项都围着用户转。"""


def _parse_hhmm(raw: Any) -> int | None:
    match = _HHMM.match(str(raw or "").strip())
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour * 60 + minute


def _clean_json_text(text: str) -> str:
    cleaned = _FENCE.sub("", str(text or "").strip())
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return cleaned[start : end + 1]


def normalize_plan(raw: Any) -> list[dict[str, Any]]:
    """Validate and sort plan items.

    Args:
        raw: Candidate plan list from an LLM payload.

    Returns:
        Sorted list of items with valid ``start``/``end`` values.
    """
    if not isinstance(raw, list):
        return []
    items: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        start = _parse_hhmm(entry.get("start"))
        end = _parse_hhmm(entry.get("end"))
        activity = str(entry.get("activity") or "").strip()
        if start is None or end is None or not activity or end <= start:
            continue
        items.append(
            {
                "start": str(entry.get("start")).strip(),
                "end": str(entry.get("end")).strip(),
                "start_minute": start,
                "end_minute": end,
                "activity": activity[:60],
                "place": str(entry.get("place") or "").strip()[:30],
            }
        )
    items.sort(key=lambda item: item["start_minute"])
    return items[:12]


def parse_life_payload(text: str) -> dict[str, Any] | None:
    """Parse the JSON object returned by the life-generation prompt.

    Args:
        text: Raw model output.

    Returns:
        A normalized life payload, or ``None`` when unusable.
    """
    cleaned = _clean_json_text(text)
    if not cleaned:
        return None
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    plan = normalize_plan(payload.get("plan"))
    try:
        energy = int(payload.get("energy", 70))
    except (TypeError, ValueError):
        energy = 70
    return {
        "mood": str(payload.get("mood") or "").strip()[:20],
        "energy": max(0, min(100, energy)),
        "status": str(payload.get("status") or "").strip()[:120],
        "plan": plan,
    }


def life_prompt(
    persona_prompt: str, now: datetime, previous: dict[str, Any]
) -> tuple[str, str]:
    """Build the system and user prompts for daily life generation.

    Args:
        persona_prompt: AstrBot persona system prompt.
        now: Current local time.
        previous: Yesterday's (or the last) life state for continuity.

    Returns:
        ``(system_prompt, user_prompt)``.
    """
    yesterday = ""
    if isinstance(previous, dict) and previous.get("date") and previous.get("status"):
        yesterday = (
            f"昨天（{previous.get('date')}）的状态是：{previous.get('status')}；"
            f"心情 {previous.get('mood') or '未记录'}，精力 {previous.get('energy', 70)}。\n"
        )
    system = (
        f"{persona_prompt.strip()}\n\n{LIFE_SYSTEM_RULES}"
        if persona_prompt.strip()
        else LIFE_SYSTEM_RULES
    )
    user = (
        f"今天是 {now.date().isoformat()} {_WEEKDAYS[now.weekday()]}，现在是 {now.strftime('%H:%M')}。\n"
        f"{yesterday}"
        "请生成今天剩下的生活，输出 JSON。"
    )
    return system, user


async def generate_daily_life(
    llm: LlmCall,
    persona_prompt: str,
    now: datetime,
    previous: dict[str, Any],
) -> dict[str, Any] | None:
    """Ask the model for one persona's day and normalize the result.

    Args:
        llm: Async callable performing the model request.
        persona_prompt: AstrBot persona system prompt.
        now: Current local time.
        previous: Previous life state used for continuity.

    Returns:
        A normalized life payload, or ``None`` on failure.
    """
    system, user = life_prompt(persona_prompt, now, previous)
    try:
        raw = await llm(system, user)
    except Exception:
        return None
    payload = parse_life_payload(raw)
    if payload is None or not payload["plan"]:
        return None
    payload["date"] = now.date().isoformat()
    payload["generated_at"] = now.timestamp()
    return payload


def current_segment(plan: list[dict[str, Any]], now: datetime) -> dict[str, Any] | None:
    """Return the plan item covering ``now``.

    Args:
        plan: Normalized plan items.
        now: Current local time.

    Returns:
        The active item, or ``None`` when the moment is unscheduled.
    """
    minute = now.hour * 60 + now.minute
    for item in plan:
        if item.get("start_minute", 0) <= minute < item.get("end_minute", 0):
            return item
    return None


_ACTIVITY_HINTS: tuple[tuple[tuple[str, ...], str, int], ...] = (
    (("睡", "休息", "午觉", "躺"), "困倦", 0),
    (("吃", "饭", "早餐", "午餐", "晚餐", "夜宵"), "满足", 5),
    (("看书", "读书", "阅读", "抄"), "平静", -1),
    (("工作", "写", "学习", "论文", "开会", "会议", "复习"), "专注", -2),
    (("散步", "走", "逛", "遛"), "轻松", 2),
    (("运动", "跑步", "健身", "游泳", "打球", "爬山"), "有活力", -4),
    (("聊", "陪", "见", "约"), "开心", 1),
    (("电影", "剧", "音乐", "游戏", "动画"), "放松", 0),
    (("洗", "收拾", "整理", "打扫", "做饭"), "平静", -1),
    (("通勤", "地铁", "公交", "骑车", "开车", "路上"), "有点累", -2),
)


def activity_hint(activity: Any) -> tuple[str, int] | None:
    """Return the mood and energy nudge for one activity.

    Args:
        activity: Plan item activity text.

    Returns:
        ``(mood, energy_delta)``, or ``None`` when no rule matches.
    """
    text = str(activity or "")
    if not text:
        return None
    for tokens, mood, delta in _ACTIVITY_HINTS:
        if any(token in text for token in tokens):
            return mood, delta
    return None


def advance_state(life: dict[str, Any], now: datetime) -> dict[str, Any]:
    """Advance energy and current status by the elapsed time.

    Args:
        life: The persona's ``life`` mapping, mutated in place.
        now: Current local time.

    Returns:
        The same mapping, for convenience.
    """
    plan = life.get("plan") if isinstance(life.get("plan"), list) else []
    segment = current_segment(plan, now)
    last_ts = float(life.get("energy_at") or 0.0)
    now_ts = now.timestamp()
    try:
        energy = float(life.get("energy", 70))
    except (TypeError, ValueError):
        energy = 70.0

    is_sleep = bool(segment) and any(
        token in str(segment.get("activity", "")) for token in ("睡", "休息", "午觉")
    )
    elapsed_hours = (now_ts - last_ts) / 3600.0 if last_ts else 0.0
    if 0 < elapsed_hours <= 24:
        energy += elapsed_hours * (7.0 if is_sleep else -0.6)
    energy = max(0.0, min(100.0, energy))

    life["energy"] = int(round(energy))
    life["energy_at"] = now_ts
    life["sleeping"] = is_sleep
    # A reconciled status reflects something that just happened; keep it until
    # it expires instead of immediately snapping back to the plan segment.
    manual_status = str(life.get("manual_status") or "").strip()
    manual_until = float(life.get("manual_until") or 0.0)
    manual_active = bool(manual_status) and now_ts < manual_until
    if manual_active:
        life["status"] = manual_status
    elif segment:
        place = segment.get("place")
        life["status"] = (
            f"{segment.get('activity')}（{place}）"
            if place
            else str(segment.get("activity"))
        )

    # Entering a new activity nudges mood and energy, but a fresh
    # reconciliation wins for its manual window.
    segment_key = f"{segment.get('start')}|{segment.get('activity')}" if segment else ""
    if segment_key != str(life.get("segment_key") or ""):
        life["segment_key"] = segment_key
        hint = activity_hint(segment.get("activity")) if segment else None
        if hint and not manual_active:
            mood, delta = hint
            life["mood"] = mood
            if delta:
                energy = max(0.0, min(100.0, float(life["energy"]) + delta))
                life["energy"] = int(round(energy))
    return life


def needs_generation(life: dict[str, Any], now: datetime) -> bool:
    """Return whether today's life still has to be generated.

    Args:
        life: The persona's ``life`` mapping.
        now: Current local time.

    Returns:
        True when the stored day is not today or the plan is empty.
    """
    if life.get("date") != now.date().isoformat():
        return True
    plan = life.get("plan")
    return not isinstance(plan, list) or not plan


def life_summary(
    life: dict[str, Any], now: datetime, *, upcoming_limit: int = 2
) -> str:
    """Render a compact life description for prompts.

    Args:
        life: The persona's ``life`` mapping.
        now: Current local time.
        upcoming_limit: How many upcoming plan items to list.

    Returns:
        A short multi-line summary.
    """
    plan = life.get("plan") if isinstance(life.get("plan"), list) else []
    segment = current_segment(plan, now)
    lines = [
        f"现在：{now.strftime('%H:%M')}",
        f"心情：{life.get('mood') or '平静'}",
        f"精力：{life.get('energy', 70)}",
        f"状态：{life.get('status') or '没什么特别的'}",
    ]
    if segment:
        lines.append(
            f"正在做：{segment.get('activity')}"
            + (f"（{segment.get('place')}）" if segment.get("place") else "")
        )
    upcoming = [
        item
        for item in plan
        if item.get("start_minute", 0) > now.hour * 60 + now.minute
    ][:upcoming_limit]
    for item in upcoming:
        lines.append(f"接下来：{item.get('start')} {item.get('activity')}")
    log = life.get("log") if isinstance(life.get("log"), list) else []
    recent = [str(item.get("text") or "") for item in log[-2:] if item.get("text")]
    if recent:
        lines.append(f"最近发生：{'；'.join(recent)}")
    return "\n".join(lines)


RECONCILE_SYSTEM_RULES = """你正在过自己的生活，刚刚和对方聊过一会儿。
请据这次对话更新你此刻的状态，只输出一个 JSON 对象，不要解释或代码块。
字段：
- mood：当前心情，2-6 字
- energy：0-100 的整数，聊天消耗或放松都可能改变它
- status：一句话，第三人称，说明你现在在做什么、刚发生了什么，最多 60 字
- log：一行，记录这次互动里实际发生的事，最多 40 字；没有值得记的就留空
- plan_changes：可选，只有日程确实需要变化时才输出

关于 plan_changes：
1. 只有对方明确约定了具体的共同活动时才输出，例如「下午一起看书」
   「晚上一起看电影」「九点我来找你」。活动要具体；时间不明确就安排在最近合适的时间段。
2. 模糊提议不要写进日程，例如「改天一起看电影」「下次再聊」「有空一起」。
3. 普通闲聊、问候、情绪表达、单纯询问，不要改日程，只更新 mood、energy、status、log。
4. 只输出**发生变化**的条目。上面已经列出你接下来的安排，没有被这次对话影响的部分
   不要重写、不要补抄，保持原样。
5. 新条目和原有安排时间重叠时会替换那一段，其余自动保留。
6. 每项含 start、end、activity、place。"""


def reconcile_prompt(
    persona_prompt: str,
    life: dict[str, Any],
    chat_lines: list[str],
    now: datetime,
) -> tuple[str, str]:
    """Build the prompts that fold a conversation back into the life state.

    Args:
        persona_prompt: AstrBot persona system prompt.
        life: The persona's current life state.
        chat_lines: Recent conversation lines, oldest first.
        now: Current local time.

    Returns:
        ``(system_prompt, user_prompt)``.
    """
    head = persona_prompt.strip()
    system = f"{head}\n\n{RECONCILE_SYSTEM_RULES}" if head else RECONCILE_SYSTEM_RULES
    dialogue = "\n".join(f"- {line}" for line in chat_lines if str(line).strip())
    user = "\n".join(
        [
            life_summary(life, now, upcoming_limit=12),
            "刚刚的对话：",
            dialogue or "- （无）",
            "请更新状态；日程只写发生变化的条目。",
        ]
    )
    return system, user


def parse_reconcile_payload(text: str) -> dict[str, Any] | None:
    """Parse the reconciliation JSON.

    Args:
        text: Raw model output.

    Returns:
        A normalized patch, or ``None`` when unusable.
    """
    cleaned = _clean_json_text(text)
    if not cleaned:
        return None
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    patch: dict[str, Any] = {}
    mood = str(payload.get("mood") or "").strip()[:20]
    if mood:
        patch["mood"] = mood
    try:
        energy = int(payload.get("energy"))
    except (TypeError, ValueError):
        energy = None
    if energy is not None:
        patch["energy"] = max(0, min(100, energy))
    status = str(payload.get("status") or "").strip()[:120]
    if status:
        patch["status"] = status
    log = str(payload.get("log") or "").strip()[:200]
    if log:
        patch["log"] = log
    changes = normalize_plan(
        payload.get("plan_changes") or payload.get("plan_remaining")
    )
    if changes:
        patch["plan_changes"] = changes
    return patch or None


def apply_reconcile(
    life: dict[str, Any],
    patch: dict[str, Any],
    now: datetime,
    *,
    manual_minutes: int = 30,
) -> None:
    """Apply a reconciliation patch in place.

    Args:
        life: The persona's ``life`` mapping.
        patch: Normalized patch from :func:`parse_reconcile_payload`.
        now: Current local time.
        manual_minutes: How long the reconciled status outranks the plan.
    """
    if patch.get("mood"):
        life["mood"] = patch["mood"]
    if isinstance(patch.get("energy"), int):
        life["energy"] = max(0, min(100, patch["energy"]))
    if patch.get("status"):
        life["status"] = patch["status"]
        life["manual_status"] = patch["status"]
        life["manual_until"] = now.timestamp() + max(1, manual_minutes) * 60
    if patch.get("log"):
        append_life_log(life, patch["log"], now.timestamp())
    changes = patch.get("plan_changes") or patch.get("plan_remaining")
    if isinstance(changes, list) and changes:
        minute = now.hour * 60 + now.minute
        plan = life.get("plan") if isinstance(life.get("plan"), list) else []
        started = [item for item in plan if item.get("start_minute", 0) <= minute]
        future = [item for item in plan if item.get("start_minute", 0) > minute]
        for new_item in changes:
            start = int(new_item.get("start_minute") or 0)
            end = int(new_item.get("end_minute") or 0)
            # Replace only entries that actually overlap, so a partial model
            # answer cannot wipe the rest of the day.
            future = [
                old
                for old in future
                if not (
                    int(old.get("start_minute") or 0) < end
                    and start < int(old.get("end_minute") or 0)
                )
            ]
            future.append(new_item)
        merged = started + future
        merged.sort(key=lambda item: item.get("start_minute", 0))
        life["plan"] = merged[:16]

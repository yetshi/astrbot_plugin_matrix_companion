"""Pure snapshot builders for the read-only panel.

No AstrBot imports: the panel API layer supplies runtime facts (platform
availability, persona prompt presence) and this module shapes the payload.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from .config import Settings, Target
from .life import advance_state, current_segment
from .user_state import is_user_sleeping, user_summary

GATE_LABELS = {
    None: "可以发送",
    "no_target": "未绑定投递会话",
    "disabled": "总开关关闭",
    "quiet_hours": "免打扰时段",
    "daily_limit": "已达每日上限",
    "min_interval": "距上次太近",
    "user_active": "用户刚说过话",
    "user_sleeping": "对方说去睡了",
    "sleeping": "正在休息",
    "unanswered_backoff": "连续未回应降速",
}


def gate_label(reason: str | None) -> str:
    """Return a human label for a gate reason.

    Args:
        reason: Machine reason from :func:`proactive.gate_reason`.

    Returns:
        A short Chinese label.
    """
    if reason in GATE_LABELS:
        return GATE_LABELS[reason]
    return str(reason or "未知原因")


def persona_snapshot(
    persona_id: str,
    target: Target,
    state: dict[str, Any],
    settings: Settings,
    now: datetime,
    *,
    prompt_ready: bool = False,
    platform_ready: bool = False,
    gate: str | None = None,
    audit_limit: int = 8,
) -> dict[str, Any]:
    """Build one persona card for the panel.

    Args:
        persona_id: Persona id.
        target: Delivery route.
        state: The persona's stored state.
        settings: Resolved settings.
        now: Current local time.
        prompt_ready: Whether the AstrBot persona prompt was found.
        platform_ready: Whether the platform instance is running.
        gate: Proactive gate reason, ``None`` when sending is allowed.
        audit_limit: How many audit rows to include.

    Returns:
        A JSON-serializable card payload.
    """
    stored_life = state.get("life") if isinstance(state.get("life"), dict) else {}
    life = advance_state(dict(stored_life), now)
    plan = life.get("plan") if isinstance(life.get("plan"), list) else []
    segment = current_segment(plan, now)
    proactive = (
        state.get("proactive") if isinstance(state.get("proactive"), dict) else {}
    )
    inbound = state.get("inbound") if isinstance(state.get("inbound"), dict) else {}
    user = state.get("user") if isinstance(state.get("user"), dict) else {}
    audit = state.get("audit") if isinstance(state.get("audit"), list) else []

    return {
        "persona_id": persona_id,
        "platform_id": target.platform_id,
        "umo": target.umo,
        "message_type": target.message_type,
        "prompt_ready": bool(prompt_ready),
        "platform_ready": bool(platform_ready),
        "life": {
            "date": life.get("date") or "",
            "mood": life.get("mood") or "",
            "energy": int(life.get("energy") or 0),
            "status": life.get("status") or "",
            "sleeping": bool(life.get("sleeping")),
            "segment": (
                {
                    "start": segment.get("start") or "",
                    "end": segment.get("end") or "",
                    "activity": segment.get("activity") or "",
                    "place": segment.get("place") or "",
                }
                if segment
                else None
            ),
            "plan": [
                {
                    "start": item.get("start") or "",
                    "end": item.get("end") or "",
                    "activity": item.get("activity") or "",
                    "place": item.get("place") or "",
                }
                for item in plan
            ],
            "log": [
                {"at": item.get("at") or 0.0, "text": item.get("text") or ""}
                for item in (life.get("log") or [])[-5:]
            ],
        },
        "proactive": {
            "sent_today": int(proactive.get("sent_today") or 0),
            "daily_limit": settings.daily_limit,
            "last_sent_at": float(proactive.get("last_sent_at") or 0.0),
            "last_text": proactive.get("last_text") or "",
            "unanswered": int(proactive.get("unanswered") or 0),
            "min_interval_minutes": settings.min_interval_minutes,
        },
        "inbound": {
            "last_at": float(inbound.get("last_at") or 0.0),
            "last_text": inbound.get("last_text") or "",
        },
        "user": {
            "last_text": str(user.get("last_text") or ""),
            "sleeping": is_user_sleeping(user, now),
            "away": str(user.get("away") or ""),
            "note": user_summary(user, now),
        },
        "gate": gate,
        "gate_label": gate_label(gate),
        "can_send": gate is None,
        "audit": list(reversed(audit[-max(1, audit_limit) :])),
    }


TASK_LABELS = {
    "life_generate": "生活生成",
    "reconcile": "对话回写",
    "proactive": "主动判断",
}
TASK_ORDER = ("life_generate", "reconcile", "proactive")


def usage_summary(
    states: dict[str, dict[str, Any]], now: datetime, *, days: int = 7
) -> dict[str, Any]:
    """Aggregate model-call counters across personas.

    Args:
        states: Mapping of persona id to state document.
        now: Current local time.
        days: How many days to include, ending today.

    Returns:
        A JSON-serializable usage block for the panel.
    """
    wanted: list[str] = []
    for offset in range(max(1, days) - 1, -1, -1):
        wanted.append((now.date() - timedelta(days=offset)).isoformat())

    per_day: dict[str, dict[str, dict[str, int]]] = {}
    for state in states.values():
        usage = state.get("usage") if isinstance(state.get("usage"), dict) else {}
        for day, tasks in usage.items():
            if day not in wanted or not isinstance(tasks, dict):
                continue
            day_bucket = per_day.setdefault(day, {})
            for task, entry in tasks.items():
                if not isinstance(entry, dict):
                    continue
                bucket = day_bucket.setdefault(
                    task, {"calls": 0, "input": 0, "cached": 0, "output": 0}
                )
                for key in ("calls", "input", "cached", "output"):
                    bucket[key] += int(entry.get(key) or 0)

    today = wanted[-1]
    summary: list[dict[str, Any]] = []
    today_total = {"calls": 0, "input": 0, "cached": 0, "output": 0}
    today_tasks = per_day.get(today, {})
    for task in TASK_ORDER:
        entry = today_tasks.get(task)
        if not entry:
            continue
        total = entry["input"] + entry["cached"] + entry["output"]
        summary.append(
            {
                "task": task,
                "label": TASK_LABELS.get(task, task),
                "calls": entry["calls"],
                "input": entry["input"],
                "cached": entry["cached"],
                "output": entry["output"],
                "total": total,
            }
        )
        for key in today_total:
            today_total[key] += entry[key]
    today_total["total"] = (
        today_total["input"] + today_total["cached"] + today_total["output"]
    )

    trend: list[dict[str, Any]] = []
    for day in wanted:
        tasks = per_day.get(day, {})
        calls = sum(int(item.get("calls") or 0) for item in tasks.values())
        total = sum(
            int(item.get("input") or 0)
            + int(item.get("cached") or 0)
            + int(item.get("output") or 0)
            for item in tasks.values()
        )
        trend.append({"date": day, "calls": calls, "total": total})

    return {
        "today": today,
        "tasks": summary,
        "today_total": today_total,
        "trend": trend,
    }


def overview_payload(
    settings: Settings,
    now: datetime,
    cards: list[dict[str, Any]],
    *,
    config_error: str = "",
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Wrap persona cards with run-level metadata.

    Args:
        settings: Resolved settings.
        now: Current local time.
        cards: Per-persona snapshots.
        config_error: Persona-target parse error, if any.
        usage: Model-call summary from :func:`usage_summary`.

    Returns:
        The panel payload.
    """
    return {
        "now": now.strftime("%Y-%m-%d %H:%M:%S"),
        "tick_minutes": settings.tick_minutes,
        "quiet_hours": settings.quiet_hours,
        "plan_time": settings.plan_time,
        "min_interval_minutes": settings.min_interval_minutes,
        "daily_limit": settings.daily_limit,
        "life_sim": settings.enable_life_sim,
        "proactive_enabled": settings.enable_proactive,
        "config_error": config_error,
        "usage": usage or {},
        "personas": cards,
    }

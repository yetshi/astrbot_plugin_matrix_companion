"""Proactive decision making: gates, prompt and response parsing.

No AstrBot imports; the caller injects the LLM function and the clock.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from .config import Settings, Target, in_quiet_hours
from .life import life_summary
from .store import ensure_proactive_day
from .user_state import is_user_sleeping

LlmCall = Callable[[str, str], Awaitable[str]]

_SKIP_TOKENS = {"skip", "none", "无", "跳过", "不说", "沉默", ""}
_FENCE = re.compile(r"^```(?:text|plain)?\s*|\s*```$", re.IGNORECASE)

PROACTIVE_SYSTEM_RULES = """你要决定此刻是否主动给对方发一条消息。
规则：
- 只输出要发送的那一句话，不要引号、不要解释、不要 Markdown。
- 必须符合你当前的状态和日程，不要凭空编造正在做的事。
- 不要复述上一次说过的话，不要连续追问，不要写“在吗”。
- 一句话，最多 40 个汉字。
- 如果此刻没什么想说的，只输出 SKIP。"""


def parse_decision(raw: str) -> str | None:
    """Normalize the model reply into a sendable message.

    Args:
        raw: Raw model output.

    Returns:
        The message text, or ``None`` when the model chose to stay silent.
    """
    text = _FENCE.sub("", str(raw or "").strip()).strip()
    text = text.strip("\"'“”‘’").strip()
    if not text:
        return None
    if text.lower() in _SKIP_TOKENS:
        return None
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    # Quotes are stripped per line: a multi-line reply still ends its first
    # line with a closing quote that the outer strip cannot reach.
    first_line = first_line.strip("\"'“”‘’").strip()
    if not first_line or first_line.lower() in _SKIP_TOKENS:
        return None
    return first_line[:160]


def proactive_prompt(
    persona_prompt: str,
    life: dict[str, Any],
    now: datetime,
    *,
    last_text: str = "",
    unanswered: int = 0,
    user_note: str = "",
    user_sleeping: bool = False,
) -> tuple[str, str]:
    """Build the prompts that decide whether to send a message.

    Args:
        persona_prompt: AstrBot persona system prompt.
        life: The persona's life state.
        now: Current local time.
        last_text: The last message this persona sent.
        unanswered: Consecutive proactive messages without a reply.
        user_note: Short description of what the user last said or is doing.
        user_sleeping: Whether the user said they went to sleep.

    Returns:
        ``(system_prompt, user_prompt)``.
    """
    head = persona_prompt.strip()
    system = f"{head}\n\n{PROACTIVE_SYSTEM_RULES}" if head else PROACTIVE_SYSTEM_RULES
    parts = [life_summary(life, now)]
    if user_note:
        parts.append(f"对方的情况：{user_note}")
    if user_sleeping:
        parts.append("对方说过去睡了，除非确实有事，否则不要发消息。")
    if last_text:
        parts.append(f"你上次主动说：{last_text}")
    if unanswered >= 2:
        parts.append("你已经连续几条没有得到回应，这次只有确实想分享点什么才开口。")
    parts.append(
        "现在要发消息吗？内容必须和对方最后说的话接得上，不要自顾自换话题。"
        "想发就直接给出一句话，不想发输出 SKIP。"
    )
    return system, "\n".join(parts)


async def decide_message(llm: LlmCall, system: str, user: str) -> str | None:
    """Run the proactive decision through the model.

    Args:
        llm: Async callable performing the model request.
        system: System prompt.
        user: User prompt.

    Returns:
        The message to send, or ``None`` to stay silent.
    """
    try:
        raw = await llm(system, user)
    except Exception:
        return None
    return parse_decision(raw)


def gate_reason(
    state: dict[str, Any],
    settings: Settings,
    now: datetime,
    *,
    target: Target | None = None,
) -> str | None:
    """Return why a proactive message may not be sent right now.

    Args:
        state: The persona's state document.
        settings: Resolved plugin settings.
        now: Current local time.
        target: Delivery route; required to exist.

    Returns:
        A short machine reason, or ``None`` when sending is allowed.
    """
    if target is None:
        return "no_target"
    if not settings.enable_proactive:
        return "disabled"
    if in_quiet_hours(now, settings.quiet_hours):
        return "quiet_hours"

    ensure_proactive_day(state, now.date().isoformat())
    proactive = (
        state.get("proactive") if isinstance(state.get("proactive"), dict) else {}
    )
    if int(proactive.get("sent_today") or 0) >= settings.daily_limit:
        return "daily_limit"

    last_sent = float(proactive.get("last_sent_at") or 0.0)
    if last_sent and (now.timestamp() - last_sent) < settings.min_interval_minutes * 60:
        return "min_interval"

    inbound = state.get("inbound") if isinstance(state.get("inbound"), dict) else {}
    last_inbound = float(inbound.get("last_at") or 0.0)
    if (
        settings.recent_user_window_minutes > 0
        and last_inbound
        and (now.timestamp() - last_inbound) < settings.recent_user_window_minutes * 60
    ):
        return "user_active"

    # The user said they went to sleep: stay quiet even after the short
    # recent-message window above has expired.
    if is_user_sleeping(state.get("user"), now):
        return "user_sleeping"

    life = state.get("life") if isinstance(state.get("life"), dict) else {}
    if bool(life.get("sleeping")):
        return "sleeping"

    if int(proactive.get("unanswered") or 0) >= 3:
        return "unanswered_backoff"
    return None


def record_sent(state: dict[str, Any], now: datetime, text: str) -> None:
    """Update proactive counters after a successful send.

    Args:
        state: The persona's state document.
        now: Send time.
        text: Delivered text.
    """
    proactive = state.setdefault("proactive", {})
    ensure_proactive_day(state, now.date().isoformat())
    proactive["sent_today"] = int(proactive.get("sent_today") or 0) + 1
    proactive["last_sent_at"] = now.timestamp()
    proactive["last_text"] = text[:160]
    proactive["unanswered"] = int(proactive.get("unanswered") or 0) + 1


def record_reply(state: dict[str, Any]) -> None:
    """Reset the unanswered streak after the user replies.

    Args:
        state: The persona's state document.
    """
    proactive = state.setdefault("proactive", {})
    proactive["unanswered"] = 0

"""Configuration parsing for the slim companion plugin.

This module intentionally has no AstrBot imports so it can be unit tested
without a running host.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any

HINT_MAX_ITEMS = 200

_FRIEND_ALIASES = {"friendmessage", "friend", "private", "dm", "single"}
_GROUP_ALIASES = {"groupmessage", "group", "room", "channel"}


class ConfigError(ValueError):
    """Raised when a persona target line cannot be parsed."""


@dataclass(frozen=True)
class Target:
    """One persona's delivery route.

    Attributes:
        persona_id: AstrBot persona id whose life is simulated.
        platform_id: AstrBot platform instance id (matches ``meta().id``).
        message_type: ``FriendMessage`` or ``GroupMessage``.
        session_id: Platform-side user id or room id, kept opaque.
    """

    persona_id: str
    platform_id: str
    message_type: str
    session_id: str

    @property
    def umo(self) -> str:
        """Return the AstrBot unified message origin string."""
        return f"{self.platform_id}:{self.message_type}:{self.session_id}"


def _normalize_message_type(raw: str) -> str:
    compact = re.sub(r"[\s_-]+", "", str(raw or "")).lower()
    if compact in _FRIEND_ALIASES:
        return "FriendMessage"
    if compact in _GROUP_ALIASES:
        return "GroupMessage"
    raise ConfigError(f"会话类型必须是 FriendMessage 或 GroupMessage，收到：{raw!r}")


def parse_target_line(line: str, *, line_number: int = 0) -> Target | None:
    """Parse one ``persona = platform:type:session`` line.

    Args:
        line: Raw configuration line.
        line_number: 1-based line number used in error messages.

    Returns:
        The parsed target, or ``None`` for blank and comment lines.

    Raises:
        ConfigError: If the line is malformed.
    """
    text = str(line or "").strip()
    if not text or text.startswith("#"):
        return None
    prefix = f"第 {line_number} 行" if line_number else "配置行"
    if "=" not in text:
        raise ConfigError(
            f"{prefix}缺少 '='，应为：人格ID = 平台实例ID:会话类型:会话ID"
        )
    persona_raw, route_raw = text.split("=", 1)
    persona_id = persona_raw.strip()
    route = route_raw.strip()
    if not persona_id:
        raise ConfigError(f"{prefix}人格 ID 为空")
    if not route:
        raise ConfigError(f"{prefix}投递路由为空")
    parts = route.split(":", 2)
    if len(parts) != 3:
        raise ConfigError(f"{prefix}投递路由应包含三段：平台实例ID:会话类型:会话ID")
    platform_id, message_type, session_id = (item.strip() for item in parts)
    if not platform_id:
        raise ConfigError(f"{prefix}平台实例 ID 为空")
    if not session_id:
        raise ConfigError(f"{prefix}会话 ID 为空")
    return Target(
        persona_id=persona_id,
        platform_id=platform_id,
        message_type=_normalize_message_type(message_type),
        session_id=session_id,
    )


def parse_targets(raw: Any) -> dict[str, Target]:
    """Parse the whole persona target text into an ordered mapping.

    Args:
        raw: Configuration value, either text or an existing mapping.

    Returns:
        Mapping of persona id to target, preserving first-seen order.

    Raises:
        ConfigError: If any line is malformed or a persona repeats.
    """
    if isinstance(raw, dict):
        lines = [
            f"{key} = {value}" for key, value in raw.items() if str(value or "").strip()
        ]
    elif isinstance(raw, (list, tuple)):
        lines = [str(item) for item in raw]
    else:
        lines = str(raw or "").splitlines()
    targets: dict[str, Target] = {}
    for index, line in enumerate(lines, start=1):
        target = parse_target_line(line, line_number=index)
        if target is None:
            continue
        if target.persona_id in targets:
            raise ConfigError(f"人格 {target.persona_id} 在配置里重复出现")
        targets[target.persona_id] = target
    return targets


_TIME_PATTERN = re.compile(r"^(\d{1,2}):(\d{2})$")


def _parse_hhmm(raw: str) -> time | None:
    match = _TIME_PATTERN.match(str(raw or "").strip())
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return time(hour=hour, minute=minute)


def parse_quiet_hours(spec: str) -> tuple[time, time] | None:
    """Parse ``HH:MM-HH:MM`` into a start/end pair.

    Args:
        spec: Quiet-hours specification; empty disables the window.

    Returns:
        ``(start, end)`` or ``None`` when disabled or unparseable.
    """
    text = str(spec or "").strip()
    if not text:
        return None
    if "-" not in text:
        return None
    start_raw, end_raw = text.split("-", 1)
    start, end = _parse_hhmm(start_raw), _parse_hhmm(end_raw)
    if start is None or end is None or start == end:
        return None
    return start, end


def in_quiet_hours(now: datetime, spec: str) -> bool:
    """Return whether ``now`` falls inside the quiet-hours window.

    Args:
        now: Timezone-aware or naive current time.
        spec: ``HH:MM-HH:MM``; crossing midnight is supported.

    Returns:
        True when the moment is inside the window.
    """
    window = parse_quiet_hours(spec)
    if window is None:
        return False
    start, end = window
    current = now.time()
    if start < end:
        return start <= current < end
    return current >= start or current < end


def parse_hhmm_or_default(raw: str, default: time) -> time:
    """Parse ``HH:MM`` and fall back to ``default`` when invalid."""
    parsed = _parse_hhmm(raw)
    return parsed if parsed is not None else default


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "是", "开"}:
            return True
        if lowered in {"0", "false", "no", "off", "否", "关"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _as_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


@dataclass(frozen=True)
class Settings:
    """Resolved plugin settings."""

    targets: dict[str, Target] = field(default_factory=dict)
    enable_proactive: bool = True
    enable_life_sim: bool = True
    plan_time: str = "07:30"
    min_interval_minutes: int = 90
    daily_limit: int = 6
    quiet_hours: str = "23:30-07:30"
    tick_minutes: int = 10
    llm_provider_id: str = ""
    recent_user_window_minutes: int = 15

    @property
    def persona_ids(self) -> list[str]:
        """Return configured persona ids in declaration order."""
        return list(self.targets)

    @property
    def plan_at(self) -> time:
        """Return the daily generation time."""
        return parse_hhmm_or_default(self.plan_time, time(hour=7, minute=30))

    @classmethod
    def from_config(cls, config: Any) -> Settings:
        """Build settings from an AstrBot config mapping.

        Args:
            config: Plugin configuration mapping (may be missing keys).

        Returns:
            The resolved immutable settings.
        """

        def get(key: str, default: Any = None) -> Any:
            if config is None:
                return default
            try:
                value = config.get(key, default)
            except Exception:
                return default
            return default if value is None else value

        return cls(
            targets=parse_targets(get("persona_targets", "")),
            enable_proactive=_as_bool(get("enable_proactive"), True),
            enable_life_sim=_as_bool(get("enable_life_sim"), True),
            plan_time=str(get("plan_time", "07:30") or "07:30"),
            min_interval_minutes=_as_int(
                get("proactive_min_interval_minutes"), 90, 1, 10080
            ),
            daily_limit=_as_int(get("proactive_daily_limit"), 6, 0, 100),
            quiet_hours=str(get("quiet_hours", "23:30-07:30") or ""),
            tick_minutes=_as_int(get("tick_minutes"), 10, 1, 1440),
            llm_provider_id=str(get("llm_provider_id", "") or "").strip(),
            recent_user_window_minutes=_as_int(
                get("recent_user_window_minutes"), 15, 0, 1440
            ),
        )

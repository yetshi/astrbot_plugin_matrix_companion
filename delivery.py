"""Platform resolution and message delivery.

Only this module depends on AstrBot runtime objects; everything else stays
plain Python so it can be tested without a host.
"""

from __future__ import annotations

from typing import Any

from astrbot.api import logger
from astrbot.api.event import MessageChain
from astrbot.api.message_components import Plain
from astrbot.core.platform.message_session import MessageSession
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform import PlatformStatus

from .config import Target


def iter_platforms(context: Any) -> list[Any]:
    """Return every registered platform instance.

    Args:
        context: AstrBot plugin context.

    Returns:
        A list of platform instances (possibly empty).
    """
    manager = getattr(context, "platform_manager", None)
    if manager is None:
        return []
    try:
        return list(manager.get_insts())
    except Exception:
        return list(getattr(manager, "platform_insts", []) or [])


def _meta_id(platform: Any) -> str:
    try:
        return str(getattr(platform.meta(), "id", "") or "").strip()
    except Exception:
        return ""


def _is_running(platform: Any) -> bool:
    status = getattr(platform, "status", None)
    if status is None:
        return True
    return status == PlatformStatus.RUNNING


def resolve_platform(context: Any, platform_id: str) -> Any | None:
    """Find the running platform instance with the given id.

    Args:
        context: AstrBot plugin context.
        platform_id: Platform instance id from the persona target.

    Returns:
        The matching platform, or ``None`` when absent or stopped.
    """
    wanted = str(platform_id or "").strip()
    if not wanted:
        return None
    for platform in iter_platforms(context):
        if _meta_id(platform) != wanted:
            continue
        if not _is_running(platform):
            return None
        return platform
    return None


def build_session(platform: Any, target: Target) -> MessageSession:
    """Build a session for ``target`` using the live platform id.

    Args:
        platform: Resolved platform instance.
        target: Configured delivery route.

    Returns:
        A session object accepted by ``send_by_session``.
    """
    return MessageSession(
        platform_name=_meta_id(platform) or target.platform_id,
        message_type=MessageType(target.message_type),
        session_id=target.session_id,
    )


async def send_text(context: Any, target: Target, text: str) -> tuple[bool, str]:
    """Deliver one plain-text message to the configured route.

    Args:
        context: AstrBot plugin context.
        target: Configured delivery route.
        text: Message body.

    Returns:
        ``(ok, note)``. On success ``note`` names the path that delivered it
        (``send_by_session`` or ``core_fallback``); on failure it holds the
        upstream error text.
    """
    body = str(text or "").strip()
    if not body:
        return False, "empty_text"
    chain = MessageChain([Plain(body)])
    platform = resolve_platform(context, target.platform_id)
    if platform is None:
        return False, f"平台实例不可用：{target.platform_id}"

    first_error = ""
    try:
        await platform.send_by_session(build_session(platform, target), chain)
        return True, "send_by_session"
    except Exception as exc:
        first_error = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "send_by_session 失败，尝试核心发送：persona=%s route=%s error=%s",
            target.persona_id,
            target.umo,
            first_error,
        )
    try:
        await context.send_message(target.umo, chain)
        return True, "core_fallback"
    except Exception as exc:
        return False, f"{first_error}; fallback {type(exc).__name__}: {exc}"

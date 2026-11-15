"""Read-only panel API registered into the AstrBot dashboard."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from astrbot.api import logger
from astrbot.api.web import request

from .delivery import resolve_platform
from .panel import overview_payload, persona_snapshot, usage_summary
from .proactive import gate_reason

PLUGIN_NAME = "astrbot_plugin_matrix_companion"
PAGE_API_PATH = f"/{PLUGIN_NAME}/page"


def _ok(data: Any = None) -> dict[str, Any]:
    return {"success": True, "data": data, "ts": int(time.time())}


def _error(message: str) -> dict[str, Any]:
    return {"success": False, "error": message, "ts": int(time.time())}


class CompanionPageApi:
    """Serve the panel snapshot and the manual trigger endpoint."""

    def __init__(self, plugin: Any) -> None:
        self.plugin = plugin

    def register(self) -> bool:
        """Register the panel routes.

        Returns:
            True when the host supports plugin web APIs.
        """
        register = getattr(self.plugin.context, "register_web_api", None)
        if not callable(register):
            logger.warning("当前 AstrBot 不支持 register_web_api，生活面板不可用")
            return False
        register(
            f"{PAGE_API_PATH}/state",
            self.get_state,
            ["GET"],
            "生活陪伴状态快照",
        )
        register(
            f"{PAGE_API_PATH}/trigger",
            self.post_trigger,
            ["POST"],
            "立即为指定人格执行一次主动判断",
        )
        register(
            f"{PAGE_API_PATH}/test-send",
            self.post_test_send,
            ["POST"],
            "发一条测试消息验证投递链路",
        )
        return True

    async def get_state(self) -> dict[str, Any]:
        """Return the current state of every configured persona."""
        settings = self.plugin.settings
        now = datetime.now()
        cards: list[dict[str, Any]] = []
        states: dict[str, dict[str, Any]] = {}
        for persona_id, target in settings.targets.items():
            state = self.plugin.store.load(persona_id)
            states[persona_id] = state
            platform = resolve_platform(self.plugin.context, target.platform_id)
            cards.append(
                persona_snapshot(
                    persona_id,
                    target,
                    state,
                    settings,
                    now,
                    prompt_ready=bool(self.plugin._persona_prompt(persona_id)),
                    platform_ready=platform is not None,
                    gate=gate_reason(state, settings, now, target=target),
                )
            )
        return _ok(
            overview_payload(
                settings,
                now,
                cards,
                config_error=str(getattr(self.plugin, "_config_error", "") or ""),
                usage=usage_summary(states, now),
            )
        )

    async def post_trigger(self) -> dict[str, Any]:
        """Run one proactive attempt for a persona from the panel."""
        payload = await request.json(default={}) or {}
        persona_id = str(payload.get("persona_id") or "").strip()
        target = self.plugin.settings.targets.get(persona_id)
        if target is None:
            return _error("未找到该人格的投递配置")
        now = datetime.now()
        try:
            await self.plugin._tick_persona(
                persona_id, target, self.plugin.settings, now, force=True
            )
        except Exception as exc:
            logger.warning(
                "面板触发的主动执行失败：persona=%s error=%s", persona_id, exc
            )
            return _error(f"执行失败：{exc}")
        state = self.plugin.store.load(persona_id)
        audit = state.get("audit") if isinstance(state.get("audit"), list) else []
        last = audit[-1] if audit else {}
        return _ok(
            {
                "event": last.get("event") or "",
                "reason": last.get("reason") or "",
                "text": last.get("text") or "",
                "error": last.get("error") or "",
            }
        )

    async def post_test_send(self) -> dict[str, Any]:
        """Send one fixed message to verify the delivery route."""
        payload = await request.json(default={}) or {}
        persona_id = str(payload.get("persona_id") or "").strip()
        text = str(payload.get("text") or "").strip()
        if not persona_id:
            return _error("缺少 persona_id")
        try:
            ok, note, body = await self.plugin.send_test_message(persona_id, text)
        except Exception as exc:
            logger.warning("面板测试发送失败：persona=%s error=%s", persona_id, exc)
            return _error(f"发送失败：{exc}")
        if not ok:
            return _error(note or "发送失败")
        return _ok({"delivery": note, "text": body})

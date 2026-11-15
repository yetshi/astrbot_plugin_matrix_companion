"""Slim companion plugin: life simulation plus proactive messages.

The plugin keeps one state document per AstrBot persona. Each persona is
bound to exactly one delivery route through the ``persona_targets`` setting,
so replacing the Matrix account or the platform instance only changes that
line and never touches the simulated life.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools

from .config import ConfigError, Settings, Target
from .delivery import send_text
from .life import (
    advance_state,
    apply_reconcile,
    generate_daily_life,
    life_summary,
    needs_generation,
    parse_reconcile_payload,
    reconcile_prompt,
)
from .page_api import CompanionPageApi
from .proactive import (
    decide_message,
    gate_reason,
    proactive_prompt,
    record_reply,
    record_sent,
)
from .store import PersonaStore, append_audit, append_pending_chat, now_iso
from .user_state import apply_cue, detect_cue, is_user_sleeping, user_summary

PLUGIN_NAME = "astrbot_plugin_matrix_companion"
GENERATION_RETRY_SECONDS = 1800


class MatrixCompanionPlugin(Star):
    """Life simulation and proactive delivery, isolated per persona."""

    def __init__(self, context: Context, config: AstrBotConfig | None = None) -> None:
        super().__init__(context, config)
        self.config = config if config is not None else {}
        self._config_error = ""
        self.settings = self._load_settings()
        self.store = PersonaStore(
            Path(StarTools.get_data_dir(PLUGIN_NAME)) / "personas"
        )
        self._task: asyncio.Task | None = None
        self._stopping = False

    # ── lifecycle ────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Start the background scheduler."""
        if self._config_error:
            logger.warning("人格投递配置有误，主动消息不会运行：%s", self._config_error)
        if not self.settings.targets:
            logger.warning("尚未配置「人格投递配置」，生活陪伴只会在配置后开始工作")
        self._stopping = False
        self._task = asyncio.create_task(
            self._scheduler_loop(), name=f"{PLUGIN_NAME}-scheduler"
        )
        self.page_api = CompanionPageApi(self)
        try:
            self.page_api.register()
        except Exception as exc:
            logger.warning("生活面板接口注册失败：%s", exc)
        logger.info(
            "生活陪伴已启动：人格 %s 个，调度间隔 %s 分钟",
            len(self.settings.targets),
            self.settings.tick_minutes,
        )

    async def terminate(self) -> None:
        """Stop the background scheduler."""
        self._stopping = True
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    # ── configuration ────────────────────────────────────────────────────

    def _load_settings(self) -> Settings:
        try:
            self._config_error = ""
            return Settings.from_config(self.config)
        except ConfigError as exc:
            self._config_error = str(exc)
            return Settings(targets={})

    def _target_for_umo(self, umo: str) -> Target | None:
        text = str(umo or "").strip()
        if not text:
            return None
        for target in self.settings.targets.values():
            if target.umo == text:
                return target
        parts = text.split(":", 2)
        if len(parts) != 3:
            return None
        for target in self.settings.targets.values():
            if (
                target.platform_id,
                target.message_type,
                target.session_id,
            ) == (parts[0], parts[1], parts[2]):
                return target
        return None

    # ── inbound observation ──────────────────────────────────────────────

    @filter.event_message_type(filter.EventMessageType.ALL, priority=1000)
    async def observe_inbound(
        self, event: AstrMessageEvent, *args: Any, **kwargs: Any
    ) -> None:
        """Remember the latest inbound message so proactive timing stays polite."""
        if self is None or self._stopping:
            return
        umo = str(getattr(event, "unified_msg_origin", "") or "")
        target = self._target_for_umo(umo)
        if target is None:
            return
        now = datetime.now()
        text = str(getattr(event, "message_str", "") or "").strip()[:200]
        state = self.store.load(target.persona_id)
        state["inbound"] = {"last_at": now.timestamp(), "last_text": text}
        record_reply(state)
        apply_cue(state.setdefault("user", {}), detect_cue(text), text, now)
        append_pending_chat(
            state.setdefault("life", {}), f"对方：{text}", now.timestamp()
        )
        try:
            self.store.save(target.persona_id, state)
        except OSError as exc:
            logger.warning(
                "记录来信时间失败：persona=%s error=%s", target.persona_id, exc
            )

    @filter.after_message_sent(priority=-1000)
    async def observe_outbound(
        self, event: AstrMessageEvent, *args: Any, **kwargs: Any
    ) -> None:
        """Queue the bot's own reply so roleplay reaches the life state."""
        if self is None or self._stopping:
            return
        if event.get_extra("_mc_skip_observe"):
            return
        umo = str(getattr(event, "unified_msg_origin", "") or "")
        target = self._target_for_umo(umo)
        if target is None:
            return
        text = self._result_text(event.get_result())
        if not text:
            return
        now = datetime.now()
        state = self.store.load(target.persona_id)
        append_pending_chat(
            state.setdefault("life", {}), f"你：{text[:200]}", now.timestamp()
        )
        try:
            self.store.save(target.persona_id, state)
        except OSError as exc:
            logger.warning("记录回复失败：persona=%s error=%s", target.persona_id, exc)

    @staticmethod
    def _result_text(result: Any) -> str:
        """Extract plain text from a message event result.

        Args:
            result: ``MessageEventResult`` or ``None``.

        Returns:
            Joined text of every text component, trimmed.
        """
        if result is None:
            return ""
        parts: list[str] = []
        for component in getattr(result, "chain", None) or []:
            value = getattr(component, "text", None)
            if value:
                parts.append(str(value))
        return "".join(parts).strip()

    # ── scheduler ────────────────────────────────────────────────────────

    async def _scheduler_loop(self) -> None:
        while not self._stopping:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # keep the loop alive across one bad tick
                logger.warning("生活陪伴调度失败：%s", exc)
            try:
                await asyncio.sleep(max(60, self.settings.tick_minutes * 60))
            except asyncio.CancelledError:
                raise

    async def _tick(self) -> None:
        settings = self._load_settings()
        self.settings = settings
        if not settings.targets:
            return
        now = datetime.now()
        for persona_id, target in settings.targets.items():
            try:
                await self._tick_persona(persona_id, target, settings, now)
            except Exception as exc:
                logger.warning("人格执行失败：persona=%s error=%s", persona_id, exc)

    async def _tick_persona(
        self,
        persona_id: str,
        target: Target,
        settings: Settings,
        now: datetime,
        *,
        force: bool = False,
    ) -> None:
        state = self.store.load(persona_id)
        life = state.setdefault("life", {})

        if settings.enable_life_sim:
            if needs_generation(life, now) and self._can_generate(settings, now, life):
                await self._generate_life(persona_id, state, settings, now)
                life = state.setdefault("life", {})
            advance_state(life, now)
            await self._reconcile_life(persona_id, state, settings, now)
            self.store.save(persona_id, state)

        effective = settings
        if force:
            effective = replace(settings, quiet_hours="", recent_user_window_minutes=0)

        reason = gate_reason(state, effective, now, target=target)
        if reason:
            self._audit(state, "proactive_skipped", now=now, reason=reason)
            self.store.save(persona_id, state)
            return

        message = await self._decide(persona_id, state, effective, now)
        if not message:
            self._audit(state, "proactive_skipped", now=now, reason="model_silent")
            self.store.save(persona_id, state)
            return

        ok, note = await send_text(self.context, target, message)
        if ok:
            record_sent(state, now, message)
            self._audit(state, "proactive_sent", now=now, text=message, delivery=note)
        else:
            self._audit(state, "proactive_failed", now=now, error=note)
            logger.warning("主动消息发送失败：persona=%s error=%s", persona_id, note)
        self.store.save(persona_id, state)

    async def send_test_message(
        self, persona_id: str, text: str = ""
    ) -> tuple[bool, str, str]:
        """Send one fixed message without touching gates, counters or the model.

        This is the panel's delivery probe: it answers "can this route receive
        a proactive message right now" without consuming quota or moving the
        last-sent timestamp.

        Args:
            persona_id: Target persona.
            text: Optional body; a default probe text is used when empty.

        Returns:
            ``(ok, note, body)`` where ``note`` names the delivery path or the
            upstream error.
        """
        target = self.settings.targets.get(persona_id)
        if target is None:
            return False, "未找到该人格的投递配置", ""
        body = (
            str(text or "").strip()
            or "这是一条测试消息，用来确认主动投递链路是否可用。"
        )
        now = datetime.now()
        state = self.store.load(persona_id)
        ok, note = await send_text(self.context, target, body)
        self._audit(
            state,
            "test_sent" if ok else "test_failed",
            now=now,
            text=body if ok else "",
            delivery=note if ok else "",
            error="" if ok else note,
        )
        try:
            self.store.save(persona_id, state)
        except OSError as exc:
            logger.warning("记录测试发送结果失败：persona=%s error=%s", persona_id, exc)
        return ok, note, body

    def _can_generate(
        self, settings: Settings, now: datetime, life: dict[str, Any]
    ) -> bool:
        if now.time() < settings.plan_at:
            return False
        failed_at = float(life.get("gen_failed_at") or 0.0)
        return not failed_at or (now.timestamp() - failed_at) > GENERATION_RETRY_SECONDS

    async def _generate_life(
        self,
        persona_id: str,
        state: dict[str, Any],
        settings: Settings,
        now: datetime,
    ) -> None:
        previous = dict(state.get("life") or {})
        payload = await generate_daily_life(
            self._call_llm, self._persona_prompt(persona_id), now, previous
        )
        life = state.setdefault("life", {})
        if payload is None:
            life["gen_failed_at"] = now.timestamp()
            self._audit(state, "life_generation_failed", now=now)
            self.store.save(persona_id, state)
            logger.warning("生活模拟生成失败：persona=%s", persona_id)
            return
        payload["energy_at"] = now.timestamp()
        state["life"] = payload
        self._audit(
            state, "life_generated", now=now, text=str(payload.get("status") or "")
        )

    async def _reconcile_life(
        self,
        persona_id: str,
        state: dict[str, Any],
        settings: Settings,
        now: datetime,
    ) -> bool:
        """Fold recent conversation back into the persona's life state.

        Args:
            persona_id: Persona being reconciled.
            state: The persona's state, mutated in place.
            settings: Resolved settings.
            now: Current local time.

        Returns:
            True when a patch was applied.
        """
        life = state.get("life")
        if not isinstance(life, dict):
            return False
        pending = life.get("pending_chat")
        if not isinstance(pending, list) or not pending:
            return False
        last_at = float(life.get("reconciled_at") or 0.0)
        interval = max(60, settings.tick_minutes * 60)
        if last_at and (now.timestamp() - last_at) < interval:
            return False
        lines = [
            str(item.get("text") or "")
            for item in pending[-10:]
            if isinstance(item, dict) and item.get("text")
        ]
        if not lines:
            life["pending_chat"] = []
            return False
        system, user = reconcile_prompt(
            self._persona_prompt(persona_id), life, lines, now
        )
        try:
            raw = await self._call_llm(system, user)
        except Exception as exc:
            logger.warning("生活状态回写失败：persona=%s error=%s", persona_id, exc)
            return False
        life["reconciled_at"] = now.timestamp()
        life["pending_chat"] = []
        patch = parse_reconcile_payload(raw)
        if patch is None:
            self._audit(state, "life_reconcile_failed", now=now)
            return False
        apply_reconcile(life, patch, now)
        self._audit(
            state,
            "life_reconciled",
            now=now,
            text=str(patch.get("status") or patch.get("log") or ""),
        )
        logger.info(
            "生活状态已按对话更新：persona=%s status=%s",
            persona_id,
            life.get("status"),
        )
        return True

    async def _decide(
        self,
        persona_id: str,
        state: dict[str, Any],
        settings: Settings,
        now: datetime,
    ) -> str | None:
        life = state.get("life") if isinstance(state.get("life"), dict) else {}
        proactive = (
            state.get("proactive") if isinstance(state.get("proactive"), dict) else {}
        )
        system, user = proactive_prompt(
            self._persona_prompt(persona_id),
            life,
            now,
            last_text=str(proactive.get("last_text") or ""),
            unanswered=int(proactive.get("unanswered") or 0),
            user_note=user_summary(state.get("user"), now),
            user_sleeping=is_user_sleeping(state.get("user"), now),
        )
        return await decide_message(self._call_llm, system, user)

    # ── model access ─────────────────────────────────────────────────────

    def _resolve_provider(self) -> Any | None:
        provider_id = str(self.settings.llm_provider_id or "").strip()
        if provider_id:
            getter = getattr(self.context, "get_provider_by_id", None)
            if callable(getter):
                try:
                    provider = getter(provider_id)
                except Exception:
                    provider = None
                if provider is not None:
                    return provider
        for target in self.settings.targets.values():
            try:
                provider = self.context.get_using_provider(target.umo)
            except Exception:
                provider = None
            if provider is not None:
                return provider
        try:
            return self.context.get_using_provider()
        except Exception:
            return None

    async def _call_llm(self, system: str, user: str) -> str:
        provider = self._resolve_provider()
        if provider is None:
            raise RuntimeError("没有可用的模型 provider")
        response = await provider.text_chat(prompt=user, system_prompt=system)
        text = getattr(response, "completion_text", "") or ""
        if text:
            return str(text)
        chain = getattr(response, "result_chain", None)
        parts: list[str] = []
        for component in getattr(chain, "chain", []) or []:
            value = getattr(component, "text", None)
            if value:
                parts.append(str(value))
        return "".join(parts)

    def _persona_prompt(self, persona_id: str) -> str:
        manager = getattr(self.context, "persona_manager", None)
        if manager is None:
            return ""
        persona: Any = None
        getter = getattr(manager, "get_persona_v3_by_id", None)
        if callable(getter):
            try:
                persona = getter(persona_id)
            except Exception:
                persona = None
        if persona is None:
            for item in list(getattr(manager, "personas_v3", None) or []):
                if isinstance(item, dict) and str(item.get("name") or "") == persona_id:
                    persona = item
                    break
        if isinstance(persona, dict):
            return str(persona.get("prompt") or "").strip()
        return str(getattr(persona, "prompt", "") or "").strip()

    # ── commands ─────────────────────────────────────────────────────────

    @filter.command("生活")
    async def show_life(self, event: AstrMessageEvent):
        """Show the current persona's life state and plan."""
        event.set_extra("_mc_skip_observe", True)
        target = self._target_for_umo(
            str(getattr(event, "unified_msg_origin", "") or "")
        )
        if target is None:
            yield event.plain_result(
                "这个会话还没有绑定人格。请在插件配置的「人格投递配置」里加一行：\n"
                "人格ID = 平台实例ID:FriendMessage:会话ID"
            )
            return
        now = datetime.now()
        state = self.store.load(target.persona_id)
        life = advance_state(dict(state.get("life") or {}), now)
        lines = [f"人格：{target.persona_id}", life_summary(life, now)]
        plan = life.get("plan") if isinstance(life.get("plan"), list) else []
        if plan:
            lines.append("今日日程：")
            for item in plan:
                place = f"（{item['place']}）" if item.get("place") else ""
                lines.append(
                    f"  {item['start']}-{item['end']} {item['activity']}{place}"
                )
        reason = gate_reason(state, self.settings, now, target=target)
        lines.append(
            f"主动状态：{'可以发送' if reason is None else f'暂不发送（{reason}）'}"
        )
        yield event.plain_result("\n".join(lines))

    @filter.command("主动")
    async def force_proactive(self, event: AstrMessageEvent):
        """Run one proactive attempt immediately for this persona."""
        event.set_extra("_mc_skip_observe", True)
        target = self._target_for_umo(
            str(getattr(event, "unified_msg_origin", "") or "")
        )
        if target is None:
            yield event.plain_result("这个会话还没有绑定人格，无法执行主动判断。")
            return
        now = datetime.now()
        await self._tick_persona(
            target.persona_id, target, self.settings, now, force=True
        )
        state = self.store.load(target.persona_id)
        audit = state.get("audit") if isinstance(state.get("audit"), list) else []
        last = audit[-1] if audit else {}
        event_name = str(last.get("event") or "无记录")
        detail = str(last.get("reason") or last.get("text") or last.get("error") or "")
        yield event.plain_result(
            f"已执行一次主动判断：{event_name}" + (f"（{detail}）" if detail else "")
        )

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _audit(
        state: dict[str, Any], event: str, *, now: datetime, **fields: Any
    ) -> None:
        entry: dict[str, Any] = {"at": now_iso(now), "event": event}
        for key, value in fields.items():
            if value is None or value == "":
                continue
            entry[key] = value
        append_audit(state, entry)

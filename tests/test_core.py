# -*- coding: utf-8 -*-
"""Unit tests for the AstrBot-independent modules of the slim companion."""

from __future__ import annotations

import json
import shutil
import sys
import uuid
from datetime import datetime, time, timedelta
from pathlib import Path

import pytest

PLUGINS_DIR = Path(__file__).resolve().parents[2]
if str(PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGINS_DIR))

from astrbot_plugin_matrix_companion import config as cfg  # noqa: E402
from astrbot_plugin_matrix_companion import life as life_mod  # noqa: E402
from astrbot_plugin_matrix_companion import panel as panel_mod  # noqa: E402
from astrbot_plugin_matrix_companion import proactive as pro  # noqa: E402
from astrbot_plugin_matrix_companion import store as store_mod  # noqa: E402
from astrbot_plugin_matrix_companion import user_state as user_mod  # noqa: E402


@pytest.fixture()
def store_dir():
    """Provide an isolated directory without pytest's mode=0o700 temp root.

    pytest's ``tmp_path`` creates the base directory with ``mode=0o700``, which
    grants only the user SID; a restricted sandbox token cannot then list it.
    A plain directory under the plugin keeps the tests runnable in both
    confined and unconfined environments.
    """
    root = Path(__file__).resolve().parents[1] / ".test-tmp"
    root.mkdir(exist_ok=True)
    case = root / f"case-{uuid.uuid4().hex[:8]}"
    case.mkdir()
    try:
        yield case
    finally:
        shutil.rmtree(case, ignore_errors=True)


# ── config ───────────────────────────────────────────────────────────────


def test_parse_target_keeps_matrix_id_with_colons() -> None:
    target = cfg.parse_target_line("阿星 = matrix:FriendMessage:@alice:matrix.org")
    assert target is not None
    assert target.persona_id == "阿星"
    assert target.platform_id == "matrix"
    assert target.message_type == "FriendMessage"
    assert target.session_id == "@alice:matrix.org"
    assert target.umo == "matrix:FriendMessage:@alice:matrix.org"
    assert target.umo.split(":", 2) == ["matrix", "FriendMessage", "@alice:matrix.org"]


@pytest.mark.parametrize("raw", ["friend", "DM", "private", "FriendMessage"])
def test_parse_target_normalizes_friend_aliases(raw: str) -> None:
    target = cfg.parse_target_line(f"A = matrix:{raw}:session-1")
    assert target is not None
    assert target.message_type == "FriendMessage"


@pytest.mark.parametrize("raw", ["group", "Room", "GroupMessage"])
def test_parse_target_normalizes_group_aliases(raw: str) -> None:
    target = cfg.parse_target_line(f"A = matrix:{raw}:!room:server")
    assert target is not None
    assert target.message_type == "GroupMessage"
    assert target.session_id == "!room:server"


def test_parse_target_skips_blank_and_comment_lines() -> None:
    assert cfg.parse_target_line("") is None
    assert cfg.parse_target_line("   ") is None
    assert cfg.parse_target_line("# 说明") is None


@pytest.mark.parametrize(
    "line",
    [
        "A matrix:FriendMessage:x",
        "= matrix:FriendMessage:x",
        "A = ",
        "A = matrix",
        "A = matrix:unknown:x",
        "A = :FriendMessage:x",
        "A = matrix:FriendMessage:",
    ],
)
def test_parse_target_rejects_malformed_lines(line: str) -> None:
    with pytest.raises(cfg.ConfigError):
        cfg.parse_target_line(line)


def test_parse_targets_rejects_duplicate_persona() -> None:
    with pytest.raises(cfg.ConfigError):
        cfg.parse_targets("A = matrix:FriendMessage:x\nA = matrix:FriendMessage:y")


def test_parse_targets_accepts_mapping() -> None:
    targets = cfg.parse_targets({"A": "matrix:FriendMessage:x"})
    assert list(targets) == ["A"]


@pytest.mark.parametrize(
    ("spec", "now", "expected"),
    [
        ("23:30-07:30", datetime(2026, 1, 1, 23, 45), True),
        ("23:30-07:30", datetime(2026, 1, 1, 3, 0), True),
        ("23:30-07:30", datetime(2026, 1, 1, 12, 0), False),
        ("09:00-18:00", datetime(2026, 1, 1, 9, 0), True),
        ("09:00-18:00", datetime(2026, 1, 1, 18, 0), False),
        ("", datetime(2026, 1, 1, 23, 0), False),
        ("bad", datetime(2026, 1, 1, 23, 0), False),
    ],
)
def test_in_quiet_hours(spec: str, now: datetime, expected: bool) -> None:
    assert cfg.in_quiet_hours(now, spec) is expected


def test_settings_from_config_clamps_and_defaults() -> None:
    settings = cfg.Settings.from_config(
        {
            "persona_targets": "A = matrix:FriendMessage:x",
            "proactive_daily_limit": 999,
            "proactive_min_interval_minutes": 0,
            "tick_minutes": -5,
            "enable_life_sim": "false",
        }
    )
    assert settings.daily_limit == 100
    assert settings.min_interval_minutes == 1
    assert settings.tick_minutes == 1
    assert settings.enable_life_sim is False
    assert settings.persona_ids == ["A"]
    assert settings.plan_at == time(7, 30)


def test_settings_from_empty_config_is_usable() -> None:
    settings = cfg.Settings.from_config({})
    assert settings.targets == {}
    assert settings.enable_proactive is True


# ── store ────────────────────────────────────────────────────────────────


def test_store_round_trip(store_dir: Path) -> None:
    store = store_mod.PersonaStore(store_dir)
    state = store.load("阿星")
    state["life"]["status"] = "在看书"
    store.save("阿星")
    assert (store_dir / f"{store_mod.safe_stem('阿星')}.json").is_file()

    fresh = store_mod.PersonaStore(store_dir)
    reloaded = fresh.load("阿星")
    assert reloaded["life"]["status"] == "在看书"


def test_store_merges_missing_new_fields(store_dir: Path) -> None:
    path = store_dir / f"{store_mod.safe_stem('A')}.json"
    path.write_text(
        json.dumps({"persona_id": "A", "life": {"status": "旧"}}), encoding="utf-8"
    )
    state = store_mod.PersonaStore(store_dir).load("A")
    assert state["life"]["status"] == "旧"
    assert state["life"]["energy"] == 70
    assert state["proactive"]["sent_today"] == 0


def test_store_recovers_from_corrupt_file(store_dir: Path) -> None:
    path = store_dir / f"{store_mod.safe_stem('A')}.json"
    path.write_text("{not json", encoding="utf-8")
    state = store_mod.PersonaStore(store_dir).load("A")
    assert state["load_error"] is True
    assert state["life"]["plan"] == []


def test_safe_stem_encodes_path_separators() -> None:
    assert "/" not in store_mod.safe_stem("a/b")
    assert ":" not in store_mod.safe_stem("a:b")


def test_ensure_proactive_day_resets_counter() -> None:
    state = store_mod.empty_state("A")
    state["proactive"].update({"day": "2026-01-01", "sent_today": 3})
    store_mod.ensure_proactive_day(state, "2026-01-02")
    assert state["proactive"]["sent_today"] == 0
    assert state["proactive"]["day"] == "2026-01-02"


# ── life ─────────────────────────────────────────────────────────────────


def test_parse_life_payload_accepts_fenced_json() -> None:
    payload = life_mod.parse_life_payload(
        '```json\n{"mood":"平静","energy":80,"status":"在整理书架",'
        '"plan":[{"start":"09:00","end":"10:30","activity":"整理书架","place":"家"}]}\n```'
    )
    assert payload is not None
    assert payload["mood"] == "平静"
    assert payload["energy"] == 80
    assert payload["plan"][0]["start_minute"] == 540


def test_parse_life_payload_rejects_garbage() -> None:
    assert life_mod.parse_life_payload("我不想输出 JSON") is None
    assert life_mod.parse_life_payload("{broken") is None


def test_normalize_plan_filters_and_sorts() -> None:
    plan = life_mod.normalize_plan(
        [
            {"start": "20:00", "end": "21:00", "activity": "读书"},
            {"start": "08:00", "end": "09:00", "activity": "早餐"},
            {"start": "10:00", "end": "09:00", "activity": "倒序"},
            {"start": "bad", "end": "09:00", "activity": "无效"},
            {"start": "11:00", "end": "12:00", "activity": ""},
        ]
    )
    assert [item["activity"] for item in plan] == ["早餐", "读书"]


def test_current_segment_and_summary() -> None:
    plan = life_mod.normalize_plan(
        [
            {"start": "09:00", "end": "10:00", "activity": "工作", "place": "书房"},
            {"start": "10:00", "end": "11:00", "activity": "散步", "place": "楼下"},
        ]
    )
    life = {"date": "2026-01-01", "mood": "平静", "energy": 70, "status": "", "plan": plan}
    assert life_mod.current_segment(plan, datetime(2026, 1, 1, 9, 30))["activity"] == "工作"
    assert life_mod.current_segment(plan, datetime(2026, 1, 1, 12, 0)) is None

    advanced = life_mod.advance_state(dict(life), datetime(2026, 1, 1, 9, 30))
    assert "工作" in advanced["status"]
    assert advanced["sleeping"] is False


def test_advance_state_decays_energy() -> None:
    life = {"energy": 70, "energy_at": 1_000_000.0, "plan": []}
    advanced = life_mod.advance_state(life, datetime.fromtimestamp(1_000_000.0 + 3600))
    assert advanced["energy"] == 69


def test_advance_state_recovers_while_sleeping() -> None:
    plan = life_mod.normalize_plan(
        [{"start": "23:00", "end": "23:59", "activity": "睡觉"}]
    )
    life = {"energy": 40, "energy_at": 1_000_000.0, "plan": plan}
    moment = datetime.fromtimestamp(1_000_000.0 + 3600).replace(hour=23, minute=30)
    advanced = life_mod.advance_state(life, moment)
    assert advanced["sleeping"] is True
    assert advanced["energy"] > 40


def test_needs_generation() -> None:
    now = datetime(2026, 1, 2, 9, 0)
    assert life_mod.needs_generation({"date": "2026-01-01", "plan": []}, now) is True
    assert life_mod.needs_generation({"date": "2026-01-02", "plan": []}, now) is True
    assert (
        life_mod.needs_generation(
            {"date": "2026-01-02", "plan": [{"start": "09:00"}]}, now
        )
        is False
    )


# ── proactive ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw", ["SKIP", "skip", " none ", "无", "跳过", "", "   "])
def test_parse_decision_silent(raw: str) -> None:
    assert pro.parse_decision(raw) is None


def test_parse_decision_extracts_first_line() -> None:
    assert pro.parse_decision('"外面下雨了，记得带伞。"\n不要解释') == "外面下雨了，记得带伞。"
    assert pro.parse_decision("```text\n刚看到一只猫\n```") == "刚看到一只猫"


def _state(now: datetime, **proactive: object) -> dict:
    state = store_mod.empty_state("A")
    state["proactive"].update({"day": now.date().isoformat(), "sent_today": 0})
    state["proactive"].update(proactive)
    return state


def _settings(**kwargs: object) -> cfg.Settings:
    base = dict(
        targets={"A": cfg.Target("A", "matrix", "FriendMessage", "@a:server")},
        enable_proactive=True,
        enable_life_sim=True,
        min_interval_minutes=90,
        daily_limit=6,
        quiet_hours="23:30-07:30",
        recent_user_window_minutes=15,
    )
    base.update(kwargs)
    return cfg.Settings(**base)


def test_gate_allows_by_default() -> None:
    now = datetime(2026, 1, 1, 14, 0)
    target = cfg.Target("A", "matrix", "FriendMessage", "@a:server")
    assert pro.gate_reason(_state(now), _settings(), now, target=target) is None


def test_gate_requires_target() -> None:
    now = datetime(2026, 1, 1, 14, 0)
    assert pro.gate_reason(_state(now), _settings(), now, target=None) == "no_target"


def test_gate_disabled() -> None:
    now = datetime(2026, 1, 1, 14, 0)
    target = cfg.Target("A", "matrix", "FriendMessage", "@a:server")
    settings = _settings(enable_proactive=False)
    assert pro.gate_reason(_state(now), settings, now, target=target) == "disabled"


def test_gate_quiet_hours() -> None:
    now = datetime(2026, 1, 1, 23, 45)
    target = cfg.Target("A", "matrix", "FriendMessage", "@a:server")
    assert pro.gate_reason(_state(now), _settings(), now, target=target) == "quiet_hours"


def test_gate_daily_limit() -> None:
    now = datetime(2026, 1, 1, 14, 0)
    target = cfg.Target("A", "matrix", "FriendMessage", "@a:server")
    state = _state(now, sent_today=6)
    assert pro.gate_reason(state, _settings(), now, target=target) == "daily_limit"


def test_gate_min_interval() -> None:
    now = datetime(2026, 1, 1, 14, 0)
    target = cfg.Target("A", "matrix", "FriendMessage", "@a:server")
    state = _state(now, last_sent_at=(now - timedelta(minutes=10)).timestamp())
    assert pro.gate_reason(state, _settings(), now, target=target) == "min_interval"


def test_gate_user_active() -> None:
    now = datetime(2026, 1, 1, 14, 0)
    target = cfg.Target("A", "matrix", "FriendMessage", "@a:server")
    state = _state(now)
    state["inbound"] = {"last_at": (now - timedelta(minutes=3)).timestamp()}
    assert pro.gate_reason(state, _settings(), now, target=target) == "user_active"


def test_gate_sleeping() -> None:
    now = datetime(2026, 1, 1, 14, 0)
    target = cfg.Target("A", "matrix", "FriendMessage", "@a:server")
    state = _state(now)
    state["life"]["sleeping"] = True
    assert pro.gate_reason(state, _settings(), now, target=target) == "sleeping"


def test_gate_unanswered_backoff() -> None:
    now = datetime(2026, 1, 1, 14, 0)
    target = cfg.Target("A", "matrix", "FriendMessage", "@a:server")
    state = _state(now, unanswered=3)
    assert pro.gate_reason(state, _settings(), now, target=target) == "unanswered_backoff"


def test_record_sent_and_reply() -> None:
    now = datetime(2026, 1, 1, 14, 0)
    state = _state(now)
    pro.record_sent(state, now, "你好")
    assert state["proactive"]["sent_today"] == 1
    assert state["proactive"]["unanswered"] == 1
    assert state["proactive"]["last_text"] == "你好"
    pro.record_reply(state)
    assert state["proactive"]["unanswered"] == 0


def test_proactive_prompt_mentions_last_text_and_fatigue() -> None:
    now = datetime(2026, 1, 1, 14, 0)
    life = {"mood": "平静", "energy": 60, "status": "在窗边", "plan": []}
    system, user = pro.proactive_prompt(
        "你是阿星", life, now, last_text="刚吃过饭", unanswered=2
    )
    assert "你是阿星" in system
    assert "SKIP" in system
    assert "刚吃过饭" in user
    assert "连续几条" in user

# ── panel ────────────────────────────────────────────────────────────────


def _panel_settings(**kwargs: object) -> cfg.Settings:
    base = dict(
        targets={"A": cfg.Target("A", "matrix", "FriendMessage", "@a:server")},
        daily_limit=6,
        min_interval_minutes=90,
    )
    base.update(kwargs)
    return cfg.Settings(**base)


def test_gate_label_maps_known_and_unknown_reasons() -> None:
    assert panel_mod.gate_label(None) == "可以发送"
    assert panel_mod.gate_label("quiet_hours") == "免打扰时段"
    assert panel_mod.gate_label("something_new") == "something_new"


def test_persona_snapshot_exposes_current_segment_and_plan() -> None:
    now = datetime(2026, 1, 1, 9, 30)
    plan = life_mod.normalize_plan(
        [{"start": "09:00", "end": "10:00", "activity": "工作", "place": "书房"}]
    )
    state = store_mod.empty_state("A")
    state["life"] = {
        "date": "2026-01-01",
        "mood": "平静",
        "energy": 66,
        "status": "在书房",
        "plan": plan,
        "energy_at": now.timestamp(),
    }
    state["proactive"].update({"sent_today": 2, "unanswered": 1})
    state["audit"] = [{"at": "2026-01-01T09:00:00", "event": "proactive_sent"}]

    card = panel_mod.persona_snapshot(
        "A",
        cfg.Target("A", "matrix", "FriendMessage", "@a:server"),
        state,
        _panel_settings(),
        now,
        prompt_ready=True,
        platform_ready=True,
        gate=None,
    )

    assert card["persona_id"] == "A"
    assert card["life"]["segment"]["activity"] == "工作"
    assert card["life"]["plan"][0]["place"] == "书房"
    assert card["proactive"]["sent_today"] == 2
    assert card["proactive"]["daily_limit"] == 6
    assert card["can_send"] is True
    assert card["gate_label"] == "可以发送"
    assert card["audit"][0]["event"] == "proactive_sent"


def test_persona_snapshot_reports_blocked_gate() -> None:
    now = datetime(2026, 1, 1, 9, 30)
    state = store_mod.empty_state("A")
    card = panel_mod.persona_snapshot(
        "A",
        cfg.Target("A", "matrix", "FriendMessage", "@a:server"),
        state,
        _panel_settings(),
        now,
        gate="quiet_hours",
    )
    assert card["can_send"] is False
    assert card["gate_label"] == "免打扰时段"


def test_overview_payload_carries_run_metadata() -> None:
    now = datetime(2026, 1, 1, 9, 30)
    payload = panel_mod.overview_payload(
        _panel_settings(tick_minutes=7), now, [], config_error="bad line"
    )
    assert payload["tick_minutes"] == 7
    assert payload["config_error"] == "bad line"
    assert payload["personas"] == []
    assert payload["now"].startswith("2026-01-01")

def test_page_api_registers_exactly_three_routes() -> None:
    from astrbot_plugin_matrix_companion import page_api as page_mod

    recorded: list[tuple[str, tuple[str, ...]]] = []

    class _Ctx:
        def register_web_api(self, route, handler, methods, desc):
            recorded.append((route, tuple(methods)))

    plugin = type("P", (), {"context": _Ctx()})()
    api = page_mod.CompanionPageApi(plugin)
    assert api.register() is True
    routes = {route: methods for route, methods in recorded}
    prefix = "/astrbot_plugin_matrix_companion/page"
    assert routes[f"{prefix}/state"] == ("GET",)
    assert routes[f"{prefix}/trigger"] == ("POST",)
    assert routes[f"{prefix}/test-send"] == ("POST",)


def test_page_api_degrades_without_web_api_support() -> None:
    from astrbot_plugin_matrix_companion import page_api as page_mod

    plugin = type("P", (), {"context": object()})()
    assert page_mod.CompanionPageApi(plugin).register() is False

def test_panel_bridge_uses_relative_endpoint() -> None:
    html = (
        Path(__file__).resolve().parents[1]
        / "pages"
        / "companion-panel"
        / "index.html"
    ).read_text(encoding="utf-8")
    assert 'const ENDPOINT_PREFIXES = ["page", ""];' in html
    # The dashboard already prefixes the plugin name; a full prefix would be
    # doubled into /extensions/<plugin>/<plugin>/page/... and 404.
    assert "/astrbot_plugin_matrix_companion/page" not in html
    assert "api/plugins/extensions" not in html


def test_page_routes_match_the_bridge_path() -> None:
    from astrbot_plugin_matrix_companion import page_api as page_mod

    plugin_name = "astrbot_plugin_matrix_companion"
    assert page_mod.PAGE_API_PATH == f"/{plugin_name}/page"
    for action in ("state", "trigger", "test-send"):
        # The dashboard builds /extensions/<plugin_name>/<endpoint>, where the
        # panel sends endpoint "page/<action>"; both sides must agree.
        assert f"{page_mod.PAGE_API_PATH}/{action}" == (
            f"/{plugin_name}/page/{action}"
        )

# ── conversation feedback ────────────────────────────────────────────────


def test_parse_reconcile_payload_extracts_patch() -> None:
    raw = (
        '```json\n{"mood":"轻快","energy":55,"status":"刚和你聊完，还笑着",'
        '"log":"和你聊了很久","plan_remaining":'
        '[{"start":"21:00","end":"22:00","activity":"洗漱","place":"家"}]}\n```'
    )
    patch = life_mod.parse_reconcile_payload(raw)
    assert patch is not None
    assert patch["mood"] == "轻快"
    assert patch["energy"] == 55
    assert patch["log"] == "和你聊了很久"
    assert patch["plan_remaining"][0]["start_minute"] == 1260


@pytest.mark.parametrize("raw", ["算了", "{}", "[1,2]", "null"])
def test_parse_reconcile_payload_rejects_unusable(raw: str) -> None:
    assert life_mod.parse_reconcile_payload(raw) is None


def test_apply_reconcile_overrides_status_and_plan() -> None:
    now = datetime(2026, 1, 1, 20, 0)
    life = {
        "mood": "平静",
        "energy": 50,
        "status": "在看书",
        "plan": life_mod.normalize_plan(
            [
                {"start": "19:00", "end": "20:30", "activity": "看书"},
                {"start": "21:00", "end": "22:00", "activity": "洗澡"},
            ]
        ),
        "log": [],
        "manual_status": "",
        "manual_until": 0.0,
    }
    patch = {
        "mood": "轻快",
        "energy": 70,
        "status": "刚和你聊完天",
        "log": "一起聊了会儿",
        "plan_remaining": life_mod.normalize_plan(
            [{"start": "21:00", "end": "22:30", "activity": "看电影", "place": "客厅"}]
        ),
    }
    life_mod.apply_reconcile(life, patch, now)

    assert life["mood"] == "轻快"
    assert life["energy"] == 70
    assert life["status"] == "刚和你聊完天"
    assert life["manual_until"] > now.timestamp()
    assert life["log"][-1]["text"] == "一起聊了会儿"
    activities = [item["activity"] for item in life["plan"]]
    assert "看书" in activities
    assert "洗澡" not in activities
    assert "看电影" in activities


def test_advance_state_prefers_manual_status_until_expiry() -> None:
    now = datetime(2026, 1, 1, 20, 10)
    plan = life_mod.normalize_plan(
        [{"start": "20:00", "end": "21:00", "activity": "看书"}]
    )
    fresh = {
        "energy": 50,
        "energy_at": now.timestamp(),
        "plan": plan,
        "status": "刚聊完天",
        "manual_status": "刚聊完天",
        "manual_until": now.timestamp() + 1800,
    }
    assert life_mod.advance_state(fresh, now)["status"] == "刚聊完天"

    expired = dict(fresh)
    expired["manual_until"] = now.timestamp() - 1
    assert "看书" in life_mod.advance_state(expired, now)["status"]


def test_pending_chat_and_log_are_capped() -> None:
    life = store_mod.empty_state("A")["life"]
    for index in range(30):
        store_mod.append_pending_chat(life, f"第{index}条", float(index))
        store_mod.append_life_log(life, f"事件{index}", float(index))
    assert len(life["pending_chat"]) == 20
    assert len(life["log"]) == 20
    assert life["pending_chat"][-1]["text"] == "第29条"


def test_persona_snapshot_includes_life_log() -> None:
    now = datetime(2026, 1, 1, 12, 0)
    state = store_mod.empty_state("A")
    state["life"]["log"] = [{"at": now.timestamp(), "text": "一起看了电影"}]
    card = panel_mod.persona_snapshot(
        "A",
        cfg.Target("A", "matrix", "FriendMessage", "@a:server"),
        state,
        _panel_settings(),
        now,
    )
    assert card["life"]["log"][-1]["text"] == "一起看了电影"

# ── user state cues ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text", ["我去睡了", "晚安", "困了", "准备睡了", "睡觉去"]
)
def test_detect_sleep_cue(text: str) -> None:
    assert user_mod.detect_cue(text).kind == "sleep"


@pytest.mark.parametrize("text", ["睡不着", "还没睡", "不想睡"])
def test_sleep_negation_is_neutral(text: str) -> None:
    assert user_mod.detect_cue(text).kind == "none"


def test_detect_wake_away_back() -> None:
    assert user_mod.detect_cue("我起床了").kind == "wake"
    assert user_mod.detect_cue("我出门了").kind == "away"
    assert user_mod.detect_cue("我回来了").kind == "back"


def test_any_message_clears_sleeping() -> None:
    now = datetime(2026, 1, 1, 23, 0)
    user: dict = {}
    user_mod.apply_cue(user, user_mod.detect_cue("我去睡了"), "我去睡了", now)
    assert user["sleeping"] is True
    assert user["last_text"] == "我去睡了"

    later = datetime(2026, 1, 2, 3, 0)
    user_mod.apply_cue(user, user_mod.detect_cue("在吗"), "在吗", later)
    assert user["sleeping"] is False


def test_is_user_sleeping_times_out() -> None:
    now = datetime(2026, 1, 1, 23, 0)
    user = {"sleeping": True, "sleeping_since": now.timestamp()}
    assert user_mod.is_user_sleeping(user, now) is True
    assert user_mod.is_user_sleeping(user, datetime(2026, 1, 3, 0, 0)) is False


def test_gate_blocks_when_user_said_goodnight() -> None:
    now = datetime(2026, 1, 1, 22, 30)
    target = cfg.Target("A", "matrix", "FriendMessage", "@a:server")
    state = _state(now)
    state["user"] = {"sleeping": True, "sleeping_since": now.timestamp() - 60}
    assert pro.gate_reason(state, _settings(), now, target=target) == "user_sleeping"


def test_proactive_prompt_includes_user_note() -> None:
    now = datetime(2026, 1, 1, 22, 0)
    life = {"mood": "平静", "energy": 50, "status": "在看书", "plan": []}
    _, user_prompt = pro.proactive_prompt(
        "你是阿星", life, now, user_note="对方最后说：我先睡了"
    )
    assert "我先睡了" in user_prompt
    assert "接得上" in user_prompt


def test_persona_snapshot_includes_user_state() -> None:
    now = datetime(2026, 1, 1, 22, 30)
    state = store_mod.empty_state("A")
    state["user"] = {
        "sleeping": True,
        "sleeping_since": now.timestamp() - 60,
        "last_text": "我去睡了",
    }
    card = panel_mod.persona_snapshot(
        "A",
        cfg.Target("A", "matrix", "FriendMessage", "@a:server"),
        state,
        _panel_settings(),
        now,
    )
    assert card["user"]["sleeping"] is True
    assert card["user"]["last_text"] == "我去睡了"

def test_panel_html_has_balanced_document_tags() -> None:
    """Guards the host's bridge injection point.

    The host injects the bridge SDK by replacing the FIRST ``</body>``. A
    literal ``</body>`` inside the inline script (even in a comment) moves the
    injection into the middle of the code, which makes the browser render the
    rest of the script as text.
    """
    html = (
        Path(__file__).resolve().parents[1]
        / "pages"
        / "companion-panel"
        / "index.html"
    ).read_text(encoding="utf-8")
    assert html.count("</body>") == 1
    assert html.count("</html>") == 1
    assert html.count("<script") == html.count("</script>")
    # Every script must close before the injection point.
    assert html.rindex("</script>") < html.index("</body>")

# ── schedule merge and activity mood ─────────────────────────────────────


def test_apply_reconcile_keeps_non_overlapping_future_items() -> None:
    now = datetime(2026, 1, 1, 15, 0)
    life = {
        "plan": life_mod.normalize_plan(
            [
                {"start": "16:00", "end": "17:00", "activity": "健身"},
                {"start": "19:00", "end": "20:00", "activity": "做饭"},
            ]
        ),
        "log": [],
        "manual_status": "",
        "manual_until": 0.0,
    }
    patch = {
        "plan_remaining": life_mod.normalize_plan(
            [{"start": "19:30", "end": "20:30", "activity": "一起看书"}]
        )
    }
    life_mod.apply_reconcile(life, patch, now)

    activities = [item["activity"] for item in life["plan"]]
    assert "健身" in activities  # 不与新安排冲突，保留
    assert "做饭" not in activities  # 与新安排时间重叠，被替换
    assert "一起看书" in activities


def test_activity_hint_matches_common_activities() -> None:
    assert life_mod.activity_hint("一起看电影") == ("放松", 0)
    assert life_mod.activity_hint("一起看书") == ("平静", -1)
    assert life_mod.activity_hint("吃晚饭") == ("满足", 5)
    assert life_mod.activity_hint("跑步") == ("有活力", -4)
    assert life_mod.activity_hint("随便发会儿呆") is None


def test_advance_state_updates_mood_when_segment_changes() -> None:
    now = datetime(2026, 1, 1, 16, 10)
    plan = life_mod.normalize_plan(
        [{"start": "16:00", "end": "17:00", "activity": "一起看电影"}]
    )
    life = {
        "energy": 60,
        "energy_at": now.timestamp(),
        "plan": plan,
        "mood": "平静",
        "status": "",
        "manual_status": "",
        "manual_until": 0.0,
        "segment_key": "",
    }
    advanced = life_mod.advance_state(life, now)
    assert advanced["mood"] == "放松"
    assert "电影" in advanced["status"]


def test_manual_status_blocks_activity_mood_override() -> None:
    now = datetime(2026, 1, 1, 16, 10)
    plan = life_mod.normalize_plan(
        [{"start": "16:00", "end": "17:00", "activity": "跑步"}]
    )
    life = {
        "energy": 60,
        "energy_at": now.timestamp(),
        "plan": plan,
        "mood": "轻快",
        "status": "刚和你聊完",
        "manual_status": "刚和你聊完",
        "manual_until": now.timestamp() + 600,
        "segment_key": "",
    }
    advanced = life_mod.advance_state(life, now)
    assert advanced["mood"] == "轻快"
    assert advanced["status"] == "刚和你聊完"

# ── usage accounting ─────────────────────────────────────────────────────


def test_record_usage_accumulates_and_prunes() -> None:
    state = store_mod.empty_state("A")
    store_mod.record_usage(
        state, "2026-01-01", "reconcile", input_tokens=100, output_tokens=10
    )
    store_mod.record_usage(
        state,
        "2026-01-01",
        "reconcile",
        input_tokens=50,
        cached_tokens=5,
        output_tokens=20,
    )
    entry = state["usage"]["2026-01-01"]["reconcile"]
    assert entry == {"calls": 2, "input": 150, "cached": 5, "output": 30}

    for day in range(1, 20):
        store_mod.record_usage(state, f"2026-02-{day:02d}", "proactive")
    assert len(state["usage"]) == store_mod.USAGE_RETENTION_DAYS


def test_usage_summary_aggregates_personas() -> None:
    now = datetime(2026, 1, 2, 12, 0)
    first = store_mod.empty_state("A")
    second = store_mod.empty_state("B")
    store_mod.record_usage(
        first, "2026-01-02", "reconcile", input_tokens=100, output_tokens=10
    )
    store_mod.record_usage(
        second, "2026-01-02", "reconcile", input_tokens=200, output_tokens=20
    )
    store_mod.record_usage(first, "2026-01-02", "proactive", input_tokens=50)
    store_mod.record_usage(first, "2025-12-01", "proactive")

    summary = panel_mod.usage_summary({"A": first, "B": second}, now, days=3)

    assert summary["today"] == "2026-01-02"
    tasks = {item["task"]: item for item in summary["tasks"]}
    assert tasks["reconcile"]["calls"] == 2
    assert tasks["reconcile"]["total"] == 330
    assert tasks["proactive"]["calls"] == 1
    assert summary["today_total"]["calls"] == 3
    assert len(summary["trend"]) == 3
    assert summary["trend"][-1]["calls"] == 3
    assert all(item["calls"] == 0 for item in summary["trend"][:-1])


def test_overview_payload_includes_usage_and_interval() -> None:
    now = datetime(2026, 1, 1, 9, 30)
    payload = panel_mod.overview_payload(
        _panel_settings(), now, [], usage={"today_total": {"calls": 4}}
    )
    assert payload["usage"]["today_total"]["calls"] == 4
    assert payload["min_interval_minutes"] == 90
    assert payload["daily_limit"] == 6

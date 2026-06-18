import asyncio
import json
import logging
import random
import re
from calendar import monthrange
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import EventMessageType
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

from .db_manager import MemoryDBManager
from .llm_router import PluginLLMRouter


@register(
    "agentic_memory",
    "YourName",
    "3.0",
    "具备长期折叠记忆、即时反应与多模型路由的群聊智能体",
)
class AgenticMemoryPlugin(Star):
    """Group companion plugin with memory, immediate reactions, and role-based LLM routing."""

    def __init__(self, context: Context):
        """Initialize plugin state and runtime services.

        Args:
            context: AstrBot plugin context.
        """
        super().__init__(context)
        self.plugin_dir = Path(__file__).resolve().parent
        self.config = self._load_config(self.plugin_dir / "config.yaml")

        self.group_scope = self.config.get("group_scope", {})
        self.skill_settings = self.config.get("skill_settings", {})
        self.reaction_settings = self.config.get("message_reaction_settings", {})
        self.memory_settings = self.config.get("memory_settings", {})
        self.probability_settings = self.config.get("probability_settings", {})
        self.summary_settings = self.config.get("summary_settings", {})
        self.memory_lookup_settings = self.config.get("memory_lookup_settings", {})
        self.prompt_settings = self.config.get("prompt_settings", {})
        self.reply_dedup_settings = self.config.get("reply_dedup", {})
        self.special_reply_settings = self.config.get("special_replies", {})
        self.proactive_talk_settings = self.config.get("proactive_talk_settings", {})
        self.logging_settings = self.config.get("logging_settings", {})
        self.bot_names = [
            str(name).strip()
            for name in self.skill_settings.get("bot_names", [])
            if str(name).strip()
        ]
        self.direct_wake_question_patterns = [
            str(pattern)
            for pattern in self.skill_settings.get("direct_wake_question_patterns", [])
            if str(pattern).strip()
        ]

        self.threshold = max(1, int(self.memory_settings.get("buffer_threshold", 12)))
        self.overlap = max(0, int(self.memory_settings.get("overlap_count", 3)))
        self.max_dynamic_events = max(
            1,
            int(self.memory_settings.get("max_dynamic_events", 10)),
        )
        self.dynamic_event_merge_target_length = max(
            5,
            int(self.memory_settings.get("dynamic_event_merge_target_length", 20)),
        )
        self.recent_context_messages_for_interject = max(
            1,
            int(self.memory_settings.get("recent_context_messages_for_interject", 12)),
        )

        self.immediate_context_messages = max(
            1,
            int(self.reaction_settings.get("immediate_context_messages", 12)),
        )
        self.immediate_summary_count = max(
            1,
            int(self.reaction_settings.get("immediate_summary_count", 3)),
        )
        self.immediate_cooldown_seconds = max(
            0,
            int(self.reaction_settings.get("immediate_cooldown_seconds", 15)),
        )
        self.proactive_cooldown_seconds = max(
            0,
            int(self.reaction_settings.get("proactive_cooldown_seconds", 300)),
        )
        self.scheduled_proactive_cooldown_seconds = max(
            0,
            int(self.proactive_talk_settings.get("cooldown_seconds", 0)),
        )
        self.scheduled_proactive_failure_retry_seconds = max(
            30,
            int(self.proactive_talk_settings.get("failure_retry_seconds", 1800)),
        )
        self.short_window_enabled = bool(
            self.reaction_settings.get("short_window_enabled", True),
        )
        self.short_window_size = max(
            1,
            int(self.reaction_settings.get("short_window_size", 6)),
        )
        self.short_window_name_hit_min_count = max(
            1,
            int(self.reaction_settings.get("short_window_name_hit_min_count", 2)),
        )
        self.short_window_mention_hit_min_count = max(
            1,
            int(self.reaction_settings.get("short_window_mention_hit_min_count", 1)),
        )
        self.short_window_question_hit_min_count = max(
            1,
            int(self.reaction_settings.get("short_window_question_hit_min_count", 2)),
        )
        self.short_window_cooldown_seconds = max(
            0,
            int(self.reaction_settings.get("short_window_cooldown_seconds", 120)),
        )

        self.base_interject_probability = float(
            self.probability_settings.get("base_interject_probability", 0.15),
        )
        self.boost_interject_probability = float(
            self.probability_settings.get("boost_interject_probability", 0.6),
        )

        self.memory_lookup_enabled = bool(
            self.memory_lookup_settings.get("enabled", True),
        )
        self.memory_time_recall_enabled = bool(
            self.memory_lookup_settings.get("time_recall_enabled", True),
        )
        self.memory_event_recall_enabled = bool(
            self.memory_lookup_settings.get("event_recall_enabled", True),
        )
        self.memory_recent_paragraph_limit = max(
            1,
            int(self.memory_lookup_settings.get("recent_paragraph_limit", 5)),
        )
        self.memory_lookup_daily_limit = max(
            1,
            int(self.memory_lookup_settings.get("daily_limit", 3)),
        )
        self.memory_lookup_month_limit = max(
            1,
            int(self.memory_lookup_settings.get("month_limit", 2)),
        )
        self.memory_lookup_year_limit = max(
            1,
            int(self.memory_lookup_settings.get("year_limit", 2)),
        )
        self.memory_lookup_history_limit = max(
            1,
            int(self.memory_lookup_settings.get("history_limit", 2)),
        )
        self.memory_lookup_paragraph_limit = max(
            1,
            int(self.memory_lookup_settings.get("paragraph_limit", 3)),
        )
        self.memory_lookup_neighbor_limit = max(
            0,
            int(self.memory_lookup_settings.get("neighbor_limit", 3)),
        )
        self.memory_lookup_neighbor_window_days = max(
            1,
            int(self.memory_lookup_settings.get("neighbor_window_days", 45)),
        )
        self.memory_lookup_keyword_limit = max(
            1,
            int(self.memory_lookup_settings.get("keyword_limit", 4)),
        )
        self.memory_lookup_excerpt_max_chars = max(
            30,
            int(self.memory_lookup_settings.get("excerpt_max_chars", 120)),
        )

        rollup_hour, rollup_minute = self._parse_hhmm_time(
            str(self.summary_settings.get("rollup_time", "04:00")),
            default_hour=4,
            default_minute=0,
        )
        self.rollup_hour = rollup_hour
        self.rollup_minute = rollup_minute
        self.paragraph_cleanup_delay_days = max(
            0,
            int(self.summary_settings.get("paragraph_cleanup_delay_days", 1)),
        )
        self.daily_cleanup_delay_months = max(
            0,
            int(self.summary_settings.get("daily_cleanup_delay_months", 1)),
        )
        self.month_cleanup_delay_years = max(
            0,
            int(self.summary_settings.get("month_cleanup_delay_years", 1)),
        )
        self.year_retention_years = max(
            1,
            int(self.summary_settings.get("year_retention_years", 3)),
        )
        self.year_cleanup_delay_years = max(
            0,
            int(self.summary_settings.get("year_cleanup_delay_years", 1)),
        )

        self.dedup_enabled = bool(self.reply_dedup_settings.get("enabled", False))
        self.dedup_window_size = max(
            1,
            int(self.reply_dedup_settings.get("window_size", 8)),
        )
        self.dedup_only_proactive = bool(
            self.reply_dedup_settings.get("only_proactive", True),
        )
        self.dedup_retry_once = bool(
            self.reply_dedup_settings.get("retry_once_on_duplicate", False),
        )

        self.file_logging_enabled = bool(
            self.logging_settings.get("file_logging_enabled", True),
        )
        self.raw_event_debug_enabled = bool(
            self.logging_settings.get("raw_event_debug_enabled", False),
        )
        self.raw_event_message_chain_enabled = bool(
            self.logging_settings.get("raw_event_message_chain_enabled", True),
        )
        self.raw_event_max_chars = max(
            200,
            int(self.logging_settings.get("raw_event_max_chars", 4000)),
        )
        self.plugin_file_logger = self._setup_plugin_file_logger()

        self.skill_content = self._load_skill(
            self.skill_settings.get("active_skill_file", ""),
        )
        self.router = PluginLLMRouter(self.context, self.config.get("llm_settings", {}))
        self.db = MemoryDBManager(str(self._resolve_db_path()))

        self.message_buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.short_windows: dict[str, deque[dict[str, Any]]] = {}
        self.reply_history: dict[str, deque[str]] = {}
        self.cooldowns: dict[str, dict[str, datetime]] = defaultdict(dict)
        self.proactive_task_last_run_dates: dict[str, dict[str, str]] = defaultdict(
            dict
        )
        self.proactive_task_planned_times: dict[str, dict[str, datetime]] = defaultdict(
            dict
        )

        self._log(
            "info",
            "[agentic_memory] Plugin initialized. "
            f"file_logging_enabled={self.file_logging_enabled}, "
            f"raw_event_debug_enabled={self.raw_event_debug_enabled}, "
            f"log_file={self._resolve_plugin_log_path()}",
        )

        asyncio.create_task(self._cron_daily_compression())
        if self.proactive_talk_settings.get("enabled", False):
            asyncio.create_task(self._cron_proactive_talk())

    def _resolve_plugin_log_path(self) -> Path:
        """Resolve the plugin-local log file path.

        Returns:
            Absolute log file path inside AstrBot plugin data when a relative path is used.
        """
        configured_path = Path(
            str(
                self.logging_settings.get(
                    "log_file_path",
                    "astrbot_plugin_01/agentic_memory_plugin.log",
                )
            ),
        )
        if configured_path.is_absolute():
            log_path = configured_path
        else:
            log_path = Path(get_astrbot_plugin_data_path()) / configured_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return log_path

    def _setup_plugin_file_logger(self) -> logging.Logger | None:
        """Create the plugin-local file logger when enabled.

        Returns:
            Configured ``logging.Logger`` instance or ``None`` when disabled.
        """
        if not self.file_logging_enabled:
            return None

        log_path = self._resolve_plugin_log_path()
        plugin_logger = logging.getLogger(f"astrbot_plugin_01.file.{id(self)}")
        plugin_logger.setLevel(logging.INFO)
        plugin_logger.propagate = False
        plugin_logger.handlers.clear()

        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        plugin_logger.addHandler(handler)
        return plugin_logger

    def _log(self, level: str, message: str) -> None:
        """Write plugin logs to AstrBot logger and optional plugin file logger.

        Args:
            level: Logging level name such as ``info`` or ``error``.
            message: Final log message.
        """
        log_func = getattr(logger, level, None)
        if callable(log_func):
            log_func(message)

        file_log_func = getattr(self.plugin_file_logger, level, None)
        if callable(file_log_func):
            file_log_func(message)

    def _truncate_debug_text(self, value: Any) -> str:
        """Trim long debug text to the configured maximum length.

        Args:
            value: Original debug value.

        Returns:
            Possibly truncated debug text.
        """
        normalized = str(value)
        if len(normalized) <= self.raw_event_max_chars:
            return normalized
        return normalized[: self.raw_event_max_chars] + "...(truncated)"

    def _serialize_message_segment(self, segment: Any) -> dict[str, Any]:
        """Convert one message segment into a JSON-friendly structure.

        Args:
            segment: Message segment object from the event adapter.

        Returns:
            Dictionary that captures common fields for debugging.
        """
        serialized = {
            "repr": self._truncate_debug_text(repr(segment)),
            "type": str(getattr(segment, "type", "")),
        }

        for attr_name in ("qq", "text", "id", "file", "url"):
            attr_value = getattr(segment, attr_name, None)
            if attr_value is not None:
                serialized[attr_name] = self._truncate_debug_text(attr_value)

        segment_data = getattr(segment, "data", None)
        if segment_data is not None:
            serialized["data"] = self._truncate_debug_text(segment_data)

        to_dict = getattr(segment, "toDict", None)
        if callable(to_dict):
            try:
                serialized["toDict"] = self._truncate_debug_text(to_dict())
            except Exception as exc:
                serialized["toDict_error"] = str(exc)

        return serialized

    def _log_raw_event_debug(
        self,
        event: AstrMessageEvent,
        group_id: str,
        message_text: str,
    ) -> None:
        """Write raw incoming event structure for troubleshooting mention parsing.

        Args:
            event: Incoming group event.
            group_id: Current group identifier.
            message_text: Parsed plain message text.
        """
        if not self.raw_event_debug_enabled:
            return

        message_obj = getattr(event, "message_obj", None)
        payload = {
            "group_id": group_id,
            "session_id": str(getattr(event, "session_id", "")),
            "self_id": str(event.get_self_id()).strip(),
            "sender_name": str(event.get_sender_name()).strip(),
            "message_str": self._truncate_debug_text(getattr(event, "message_str", "")),
            "parsed_message_text": self._truncate_debug_text(message_text),
            "event_is_at": bool(getattr(event, "is_at", False)),
            "message_obj_type": type(message_obj).__name__ if message_obj else "",
            "message_obj_group_id": str(getattr(message_obj, "group_id", "")),
            "message_obj_sender_id": str(getattr(message_obj, "sender_id", "")),
            "message_obj_user_id": str(getattr(message_obj, "user_id", "")),
            "message_obj_message": self._truncate_debug_text(
                getattr(message_obj, "message", ""),
            ),
            "message_obj_raw_message": self._truncate_debug_text(
                getattr(message_obj, "raw_message", ""),
            ),
            "message_obj_repr": self._truncate_debug_text(repr(message_obj)),
        }

        if self.raw_event_message_chain_enabled:
            try:
                payload["message_chain"] = [
                    self._serialize_message_segment(segment)
                    for segment in event.get_messages()
                ]
            except Exception as exc:
                payload["message_chain_error"] = str(exc)

        self._log(
            "info",
            "[agentic_memory][raw_event] "
            + json.dumps(payload, ensure_ascii=False, default=str),
        )

    def _load_config(self, path: Path) -> dict[str, Any]:
        """Load plugin configuration.

        Args:
            path: Config file path.

        Returns:
            Parsed YAML config.
        """
        if not path.exists():
            logger.error(f"Config file is missing: {path}")
            return {}
        with path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        if not isinstance(loaded, dict):
            logger.error("Config file root must be a mapping.")
            return {}
        return loaded

    def _load_skill(self, skill_path: str) -> str:
        """Load the active skill prompt file.

        Args:
            skill_path: Relative or absolute skill file path.

        Returns:
            Skill file content.
        """
        if not skill_path:
            return "You are a natural, casual group member."
        skill_file = Path(skill_path)
        if not skill_file.is_absolute():
            skill_file = self.plugin_dir / skill_path
        if not skill_file.exists():
            self._log("warning", f"Skill file was not found: {skill_file}")
            return "You are a natural, casual group member."
        return skill_file.read_text(encoding="utf-8")

    def _parse_hhmm_time(
        self,
        value: str,
        default_hour: int,
        default_minute: int,
    ) -> tuple[int, int]:
        """Parse ``HH:MM`` text and fall back to defaults when invalid.

        Args:
            value: Time text from config.
            default_hour: Fallback hour.
            default_minute: Fallback minute.

        Returns:
            Parsed hour and minute tuple.
        """
        normalized = str(value).strip()
        if re.fullmatch(r"\d{2}:\d{2}", normalized):
            hour_text, minute_text = normalized.split(":")
            hour = int(hour_text)
            minute = int(minute_text)
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return hour, minute

        self._log(
            "warning",
            "[agentic_memory] Invalid HH:MM config value, fallback to default. "
            f"value={normalized!r}, default={default_hour:02d}:{default_minute:02d}",
        )
        return default_hour, default_minute

    def _resolve_db_path(self) -> Path:
        """Resolve the SQLite database path.

        Returns:
            Absolute database path inside AstrBot plugin data when a relative path is used.
        """
        configured_path = Path(
            str(
                self.memory_settings.get(
                    "memory_db_path", "astrbot_plugin_01/agentic_memory.db"
                )
            ),
        )
        if configured_path.is_absolute():
            db_path = configured_path
        else:
            db_path = Path(get_astrbot_plugin_data_path()) / configured_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return db_path

    def _is_group_allowed(self, group_id: str) -> bool:
        """Check whether the plugin should run in a group.

        Args:
            group_id: Group identifier.

        Returns:
            ``True`` when the group is allowed.
        """
        if not self.config.get("plugin_settings", {}).get("enabled", True):
            return False
        if not self.group_scope.get("enabled", True):
            return True
        whitelist = {
            str(item).strip()
            for item in self.group_scope.get("group_whitelist", [])
            if str(item).strip()
        }
        return group_id in whitelist if whitelist else False

    def _get_short_window(self, group_id: str) -> deque[dict[str, Any]]:
        """Get or create the short reaction window for a group.

        Args:
            group_id: Group identifier.

        Returns:
            Group-specific short reaction deque.
        """
        if group_id not in self.short_windows:
            self.short_windows[group_id] = deque(maxlen=self.short_window_size)
        return self.short_windows[group_id]

    def _get_reply_history(self, group_id: str) -> deque[str]:
        """Get or create reply history for deduplication.

        Args:
            group_id: Group identifier.

        Returns:
            Group-specific normalized reply deque.
        """
        if group_id not in self.reply_history:
            self.reply_history[group_id] = deque(maxlen=self.dedup_window_size)
        return self.reply_history[group_id]

    def _normalize_text(self, text: str) -> str:
        """Normalize reply text for lightweight deduplication.

        Args:
            text: Original text.

        Returns:
            Normalized comparison key.
        """
        lowered = text.lower().strip()
        lowered = re.sub(r"\s+", "", lowered)
        lowered = re.sub(r"[，。！？、,.!?\-~～…]+", "", lowered)
        return lowered

    def _is_duplicate_reply(self, group_id: str, reply_text: str, channel: str) -> bool:
        """Check whether a reply is duplicated under current dedup settings.

        Args:
            group_id: Group identifier.
            reply_text: Candidate reply text.
            channel: Reply channel such as ``immediate`` or ``proactive``.

        Returns:
            ``True`` when the reply should be considered duplicated.
        """
        if not self.dedup_enabled:
            return False
        if self.dedup_only_proactive and channel != "proactive":
            return False
        normalized = self._normalize_text(reply_text)
        return normalized in self._get_reply_history(group_id)

    def _record_reply_history(self, group_id: str, reply_text: str) -> None:
        """Record a reply after it has been sent.

        Args:
            group_id: Group identifier.
            reply_text: Sent reply text.
        """
        self._get_reply_history(group_id).append(self._normalize_text(reply_text))

    def _is_cooldown_ready(
        self, group_id: str, cooldown_key: str, seconds: int
    ) -> bool:
        """Check whether a cooldown window has elapsed.

        Args:
            group_id: Group identifier.
            cooldown_key: Logical cooldown key.
            seconds: Cooldown duration.

        Returns:
            ``True`` when sending is allowed.
        """
        if seconds <= 0:
            return True
        now = datetime.now()
        last_time = self.cooldowns[group_id].get(cooldown_key)
        return not last_time or (now - last_time).total_seconds() >= seconds

    def _mark_cooldown(self, group_id: str, cooldown_key: str) -> None:
        """Record the current time for a cooldown key.

        Args:
            group_id: Group identifier.
            cooldown_key: Logical cooldown key.
        """
        self.cooldowns[group_id][cooldown_key] = datetime.now()

    def _extract_is_mentioned(self, event: AstrMessageEvent) -> bool:
        """Detect whether the message directly @mentions the bot.

        Args:
            event: Incoming message event.

        Returns:
            ``True`` when the bot is directly mentioned.
        """
        if hasattr(event, "is_at") and getattr(event, "is_at"):
            return True

        bot_id = str(event.get_self_id()).strip()
        if not bot_id:
            self._log(
                "warning",
                "[agentic_memory] Missing bot self id when checking mention.",
            )
            return False

        try:
            message_chain = event.get_messages()
            for segment in message_chain:
                segment_type = str(getattr(segment, "type", "")).strip().lower()
                if segment_type != "at":
                    continue

                segment_qq = getattr(segment, "qq", None)
                if segment_qq is not None and str(segment_qq).strip() == bot_id:
                    return True

                segment_data = getattr(segment, "data", None)
                if (
                    isinstance(segment_data, dict)
                    and str(segment_data.get("qq", "")).strip() == bot_id
                ):
                    return True

                segment_dict = None
                to_dict = getattr(segment, "toDict", None)
                if callable(to_dict):
                    try:
                        segment_dict = to_dict()
                    except Exception:
                        segment_dict = None
                if (
                    isinstance(segment_dict, dict)
                    and str(segment_dict.get("data", {}).get("qq", "")).strip()
                    == bot_id
                ):
                    return True
        except Exception as exc:
            self._log("debug", f"Failed to inspect message chain mentions: {exc}")

        raw_candidates = [
            str(getattr(event, "message_str", "")).strip(),
            str(getattr(getattr(event, "message_obj", None), "message", "")).strip(),
            str(
                getattr(getattr(event, "message_obj", None), "raw_message", "")
            ).strip(),
        ]
        for raw_text in raw_candidates:
            if not raw_text:
                continue
            if f"@{bot_id}" in raw_text:
                return True
            if f"[At:{bot_id}]" in raw_text:
                return True
            if f"[CQ:at,qq={bot_id}]" in raw_text:
                return True
            if f"[CQ:at,qq={bot_id}," in raw_text:
                return True

        return False

    def _sanitize_reply_text(self, text: str) -> str:
        """Clean model output before sending it to the group.

        Args:
            text: Raw generated reply text.

        Returns:
            Sanitized reply text.
        """
        cleaned = str(text).strip()
        if not cleaned:
            return ""

        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        parsed_json = None
        try:
            parsed_json = json.loads(cleaned)
        except json.JSONDecodeError:
            parsed_json = None

        if isinstance(parsed_json, dict):
            for key in (
                "reply",
                "response",
                "answer",
                "output",
                "message",
                "content",
                "text",
            ):
                value = parsed_json.get(key)
                if isinstance(value, (str, int, float)):
                    cleaned = str(value).strip()
                    break
        elif isinstance(parsed_json, list):
            text_items = [
                str(item).strip() for item in parsed_json if str(item).strip()
            ]
            if len(text_items) == 1:
                cleaned = text_items[0]

        cleaned = re.sub(
            r"^(reply|response|answer|output|最终回复|回复|回答|输出)\s*[:：]\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

        wrappers = [
            ('"', '"'),
            ("'", "'"),
            ("“", "”"),
            ("‘", "’"),
            ("「", "」"),
            ("『", "』"),
            ("《", "》"),
            ("【", "】"),
            ("[", "]"),
            ("(", ")"),
        ]
        changed = True
        while cleaned and changed:
            changed = False
            for left, right in wrappers:
                if cleaned.startswith(left) and cleaned.endswith(right):
                    inner = cleaned[len(left) : len(cleaned) - len(right)].strip()
                    if inner:
                        cleaned = inner
                        changed = True

        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"^[\s,，.。!?！？;；:：~～\-]+", "", cleaned)
        cleaned = re.sub(r"[\s,，.。!?！？;；:：~～\-]+$", "", cleaned).strip()

        if cleaned.lower() in {"null", "none", "nil", "n/a", "[]", "{}", '""', "''"}:
            return ""

        return cleaned

    def _build_trigger_state(
        self,
        event: AstrMessageEvent,
        message_text: str,
    ) -> dict[str, Any]:
        """Build direct trigger signals from a new group message.

        Args:
            event: Incoming message event.
            message_text: Plain message text.

        Returns:
            Trigger state dictionary used by immediate and short-window logic.
        """
        lowered = message_text.lower()
        name_hit = any(name.lower() in lowered for name in self.bot_names if name)
        pattern_hit = any(
            re.search(pattern, message_text, flags=re.IGNORECASE)
            for pattern in self.direct_wake_question_patterns
        )
        question_mark = "?" in message_text or "？" in message_text
        return {
            "message_text": message_text,
            "is_mentioned": self._extract_is_mentioned(event),
            "name_hit": name_hit,
            "question_pattern_hit": pattern_hit or (name_hit and question_mark),
            "question_mark": question_mark,
        }

    def _append_message(
        self,
        group_id: str,
        sender_name: str,
        message_text: str,
        trigger_state: dict[str, Any],
    ) -> None:
        """Append a message to both the long buffer and short reaction window.

        Args:
            group_id: Group identifier.
            sender_name: Sender nickname.
            message_text: Plain message text.
            trigger_state: Precomputed trigger state.
        """
        message_item = {
            "sender": sender_name,
            "msg": message_text,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "is_mentioned": trigger_state["is_mentioned"],
            "name_hit": trigger_state["name_hit"],
            "question_pattern_hit": trigger_state["question_pattern_hit"],
        }
        self.message_buffers[group_id].append(message_item)
        self._get_short_window(group_id).append(message_item)

    def _pop_long_window_batch(self, group_id: str) -> list[dict[str, Any]] | None:
        """Pop a long-window batch when the configured threshold is reached.

        Args:
            group_id: Group identifier.

        Returns:
            Copied batch for background analysis, or ``None`` when not ready.
        """
        buffer = self.message_buffers[group_id]
        if len(buffer) < self.threshold:
            return None
        chat_batch = buffer.copy()
        kept_count = min(self.overlap, len(chat_batch))
        self.message_buffers[group_id] = chat_batch[-kept_count:] if kept_count else []
        return chat_batch

    def _match_special_reply(
        self,
        group_id: str,
        trigger_state: dict[str, Any],
    ) -> str:
        """Find a configured special reply that matches the current trigger state.

        Args:
            group_id: Group identifier.
            trigger_state: Trigger state dictionary.

        Returns:
            Selected canned reply or an empty string.
        """
        if not self.special_reply_settings.get("enabled", True):
            return ""

        rules = self.special_reply_settings.get("rules", [])
        ordered_rules = sorted(
            (rule for rule in rules if isinstance(rule, dict)),
            key=lambda item: int(item.get("priority", 0)),
            reverse=True,
        )
        direct_wake_hit = (
            trigger_state["is_mentioned"]
            or trigger_state["name_hit"]
            or trigger_state["question_pattern_hit"]
        )

        for rule in ordered_rules:
            if not rule.get("enabled", True):
                continue
            scope = str(rule.get("scope", "whitelist_only"))
            if scope == "all_groups" and not self.group_scope.get("enabled", True):
                pass
            elif scope not in {"all_groups", "whitelist_only"}:
                continue

            patterns = [
                str(pattern)
                for pattern in rule.get("patterns", [])
                if str(pattern).strip()
            ]
            if patterns and not any(
                re.search(pattern, trigger_state["message_text"], flags=re.IGNORECASE)
                for pattern in patterns
            ):
                continue

            if (
                rule.get("required_mention", False)
                and not trigger_state["is_mentioned"]
            ):
                continue
            if rule.get("required_name_hit", False) and not trigger_state["name_hit"]:
                continue
            if (
                rule.get("required_question_mark", False)
                and not trigger_state["question_mark"]
            ):
                continue

            trigger_mode = str(rule.get("trigger_mode", "any"))
            if trigger_mode == "direct_wake_only" and not direct_wake_hit:
                continue

            cooldown_seconds = max(0, int(rule.get("cooldown_seconds", 0)))
            cooldown_key = f"special:{rule.get('rule_id', 'unknown')}"
            if not self._is_cooldown_ready(group_id, cooldown_key, cooldown_seconds):
                continue

            candidates = [
                str(item).strip()
                for item in rule.get("candidate_replies", [])
                if str(item).strip()
            ]
            if not candidates:
                continue
            self._mark_cooldown(group_id, cooldown_key)
            return random.choice(candidates)

        return ""

    def _should_trigger_short_window(self, group_id: str) -> bool:
        """Decide whether the short reaction window should trigger a reply.

        Args:
            group_id: Group identifier.

        Returns:
            ``True`` when the short window should trigger.
        """
        if not self.short_window_enabled:
            return False
        window = list(self._get_short_window(group_id))
        if len(window) < self.short_window_size:
            return False

        name_hits = sum(1 for item in window if item.get("name_hit"))
        mention_hits = sum(1 for item in window if item.get("is_mentioned"))
        question_hits = sum(1 for item in window if item.get("question_pattern_hit"))
        return (
            name_hits >= self.short_window_name_hit_min_count
            or mention_hits >= self.short_window_mention_hit_min_count
            or question_hits >= self.short_window_question_hit_min_count
        )

    def _extract_balanced_json_fragment(self, text: str) -> str:
        """Extract the first balanced JSON object fragment from text.

        Args:
            text: Raw text that may contain JSON.

        Returns:
            Balanced JSON object text or an empty string.
        """
        for start_index, char in enumerate(text):
            if char != "{":
                continue

            depth = 0
            in_string = False
            escaped = False
            for index in range(start_index, len(text)):
                current = text[index]
                if escaped:
                    escaped = False
                    continue
                if current == "\\" and in_string:
                    escaped = True
                    continue
                if current == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if current == "{":
                    depth += 1
                elif current == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start_index : index + 1]

        return ""

    def _parse_json_response(
        self,
        response_text: str,
        log_failure: bool = True,
    ) -> dict[str, Any]:
        """Parse model JSON output as safely as possible.

        Args:
            response_text: Raw model text.
            log_failure: Whether to emit warning logs when parsing still fails.

        Returns:
            Parsed dictionary or an empty dict.
        """
        cleaned = response_text.strip()
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        if not cleaned:
            return {}

        try:
            loaded = json.loads(cleaned)
            return loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError as exc:
            candidate = self._extract_balanced_json_fragment(cleaned)
            if not candidate:
                start = cleaned.find("{")
                end = cleaned.rfind("}")
                if start != -1 and end != -1 and end > start:
                    candidate = cleaned[start : end + 1]
                elif start != -1:
                    candidate = cleaned[start:]
                else:
                    candidate = cleaned

            repair_candidates: list[str] = []
            for raw_candidate in (candidate, cleaned):
                normalized = raw_candidate.strip()
                if not normalized or normalized in repair_candidates:
                    continue

                repair_candidates.append(normalized)

                trimmed_trailing_commas = re.sub(r",(\s*[}\]])", r"\1", normalized)
                if trimmed_trailing_commas not in repair_candidates:
                    repair_candidates.append(trimmed_trailing_commas)

                brace_delta = normalized.count("{") - normalized.count("}")
                bracket_delta = normalized.count("[") - normalized.count("]")
                if brace_delta > 0 or bracket_delta > 0:
                    repaired_by_suffix = (
                        normalized
                        + ("]" * max(0, bracket_delta))
                        + ("}" * max(0, brace_delta))
                    )
                    if repaired_by_suffix not in repair_candidates:
                        repair_candidates.append(repaired_by_suffix)

                    repaired_trimmed = re.sub(
                        r",(\s*[}\]])",
                        r"\1",
                        repaired_by_suffix,
                    )
                    if repaired_trimmed not in repair_candidates:
                        repair_candidates.append(repaired_trimmed)

            for repaired_text in repair_candidates:
                try:
                    loaded = json.loads(repaired_text)
                    if isinstance(loaded, dict):
                        if repaired_text != cleaned:
                            self._log(
                                "info",
                                "[agentic_memory] Recovered JSON response with light repair. "
                                f"original_length={len(cleaned)}, repaired_length={len(repaired_text)}",
                            )
                        return loaded
                except json.JSONDecodeError:
                    continue

            if log_failure:
                brace_delta = candidate.count("{") - candidate.count("}")
                bracket_delta = candidate.count("[") - candidate.count("]")
                self._log(
                    "warning",
                    "[agentic_memory] Failed to parse JSON response. "
                    f"error={exc}; brace_delta={brace_delta}; bracket_delta={bracket_delta}; "
                    f"length={len(cleaned)}; head={cleaned[:200]!r}; tail={cleaned[-200:]!r}",
                )
            return {}

    async def _send_reply(
        self,
        event: AstrMessageEvent,
        group_id: str,
        reply_text: str,
        channel: str,
        cooldown_key: str | None = None,
    ) -> bool:
        """Send a reply and record dedup/cooldown state.

        Args:
            event: Incoming message event.
            group_id: Group identifier.
            reply_text: Reply text.
            channel: Reply channel such as ``immediate`` or ``proactive``.
            cooldown_key: Optional cooldown key to mark after sending.

        Returns:
            ``True`` when the message was sent successfully.
        """
        raw_reply_text = str(reply_text)
        reply_text = self._sanitize_reply_text(reply_text)
        if not reply_text:
            raw_reply_text = raw_reply_text.strip()
            if not raw_reply_text:
                self._log(
                    "info",
                    "[agentic_memory] Skip sending because reply generator returned empty text. "
                    f"group={group_id}, channel={channel}",
                )
                return False

            parsed_payload = self._parse_json_response(
                raw_reply_text,
                log_failure=False,
            )
            if parsed_payload:
                self._log(
                    "info",
                    "[agentic_memory] Skip sending because reply JSON had no usable text field after sanitize. "
                    f"group={group_id}, channel={channel}, keys={list(parsed_payload.keys())}",
                )
            else:
                self._log(
                    "warning",
                    "[agentic_memory] Empty reply after sanitize. "
                    f"group={group_id}, channel={channel}, raw={raw_reply_text[:200]!r}",
                )
            return False
        if self._is_duplicate_reply(group_id, reply_text, channel):
            self._log(
                "info", f"Skipped duplicated reply in group {group_id}: {reply_text}"
            )
            return False

        try:
            await event.send(event.plain_result(reply_text))
        except Exception as exc:
            self._log(
                "error",
                f"[agentic_memory] Failed to send reply. group={group_id}, channel={channel}, error={exc}",
            )
            return False

        self._record_reply_history(group_id, reply_text)
        if cooldown_key:
            self._mark_cooldown(group_id, cooldown_key)
        self._log("info", f"[{channel}] Sent reply in group {group_id}: {reply_text}")
        return True

    def _stop_event_flow(self, event: AstrMessageEvent) -> None:
        """Stop downstream event propagation when possible.

        Args:
            event: Incoming message event.
        """
        should_call_llm = getattr(event, "should_call_llm", None)
        if callable(should_call_llm):
            should_call_llm(True)

        stop_propagation = getattr(event, "stop_propagation", None)
        if callable(stop_propagation):
            stop_propagation()
            return

        stop_event = getattr(event, "stop_event", None)
        if callable(stop_event):
            stop_event()

    def _normalize_proactive_tasks(self) -> list[dict[str, Any]]:
        """Normalize scheduled proactive talk tasks from config.

        Returns:
            Enabled task definitions with validated ids, times, parsed clock parts,
            and messages.
        """
        if not self.proactive_talk_settings.get("enabled", False):
            return []

        tasks_value = self.proactive_talk_settings.get("tasks", [])
        if isinstance(tasks_value, list) and tasks_value:
            tasks = tasks_value
        else:
            legacy_time = str(self.proactive_talk_settings.get("time", "")).strip()
            legacy_range_start = str(
                self.proactive_talk_settings.get("time_range_start", "")
            ).strip()
            legacy_range_end = str(
                self.proactive_talk_settings.get("time_range_end", "")
            ).strip()
            legacy_messages = self.proactive_talk_settings.get(
                "messages",
                self.proactive_talk_settings.get("message", []),
            )
            if legacy_time or legacy_range_start or legacy_range_end or legacy_messages:
                tasks = [
                    {
                        "task_id": "default_task",
                        "enabled": True,
                        "time": legacy_time,
                        "time_range_start": legacy_range_start,
                        "time_range_end": legacy_range_end,
                        "messages": legacy_messages,
                        "cooldown_seconds": self.proactive_talk_settings.get(
                            "task_cooldown_seconds",
                            0,
                        ),
                        "group_ids": self.proactive_talk_settings.get(
                            "default_group_ids",
                            [],
                        ),
                    }
                ]
            else:
                tasks = []

        normalized_tasks: list[dict[str, Any]] = []

        for index, task in enumerate(tasks):
            if not isinstance(task, dict) or not task.get("enabled", True):
                continue

            task_id = (
                str(task.get("task_id", f"task_{index + 1}")).strip()
                or f"task_{index + 1}"
            )
            fixed_time = str(task.get("time", "")).strip()
            range_start = str(task.get("time_range_start", "")).strip()
            range_end = str(task.get("time_range_end", "")).strip()

            messages_value = task.get("messages", task.get("message", []))
            if isinstance(messages_value, str):
                messages = [messages_value.strip()] if messages_value.strip() else []
            elif isinstance(messages_value, list):
                messages = [
                    str(item).strip() for item in messages_value if str(item).strip()
                ]
            else:
                messages = []

            group_ids_value = task.get("group_ids", task.get("group_id", []))
            if isinstance(group_ids_value, str):
                group_ids = [group_ids_value.strip()] if group_ids_value.strip() else []
            elif isinstance(group_ids_value, list):
                group_ids = [
                    str(item).strip() for item in group_ids_value if str(item).strip()
                ]
            else:
                group_ids = []

            task_cooldown_seconds = max(0, int(task.get("cooldown_seconds", 0)))

            if not messages:
                self._log(
                    "warning",
                    f"[agentic_memory] Skip proactive task without messages: task_id={task_id}",
                )
                continue

            fixed_hour = None
            fixed_minute = None
            range_start_hour = None
            range_start_minute = None
            range_end_hour = None
            range_end_minute = None

            if fixed_time:
                if not re.fullmatch(r"\d{2}:\d{2}", fixed_time):
                    self._log(
                        "warning",
                        "[agentic_memory] Skip proactive task with invalid fixed time. "
                        f"task_id={task_id}, time={fixed_time}",
                    )
                    continue

                fixed_hour, fixed_minute = [int(part) for part in fixed_time.split(":")]
                if not (0 <= fixed_hour <= 23 and 0 <= fixed_minute <= 59):
                    self._log(
                        "warning",
                        "[agentic_memory] Skip proactive task with out-of-range fixed time. "
                        f"task_id={task_id}, time={fixed_time}",
                    )
                    continue
            else:
                if not (
                    re.fullmatch(r"\d{2}:\d{2}", range_start)
                    and re.fullmatch(r"\d{2}:\d{2}", range_end)
                ):
                    self._log(
                        "warning",
                        "[agentic_memory] Skip proactive task with invalid range. "
                        f"task_id={task_id}, start={range_start}, end={range_end}",
                    )
                    continue

                range_start_hour, range_start_minute = [
                    int(part) for part in range_start.split(":")
                ]
                range_end_hour, range_end_minute = [
                    int(part) for part in range_end.split(":")
                ]
                if not (
                    0 <= range_start_hour <= 23
                    and 0 <= range_start_minute <= 59
                    and 0 <= range_end_hour <= 23
                    and 0 <= range_end_minute <= 59
                ):
                    self._log(
                        "warning",
                        "[agentic_memory] Skip proactive task with out-of-range time window. "
                        f"task_id={task_id}, start={range_start}, end={range_end}",
                    )
                    continue

                if (range_end_hour, range_end_minute) < (
                    range_start_hour,
                    range_start_minute,
                ):
                    self._log(
                        "warning",
                        "[agentic_memory] Skip proactive task whose range end is earlier than start. "
                        f"task_id={task_id}, start={range_start}, end={range_end}",
                    )
                    continue

            normalized_tasks.append(
                {
                    "task_id": task_id,
                    "time": fixed_time,
                    "time_range_start": range_start,
                    "time_range_end": range_end,
                    "fixed_hour": fixed_hour,
                    "fixed_minute": fixed_minute,
                    "range_start_hour": range_start_hour,
                    "range_start_minute": range_start_minute,
                    "range_end_hour": range_end_hour,
                    "range_end_minute": range_end_minute,
                    "messages": messages,
                    "group_ids": group_ids,
                    "cooldown_seconds": task_cooldown_seconds,
                }
            )

        return normalized_tasks

    def _pick_proactive_task_time(
        self,
        task: dict[str, Any],
        target_day: datetime,
        earliest_time: datetime | None = None,
    ) -> datetime:
        """Pick one execution time for a proactive task on a specific date.

        Args:
            task: Normalized proactive task config.
            target_day: Datetime whose date is used for the target run day.
            earliest_time: Optional lower bound for random window scheduling.

        Returns:
            Target execution datetime on the requested day.
        """
        fixed_hour = task.get("fixed_hour")
        fixed_minute = task.get("fixed_minute")
        if fixed_hour is not None and fixed_minute is not None:
            return target_day.replace(
                hour=int(fixed_hour),
                minute=int(fixed_minute),
                second=0,
                microsecond=0,
            )

        start_dt = target_day.replace(
            hour=int(task["range_start_hour"]),
            minute=int(task["range_start_minute"]),
            second=0,
            microsecond=0,
        )
        end_dt = target_day.replace(
            hour=int(task["range_end_hour"]),
            minute=int(task["range_end_minute"]),
            second=0,
            microsecond=0,
        )
        if earliest_time and earliest_time.date() == target_day.date():
            earliest_candidate = (earliest_time + timedelta(minutes=1)).replace(
                second=0,
                microsecond=0,
            )
            start_dt = max(start_dt, earliest_candidate)

        random_seconds = random.randint(0, int((end_dt - start_dt).total_seconds()))
        return start_dt + timedelta(seconds=random_seconds)

    def _get_planned_proactive_task_time(
        self,
        group_id: str,
        task: dict[str, Any],
        now: datetime,
    ) -> datetime:
        """Get the next stable planned execution time for one task and group.

        Args:
            group_id: Target group id.
            task: Normalized proactive task config.
            now: Current datetime.

        Returns:
            Stable future execution time, or the already-planned overdue time when
            the task is waiting for a retry on the same day.
        """
        task_id = str(task["task_id"])
        planned_time = self.proactive_task_planned_times[group_id].get(task_id)
        if planned_time:
            if planned_time > now:
                return planned_time
            if planned_time.date() == now.date():
                return planned_time

        fixed_hour = task.get("fixed_hour")
        fixed_minute = task.get("fixed_minute")
        if fixed_hour is not None and fixed_minute is not None:
            planned_time = self._pick_proactive_task_time(task, now)
            if planned_time <= now:
                planned_time = self._pick_proactive_task_time(
                    task,
                    now + timedelta(days=1),
                )
        else:
            today_start = now.replace(
                hour=int(task["range_start_hour"]),
                minute=int(task["range_start_minute"]),
                second=0,
                microsecond=0,
            )
            today_end = now.replace(
                hour=int(task["range_end_hour"]),
                minute=int(task["range_end_minute"]),
                second=0,
                microsecond=0,
            )
            if now < today_start:
                planned_time = self._pick_proactive_task_time(task, now)
            elif now < today_end:
                planned_time = self._pick_proactive_task_time(
                    task,
                    now,
                    earliest_time=now,
                )
            else:
                planned_time = self._pick_proactive_task_time(
                    task,
                    now + timedelta(days=1),
                )

        self.proactive_task_planned_times[group_id][task_id] = planned_time
        self._log(
            "info",
            "[agentic_memory] Planned proactive task time. "
            f"group={group_id}, task_id={task_id}, planned_at={planned_time.isoformat(timespec='seconds')}",
        )
        return planned_time

    def _get_allowed_proactive_groups(self) -> list[str]:
        """Get the default groups that scheduled proactive tasks may send to.

        Returns:
            Default proactive target group id list from proactive config.
        """
        default_group_ids = self.proactive_talk_settings.get("default_group_ids", [])
        merged_groups: list[str] = []
        if not isinstance(default_group_ids, list):
            return merged_groups

        for item in default_group_ids:
            group_id = str(item).strip()
            if group_id and group_id not in merged_groups:
                merged_groups.append(group_id)

        return merged_groups

    async def _send_proactive_message_to_group(
        self,
        group_id: str,
        reply_text: str,
        task_id: str,
    ) -> str:
        """Send a scheduled proactive message to one group.

        Args:
            group_id: Target group id.
            reply_text: Message text to send.
            task_id: Proactive task identifier.

        Returns:
            Status string: ``sent``, ``skipped``, ``retryable_failure``, or
            ``fatal_failure``.
        """
        raw_reply_text = str(reply_text)
        reply_text = self._sanitize_reply_text(reply_text)
        if not reply_text:
            self._log(
                "warning",
                "[agentic_memory] Scheduled proactive reply is empty after sanitize. "
                f"group={group_id}, task_id={task_id}, raw={raw_reply_text[:200]!r}",
            )
            return "skipped"

        if self._is_duplicate_reply(group_id, reply_text, "proactive"):
            self._log(
                "info",
                "[agentic_memory] Scheduled proactive reply skipped by dedup. "
                f"group={group_id}, task_id={task_id}, reply={reply_text}",
            )
            return "skipped"

        try:
            await StarTools.send_message_by_id(
                "GroupMessage",
                group_id,
                MessageChain().message(reply_text),
            )
        except Exception as exc:
            error_text = str(exc)
            lowered_error = error_text.lower()
            fatal_error = any(
                marker in error_text
                for marker in (
                    "被移出该群",
                    "不在该群",
                    "群不存在",
                    "已退出该群",
                    "不是群成员",
                )
            ) or any(
                marker in lowered_error
                for marker in (
                    "not in group",
                    "group not found",
                    "removed from the group",
                    "removed from group",
                    "not a member",
                )
            )
            self._log(
                "error",
                "[agentic_memory] Failed to send scheduled proactive reply. "
                f"group={group_id}, task_id={task_id}, fatal={fatal_error}, error={exc}",
            )
            return "fatal_failure" if fatal_error else "retryable_failure"

        self._record_reply_history(group_id, reply_text)
        self._log(
            "info",
            "[scheduled_proactive] Sent reply. "
            f"group={group_id}, task_id={task_id}, reply={reply_text}",
        )
        return "sent"

    async def _run_due_proactive_tasks(self) -> None:
        """Run proactive talk tasks that are due right now."""
        tasks = self._normalize_proactive_tasks()
        if not tasks:
            return

        now = datetime.now()
        today_text = now.strftime("%Y-%m-%d")
        default_groups = self._get_allowed_proactive_groups()
        if not default_groups and not any(task.get("group_ids") for task in tasks):
            return

        for task in tasks:
            task_id = str(task["task_id"])
            task_groups = task.get("group_ids") or default_groups
            if not task_groups:
                continue

            task_cooldown_seconds = max(0, int(task.get("cooldown_seconds", 0)))
            task_cooldown_key = f"proactive_task:{task_id}"
            failure_retry_key = f"proactive_task_failure:{task_id}"

            for group_id in task_groups:
                if not self._is_group_allowed(group_id):
                    continue
                if (
                    self.proactive_task_last_run_dates[group_id].get(task_id)
                    == today_text
                ):
                    continue

                target_time = self._get_planned_proactive_task_time(group_id, task, now)
                if now < target_time:
                    continue
                if (
                    self.scheduled_proactive_cooldown_seconds > 0
                    and not self._is_cooldown_ready(
                        group_id,
                        "proactive",
                        self.scheduled_proactive_cooldown_seconds,
                    )
                ):
                    continue
                if task_cooldown_seconds > 0 and not self._is_cooldown_ready(
                    group_id,
                    task_cooldown_key,
                    task_cooldown_seconds,
                ):
                    continue
                if not self._is_cooldown_ready(
                    group_id,
                    failure_retry_key,
                    self.scheduled_proactive_failure_retry_seconds,
                ):
                    continue

                message_text = random.choice(task["messages"])
                send_status = await self._send_proactive_message_to_group(
                    group_id,
                    message_text,
                    task_id,
                )
                if send_status == "sent":
                    self.proactive_task_last_run_dates[group_id][task_id] = today_text
                    if self.scheduled_proactive_cooldown_seconds > 0:
                        self._mark_cooldown(group_id, "proactive")
                    if task_cooldown_seconds > 0:
                        self._mark_cooldown(group_id, task_cooldown_key)
                    continue

                if send_status == "retryable_failure":
                    self._mark_cooldown(group_id, failure_retry_key)
                    continue

                self.proactive_task_last_run_dates[group_id][task_id] = today_text
                self._log(
                    "info",
                    "[agentic_memory] Scheduled proactive task finished without sending and will not retry today. "
                    f"group={group_id}, task_id={task_id}, status={send_status}",
                )

    async def _cron_proactive_talk(self) -> None:
        """Run scheduled proactive talk tasks in the background."""
        self._log("info", "[agentic_memory] Proactive talk scheduler started.")
        while True:
            try:
                await self._run_due_proactive_tasks()
            except Exception as exc:
                self._log(
                    "error", f"[agentic_memory] Proactive talk scheduler failed: {exc}"
                )
            await asyncio.sleep(30)

    def _format_db_time(self, value: str | None) -> str:
        """Format one database timestamp into a more human-readable string.

        Args:
            value: Timestamp text from SQLite.

        Returns:
            Formatted timestamp text.
        """
        if not value:
            return "unknown time"
        parsed = self._parse_db_time(value)
        if not parsed:
            return value
        return parsed.strftime("%Y-%m-%d %H:%M")

    def _parse_db_time(self, value: str | None) -> datetime | None:
        """Parse one database timestamp string.

        Args:
            value: Timestamp text.

        Returns:
            Parsed datetime or ``None`` when parsing fails.
        """
        if not value:
            return None
        for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(value, pattern)
            except ValueError:
                continue
        return None

    def _shift_months(self, moment: datetime, months: int) -> datetime:
        """Shift one datetime by whole months while keeping a valid day.

        Args:
            moment: Base datetime.
            months: Month delta, may be negative.

        Returns:
            Shifted datetime.
        """
        month_index = moment.month - 1 + months
        year = moment.year + month_index // 12
        month = month_index % 12 + 1
        day = min(moment.day, monthrange(year, month)[1])
        return moment.replace(year=year, month=month, day=day)

    def _extract_memory_keywords(
        self, topic: str, recent_context: list[dict[str, Any]]
    ) -> list[str]:
        """Extract lightweight event keywords from topic and context.

        Args:
            topic: Current topic text.
            recent_context: Recent chat context.

        Returns:
            Deduplicated keyword list.
        """
        combined_text = " ".join(
            [topic] + [str(item.get("msg", "")) for item in recent_context]
        )
        raw_tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", combined_text)
        stop_words = {
            "今天",
            "昨天",
            "刚才",
            "刚刚",
            "去年",
            "上月",
            "上个月",
            "这个月",
            "那时候",
            "当时",
            "之前",
            "什么",
            "怎么",
            "大家",
            "我们",
            "你们",
            "他们",
            "一下",
            "一次",
            "这次",
        }
        keywords: list[str] = []
        for token in raw_tokens:
            cleaned = token.strip()
            if len(cleaned) < 2 or cleaned in stop_words:
                continue
            if cleaned not in keywords:
                keywords.append(cleaned)
            if len(keywords) >= self.memory_lookup_keyword_limit:
                break
        return keywords

    def _resolve_time_recall_window(
        self, topic: str, now: datetime
    ) -> tuple[str, datetime, datetime] | None:
        """Resolve fuzzy time expressions into a recall window.

        Args:
            topic: Current topic text.
            now: Current datetime.

        Returns:
            Tuple of description, start time, and end time, or ``None``.
        """
        text = topic.strip()
        if not text:
            return None

        if "刚才" in text or "刚刚" in text:
            start = now - timedelta(hours=2)
            return "刚才", start, now
        if "今天" in text:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return "今天", start, now
        if "昨天" in text:
            start = (now - timedelta(days=1)).replace(
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
            end = start + timedelta(days=1)
            return "昨天", start, end

        year_month_match = re.search(r"(\d{4})年\s*(\d{1,2})月", text)
        if year_month_match:
            year = int(year_month_match.group(1))
            month = int(year_month_match.group(2))
            if 1 <= month <= 12:
                start = datetime(year, month, 1, self.rollup_hour, self.rollup_minute)
                end = self._shift_months(start, 1)
                return f"{year}年{month}月", start, end

        last_year_month_match = re.search(r"去年\s*(\d{1,2})月", text)
        if last_year_month_match:
            month = int(last_year_month_match.group(1))
            if 1 <= month <= 12:
                year = now.year - 1
                start = datetime(year, month, 1, self.rollup_hour, self.rollup_minute)
                end = self._shift_months(start, 1)
                return f"去年{month}月", start, end

        if "上个月" in text or "上月" in text:
            current_month_start = now.replace(
                day=1,
                hour=self.rollup_hour,
                minute=self.rollup_minute,
                second=0,
                microsecond=0,
            )
            start = self._shift_months(current_month_start, -1)
            return "上个月", start, current_month_start

        if "这个月" in text or "本月" in text:
            start = now.replace(
                day=1,
                hour=self.rollup_hour,
                minute=self.rollup_minute,
                second=0,
                microsecond=0,
            )
            return "这个月", start, now

        year_match = re.search(r"(\d{4})年", text)
        if year_match:
            year = int(year_match.group(1))
            start = datetime(year, 1, 1, self.rollup_hour, self.rollup_minute)
            end = datetime(year + 1, 1, 1, self.rollup_hour, self.rollup_minute)
            return f"{year}年", start, end

        if "去年" in text:
            start = datetime(now.year - 1, 1, 1, self.rollup_hour, self.rollup_minute)
            end = datetime(now.year, 1, 1, self.rollup_hour, self.rollup_minute)
            return "去年", start, end

        if (
            "前几年" in text
            or "以前" in text
            or "之前" in text
            or "那时候" in text
            or "当时" in text
        ):
            start = datetime(
                max(1970, now.year - self.year_retention_years - 3),
                1,
                1,
                self.rollup_hour,
                self.rollup_minute,
            )
            return "更早之前", start, now

        return None

    def _row_anchor_times(self, row: tuple[Any, ...]) -> tuple[str | None, str | None]:
        """Resolve one summary row into anchor period bounds.

        Args:
            row: Summary row returned by the database layer.

        Returns:
            Tuple of start and end timestamp text.
        """
        start_time = None
        end_time = None
        if len(row) >= 8:
            start_time = row[6] or row[4]
            end_time = row[7] or row[6] or row[4]
        elif len(row) >= 5:
            start_time = row[2]
            end_time = row[2]
        return start_time, end_time

    def _summaries_to_lines(
        self, rows: list[tuple[Any, ...]], include_layer: bool = True
    ) -> list[str]:
        """Convert summary rows into prompt-friendly bullet lines.

        Args:
            rows: Summary rows.
            include_layer: Whether to include the memory layer name.

        Returns:
            Bullet line list.
        """
        lines: list[str] = []
        for row in rows:
            row_values = list(row)
            if len(row_values) >= 8:
                layer = str(row_values[2])
                content = str(row_values[3]).strip()
                created_at = str(row_values[4])
            elif len(row_values) >= 3:
                layer = "paragraph"
                content = str(row_values[1]).strip()
                created_at = str(row_values[2])
            else:
                continue
            if not content:
                continue
            prefix = f"[{layer}] " if include_layer else ""
            lines.append(
                f"- {prefix}{self._format_db_time(created_at)}: {content[: self.memory_lookup_excerpt_max_chars]}"
            )
        return lines

    def _build_memory_recall_block(
        self,
        group_id: str,
        topic: str,
        recent_context: list[dict[str, Any]],
    ) -> str:
        """Build a fuzzy human-like memory recall block for the reply prompt.

        Args:
            group_id: Group identifier.
            topic: Current topic text.
            recent_context: Recent messages.

        Returns:
            Recall block text.
        """
        if not self.memory_lookup_enabled:
            return "【Memory recall】\n- Disabled"

        recall_lines: list[str] = []
        now = datetime.now()
        used_ids: list[int] = []
        anchor_start_text = None
        anchor_end_text = None

        recent_paragraph_rows = self.db.get_summaries(
            group_id,
            "paragraph",
            limit=self.memory_recent_paragraph_limit,
            order_desc=True,
        )
        if recent_paragraph_rows:
            paragraph_lines = [
                f"- {self._format_db_time(row[2])}: {str(row[1]).strip()[: self.memory_lookup_excerpt_max_chars]}"
                for row in recent_paragraph_rows
                if str(row[1]).strip()
            ]
            if paragraph_lines:
                recall_lines.append("Recent paragraphs:")
                recall_lines.extend(paragraph_lines)

        if self.memory_time_recall_enabled:
            time_window = self._resolve_time_recall_window(topic, now)
            if time_window:
                time_label, start_time, end_time = time_window
                summary_types = [
                    ("daily", self.memory_lookup_daily_limit),
                    ("month", self.memory_lookup_month_limit),
                    ("year", self.memory_lookup_year_limit),
                    ("history", self.memory_lookup_history_limit),
                ]
                time_matches: list[tuple[Any, ...]] = []
                for summary_type, limit in summary_types:
                    rows = self.db.get_summaries(
                        group_id,
                        summary_type,
                        limit=max(limit, 1),
                        order_desc=True,
                    )
                    matched_rows = []
                    for row in rows:
                        row_start = self._parse_db_time(row[5] or row[2])
                        row_end = self._parse_db_time(row[6] or row[5] or row[2])
                        if not row_start or not row_end:
                            continue
                        if row_end >= start_time and row_start <= end_time:
                            matched_rows.append(
                                (
                                    row[0],
                                    group_id,
                                    summary_type,
                                    row[1],
                                    row[2],
                                    row[4],
                                    row[5],
                                    row[6],
                                    row[3],
                                    row[7],
                                    row[8],
                                    row[9],
                                )
                            )
                    time_matches.extend(matched_rows[:limit])

                if time_matches:
                    used_ids.extend(int(row[0]) for row in time_matches)
                    first_start, first_end = self._row_anchor_times(time_matches[0])
                    anchor_start_text = first_start
                    anchor_end_text = first_end
                    recall_lines.append(f"Time recall for {time_label}:")
                    recall_lines.extend(self._summaries_to_lines(time_matches))

        if self.memory_event_recall_enabled:
            keywords = self._extract_memory_keywords(topic, recent_context)
            if keywords:
                keyword_matches = self.db.search_summaries_by_keywords(
                    group_id,
                    ["paragraph", "daily", "month", "year", "history"],
                    keywords,
                    limit=max(
                        self.memory_lookup_daily_limit
                        + self.memory_lookup_month_limit
                        + self.memory_lookup_year_limit,
                        1,
                    ),
                )
                if keyword_matches:
                    fresh_matches = [
                        row for row in keyword_matches if int(row[0]) not in used_ids
                    ]
                    selected_matches = (
                        fresh_matches
                        or keyword_matches[: self.memory_lookup_keyword_limit]
                    )
                    used_ids.extend(int(row[0]) for row in selected_matches)
                    if not anchor_start_text or not anchor_end_text:
                        anchor_start_text, anchor_end_text = self._row_anchor_times(
                            selected_matches[0]
                        )
                    recall_lines.append(
                        f"Event recall by keywords ({', '.join(keywords)}):"
                    )
                    recall_lines.extend(self._summaries_to_lines(selected_matches))

        if (
            self.memory_lookup_neighbor_limit > 0
            and anchor_start_text
            and anchor_end_text
        ):
            anchor_start = self._parse_db_time(anchor_start_text)
            anchor_end = self._parse_db_time(anchor_end_text)
            if anchor_start and anchor_end:
                neighbor_start = (
                    anchor_start
                    - timedelta(days=self.memory_lookup_neighbor_window_days)
                ).strftime("%Y-%m-%d %H:%M:%S")
                neighbor_end = (
                    anchor_end + timedelta(days=self.memory_lookup_neighbor_window_days)
                ).strftime("%Y-%m-%d %H:%M:%S")
                neighbor_rows = self.db.get_neighbor_summaries(
                    group_id,
                    neighbor_start,
                    neighbor_end,
                    self.memory_lookup_neighbor_limit,
                    exclude_ids=used_ids,
                )
                if neighbor_rows:
                    recall_lines.append("Nearby memories from the same period:")
                    recall_lines.extend(self._summaries_to_lines(neighbor_rows))

        if not recall_lines:
            return "【Memory recall】\n- None"

        return "【Memory recall】\n" + "\n".join(recall_lines)

    async def _build_chat_reply(
        self,
        group_id: str,
        topic: str,
        recent_context: list[dict[str, Any]],
        channel: str,
    ) -> str:
        """Generate a chat-style reply using summaries, profiles, and current context.

        Args:
            group_id: Group identifier.
            topic: Trigger topic or direct message.
            recent_context: Recent context messages.
            channel: Reply channel.

        Returns:
            Generated reply text.
        """
        recent_summaries_rows = self.db.get_summaries(
            group_id,
            "paragraph",
            limit=max(self.immediate_summary_count, self.memory_recent_paragraph_limit),
            order_desc=True,
        )
        recent_summaries = [
            row[1]
            for row in reversed(recent_summaries_rows[: self.immediate_summary_count])
        ]
        if recent_summaries:
            background_str = "【Recent group background】\n- " + "\n- ".join(
                str(item).strip() for item in recent_summaries if str(item).strip()
            )
        else:
            background_str = "【Recent group background】\n- None"

        memory_recall_block = self._build_memory_recall_block(
            group_id,
            topic,
            recent_context,
        )

        involved_users = []
        seen_users = set()
        for item in recent_context:
            sender = str(item.get("sender", "")).strip()
            if sender and sender not in seen_users:
                seen_users.add(sender)
                involved_users.append(sender)

        archives_lines: list[str] = []
        for user_name in involved_users:
            user_data = self.db.get_user_profile(group_id, user_name)
            fixed_info = ", ".join(
                f"{key}:{value}"
                for key, value in user_data.get("fixed_data", {}).items()
            )
            dynamic_info = "; ".join(user_data.get("dynamic_events", []))
            if fixed_info or dynamic_info:
                archives_lines.append(
                    f"- {user_name} -> fixed=[{fixed_info}] dynamic=[{dynamic_info}]",
                )
        archives_str = (
            "【Related user memory】\n" + "\n".join(archives_lines)
            if archives_lines
            else "【Related user memory】\n- None"
        )

        context_str = "\n".join(
            f"{item['sender']}: {item['msg']}" for item in recent_context
        )
        reply_style = str(
            self.prompt_settings.get(
                "chat_reply_style",
                "像群里熟人，不要客服腔，不要自我介绍；优先一句短回复，必要时两句；少用感叹号，不要每次都接满。",
            ),
        )
        reply_max_sentences = max(
            1,
            int(self.prompt_settings.get("chat_reply_max_sentences", 2)),
        )
        fallback_memory_reply = str(
            self.prompt_settings.get("fallback_memory_reply", "我有点记不清了。"),
        )
        immediate_reply_style = str(
            self.prompt_settings.get(
                "immediate_reply_style",
                "如果别人是在直接问你，就先正面回答，再决定要不要顺手补半句。",
            ),
        )
        short_window_reply_style = str(
            self.prompt_settings.get(
                "short_window_reply_style",
                "如果只是大家连续提到你，就像路过被 cue 到一样接一句，轻一点、自然一点。",
            ),
        )
        proactive_reply_style = str(
            self.prompt_settings.get(
                "proactive_reply_style",
                "主动接话时要克制，像潜水群友偶尔冒泡，不要强行总结全场。",
            ),
        )
        avoid_phrases = self.prompt_settings.get("avoid_phrases", [])
        if isinstance(avoid_phrases, list):
            avoid_phrase_text = "、".join(
                str(item).strip() for item in avoid_phrases if str(item).strip()
            )
        else:
            avoid_phrase_text = ""

        if channel == "immediate":
            channel_style = immediate_reply_style
        elif channel == "short_window":
            channel_style = short_window_reply_style
        elif channel == "proactive":
            channel_style = proactive_reply_style
        else:
            channel_style = "自然回应，不要刻意表演。"

        prompt = (
            f"{background_str}\n\n"
            f"{memory_recall_block}\n\n"
            f"{archives_str}\n\n"
            f"【Latest context】\n{context_str}\n\n"
            f"【Task】\n"
            f"Trigger topic: {topic}\n"
            f"Reply as a casual group member.\n"
            f"Requirements:\n"
            f"1. {reply_style}\n"
            f"2. 当前触发渠道是 {channel}，补充要求：{channel_style}\n"
            f"3. Keep it within about {reply_max_sentences} sentence(s).\n"
            f"4. 优先像真实群友直接接话，不要写成解释、总结、分析或客服回复。\n"
            f"5. 不要复述整段上下文，不要叫别人“用户”，不要主动介绍自己。\n"
            f"6. 少用书面语，避免用这些表达：{avoid_phrase_text or '无'}\n"
            f"7. If memory is insufficient, say something similar to: {fallback_memory_reply}\n"
            f"8. 如果引用旧记忆但不完全确定，要明确说“大概”“好像”“我记得那阵子”。\n"
            f"9. 先用已有记忆回答，不要编造数据库里没有的往事。\n"
            f"10. 如果问题明显在问过去发生过什么，优先把时间线和同期事件说成模糊回忆。\n"
            f"11. Output reply only."
        )

        try:
            reply_text = await self.router.text_chat(
                role="chat",
                prompt=prompt,
                system_prompt=self.skill_content,
            )
        except Exception as exc:
            self._log(
                "error",
                f"[agentic_memory] Chat generation failed. group={group_id}, channel={channel}, error={exc}",
            )
            return ""

        reply_text = self._sanitize_reply_text(reply_text)
        if not reply_text:
            retry_prompt = (
                f"{prompt}\n"
                "12. You must output exactly one short natural Chinese reply only.\n"
                "13. Do not output JSON, code fences, labels, quotes, brackets, or explanation."
            )
            try:
                retry_text = await self.router.text_chat(
                    role="chat",
                    prompt=retry_prompt,
                    system_prompt=self.skill_content,
                )
            except Exception as exc:
                self._log(
                    "error",
                    "[agentic_memory] Chat retry after empty sanitize failed. "
                    f"group={group_id}, channel={channel}, error={exc}",
                )
                return ""
            reply_text = self._sanitize_reply_text(retry_text)
            if not reply_text:
                self._log(
                    "warning",
                    "[agentic_memory] Chat generation returned empty reply after retry. "
                    f"group={group_id}, channel={channel}, topic={topic[:80]!r}",
                )
                return ""

        if (
            reply_text
            and self._is_duplicate_reply(group_id, reply_text, channel)
            and self.dedup_retry_once
        ):
            retry_prompt = f"{prompt}\n6. Avoid repeating your recent wording."
            try:
                retry_text = await self.router.text_chat(
                    role="chat",
                    prompt=retry_prompt,
                    system_prompt=self.skill_content,
                )
            except Exception as exc:
                self._log(
                    "error",
                    f"[agentic_memory] Chat retry failed. group={group_id}, channel={channel}, error={exc}",
                )
                return reply_text
            return self._sanitize_reply_text(retry_text)

        return reply_text

    async def _call_llm_for_analysis(
        self,
        group_id: str,
        chat_batch: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Analyze a long-window chat batch for memory updates and proactive topics.

        Args:
            group_id: Group identifier.
            chat_batch: Long-window raw messages.

        Returns:
            Parsed analysis result dictionary.
        """
        chat_text = "\n".join(f"{item['sender']}: {item['msg']}" for item in chat_batch)
        analysis_skill_excerpt_chars = max(
            100,
            int(self.prompt_settings.get("analysis_skill_excerpt_chars", 800)),
        )
        system_prompt = (
            "You are a neutral group memory analyzer.\n"
            "Return strict JSON only.\n"
            "Do not fabricate any profile fields.\n"
            "Extract only facts explicitly stated in the conversation.\n"
            "Use concise Chinese output.\n"
            "JSON schema:\n"
            "{\n"
            '  "topic_analysis": {\n'
            '    "summary": "一句话总结",\n'
            '    "is_significant": false,\n'
            '    "matches_preference": false,\n'
            '    "profile_updates": {"用户": {"字段": "值"}},\n'
            '    "dynamic_events": {"用户": ["事件1", "事件2"]}\n'
            "  },\n"
            '  "interject_topic": "供机器人接话的切入点，没有则留空"\n'
            "}\n"
            f"Skill excerpt:\n{self.skill_content[:analysis_skill_excerpt_chars]}"
        )
        prompt = (
            f"Group ID: {group_id}\n"
            f"Message count: {len(chat_batch)}\n"
            f"Conversation:\n{chat_text}"
        )
        response_text = await self.router.text_chat(
            role="analysis",
            prompt=prompt,
            system_prompt=system_prompt,
        )
        parsed_result = self._parse_json_response(response_text, log_failure=False)
        if parsed_result:
            return parsed_result

        self._log(
            "info",
            "[agentic_memory] Analysis response was not valid JSON on first attempt, retrying once. "
            f"group={group_id}, response_head={response_text[:160]!r}",
        )
        retry_prompt = (
            f"{prompt}\n\n"
            "IMPORTANT OUTPUT RULES:\n"
            "1. Respond with exactly one valid JSON object.\n"
            "2. Do not output markdown, code fences, or explanation.\n"
            "3. Keep profile_updates and dynamic_events as JSON objects. Use {} when empty.\n"
            "4. Keep interject_topic as a string. Use an empty string when not needed.\n"
            "5. Make sure all braces and brackets are fully closed."
        )
        retry_response_text = await self.router.text_chat(
            role="analysis",
            prompt=retry_prompt,
            system_prompt=system_prompt + "\nReturn minified valid JSON only.",
        )
        return self._parse_json_response(retry_response_text)

    async def _merge_old_events(self, event_one: str, event_two: str) -> str:
        """Merge two old dynamic events into one short memory line.

        Args:
            event_one: Older event text.
            event_two: Second older event text.

        Returns:
            Merged short event text.
        """
        prompt = (
            f"请把下面两件旧事压缩成一句不超过 {self.dynamic_event_merge_target_length} 字的短句，"
            "用于人物长期印象记录。只输出一句话。\n"
            f"事件1：{event_one}\n"
            f"事件2：{event_two}"
        )
        try:
            merged = await self.router.text_chat(
                role="event_merge",
                prompt=prompt,
                system_prompt="You compress old memory events into one short sentence.",
            )
            merged = self._sanitize_reply_text(merged)
            return merged or f"{event_one}；{event_two}"
        except Exception as exc:
            self._log("error", f"Failed to merge dynamic events: {exc}")
            return f"{event_one}；{event_two}"

    async def _process_memory_task(
        self,
        group_id: str,
        chat_batch: list[dict[str, Any]],
        event: AstrMessageEvent,
    ) -> None:
        """Process long-window memory analysis and optional proactive reply.

        Args:
            group_id: Group identifier.
            chat_batch: Long-window batch.
            event: Incoming message event used for sending proactive replies.
        """
        try:
            llm_result = await self._call_llm_for_analysis(group_id, chat_batch)
            analysis = llm_result.get("topic_analysis", {})
            if not isinstance(analysis, dict):
                return

            summary_text = str(analysis.get("summary", "")).strip()
            is_significant = bool(analysis.get("is_significant", False))
            matches_preference = bool(analysis.get("matches_preference", False))
            interject_topic = str(llm_result.get("interject_topic", "")).strip()

            if summary_text:
                created_at_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                memory_key = datetime.now().strftime("%Y-%m-%d")
                period_start = created_at_text
                period_end = created_at_text
                self.db.add_summary(
                    group_id,
                    "paragraph",
                    summary_text,
                    is_significant,
                    memory_key=memory_key,
                    period_start=period_start,
                    period_end=period_end,
                    source_count=max(1, len(chat_batch)),
                )

            profile_updates = analysis.get("profile_updates", {})
            if isinstance(profile_updates, dict):
                for user_name, updates in profile_updates.items():
                    if not isinstance(updates, dict) or not updates:
                        continue
                    user_data = self.db.get_user_profile(group_id, str(user_name))
                    fixed_data = user_data["fixed_data"].copy()
                    fixed_data.update(updates)
                    self.db.upsert_user_profile(
                        group_id,
                        str(user_name),
                        fixed_data,
                        user_data["dynamic_events"],
                    )

            dynamic_events = analysis.get("dynamic_events", {})
            if isinstance(dynamic_events, dict):
                for user_name, events in dynamic_events.items():
                    if isinstance(events, str):
                        event_list = [events]
                    elif isinstance(events, list):
                        event_list = [
                            str(item).strip() for item in events if str(item).strip()
                        ]
                    else:
                        event_list = []
                    if not event_list:
                        continue

                    user_data = self.db.get_user_profile(group_id, str(user_name))
                    fixed_data = user_data["fixed_data"]
                    merged_events = list(user_data["dynamic_events"])
                    merged_events.extend(event_list)

                    while (
                        len(merged_events) > self.max_dynamic_events
                        and len(merged_events) >= 2
                    ):
                        oldest_one = merged_events.pop(0)
                        oldest_two = merged_events.pop(0)
                        merged_events.insert(
                            0,
                            await self._merge_old_events(oldest_one, oldest_two),
                        )

                    self.db.upsert_user_profile(
                        group_id,
                        str(user_name),
                        fixed_data,
                        merged_events,
                    )

            if not self._is_cooldown_ready(
                group_id,
                "proactive",
                self.proactive_cooldown_seconds,
            ):
                return

            should_speak = False
            if is_significant:
                should_speak = True
            else:
                probability = (
                    self.boost_interject_probability
                    if matches_preference
                    else self.base_interject_probability
                )
                should_speak = random.random() < probability

            if not should_speak or not interject_topic:
                return

            recent_context = chat_batch[-self.recent_context_messages_for_interject :]
            reply_text = await self._build_chat_reply(
                group_id,
                interject_topic,
                recent_context,
                channel="proactive",
            )
            sent = await self._send_reply(
                event,
                group_id,
                reply_text,
                channel="proactive",
                cooldown_key="proactive",
            )
            if not sent:
                self._log(
                    "info",
                    f"Proactive reply was skipped. group={group_id}, topic={interject_topic}",
                )
        except Exception as exc:
            self._log("error", f"Background memory task failed: {exc}")

    @filter.event_message_type(EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent) -> None:
        """Handle group messages for whitelist, immediate reply, and memory logic.

        Args:
            event: Incoming message event.
        """
        if not hasattr(event, "message_obj"):
            self._log("warning", "[agentic_memory] Group event has no message_obj.")
            return
        group_id = str(
            getattr(
                event.message_obj,
                "group_id",
                getattr(event, "session_id", ""),
            ),
        ).strip()
        if not group_id:
            self._log(
                "warning", "[agentic_memory] Received group message without group_id."
            )
            return
        if not self._is_group_allowed(group_id):
            self._log(
                "info", f"[agentic_memory] Group {group_id} is blocked by group_scope."
            )
            return

        message_text = event.message_str.strip() if event.message_str else ""
        if not message_text:
            message_text = str(
                getattr(getattr(event, "message_obj", None), "raw_message", "")
            ).strip()
        if not message_text:
            self._log(
                "info",
                f"[agentic_memory] Empty message text in group {group_id}, skip processing.",
            )
            return

        sender_name = str(event.get_sender_name()).strip() or "Unknown"
        self._log_raw_event_debug(event, group_id, message_text)
        trigger_state = self._build_trigger_state(event, message_text)
        self._log(
            "info",
            "[agentic_memory] Received group message. "
            f"group={group_id}, sender={sender_name}, mentioned={trigger_state['is_mentioned']}, "
            f"name_hit={trigger_state['name_hit']}, question_hit={trigger_state['question_pattern_hit']}, "
            f"text={message_text[:120]}",
        )

        self._append_message(group_id, sender_name, message_text, trigger_state)
        long_window_batch = self._pop_long_window_batch(group_id)
        if long_window_batch:
            asyncio.create_task(
                self._process_memory_task(group_id, long_window_batch, event)
            )

        special_reply = self._match_special_reply(group_id, trigger_state)
        if special_reply:
            sent = await self._send_reply(
                event,
                group_id,
                special_reply,
                channel="special",
            )
            if sent:
                self._stop_event_flow(event)
                return

        should_reply_immediately = (
            trigger_state["is_mentioned"]
            or trigger_state["name_hit"]
            or trigger_state["question_pattern_hit"]
        )
        if should_reply_immediately and self._is_cooldown_ready(
            group_id,
            "immediate",
            self.immediate_cooldown_seconds,
        ):
            self._log(
                "info",
                f"[agentic_memory] Immediate reply triggered in group {group_id}.",
            )
            recent_context = self.message_buffers[group_id][
                -self.immediate_context_messages :
            ]
            reply_text = await self._build_chat_reply(
                group_id,
                message_text,
                recent_context,
                channel="immediate",
            )
            sent = await self._send_reply(
                event,
                group_id,
                reply_text,
                channel="immediate",
                cooldown_key="immediate",
            )
            if sent:
                self._stop_event_flow(event)
                return
            self._log(
                "info",
                "[agentic_memory] Immediate reply trigger finished without sending. "
                f"group={group_id}. Check preceding logs for sanitize/dedup/send details.",
            )
        elif should_reply_immediately:
            self._log(
                "info",
                f"[agentic_memory] Immediate reply was blocked by cooldown. group={group_id}",
            )

        if self._should_trigger_short_window(group_id) and self._is_cooldown_ready(
            group_id,
            "short_window",
            self.short_window_cooldown_seconds,
        ):
            self._log(
                "info",
                f"[agentic_memory] Short-window reply triggered in group {group_id}.",
            )
            recent_context = self.message_buffers[group_id][
                -self.immediate_context_messages :
            ]
            reply_text = await self._build_chat_reply(
                group_id,
                "最近群里好像一直在提到你，顺着聊一句。",
                recent_context,
                channel="short_window",
            )
            sent = await self._send_reply(
                event,
                group_id,
                reply_text,
                channel="short_window",
                cooldown_key="short_window",
            )
            if sent:
                self._stop_event_flow(event)
            else:
                self._log(
                    "info",
                    "[agentic_memory] Short-window trigger finished without sending. "
                    f"group={group_id}. Check preceding logs for sanitize/dedup/send details.",
                )

    async def _call_compression_text(self, prompt: str) -> str:
        """Call the compression role for memory folding tasks.

        Args:
            prompt: Compression prompt.

        Returns:
            Compressed text result.
        """
        return self._sanitize_reply_text(
            await self.router.text_chat(
                role="compression",
                prompt=prompt,
                system_prompt="You compress long chat memory into concise summaries.",
            )
        )

    async def _merge_rollup_content(
        self,
        existing_content: str,
        new_content: str,
        summary_type: str,
        max_chars: int,
    ) -> str:
        """Merge one existing rolled-up summary with newly generated content.

        Args:
            existing_content: Existing summary text.
            new_content: Newly generated summary text.
            summary_type: Target layer name.
            max_chars: Maximum output length.

        Returns:
            Merged summary text.
        """
        if not existing_content:
            return new_content
        if not new_content:
            return existing_content

        prompt = (
            f"请把下面两段 {summary_type} 记忆融合成一段不超过 {max_chars} 字的摘要。\n"
            "保留时间感和重要事件，允许模糊但不要丢掉关键线索。\n"
            f"旧摘要：{existing_content}\n"
            f"新摘要：{new_content}"
        )
        merged = await self._call_compression_text(prompt)
        return merged or new_content

    async def _upsert_rollup_summary(
        self,
        group_id: str,
        summary_type: str,
        memory_key: str,
        content: str,
        period_start: str,
        period_end: str,
        source_count: int,
        max_chars: int,
        is_significant: bool = False,
    ) -> None:
        """Insert or merge one rolled-up summary for the target layer.

        Args:
            group_id: Group identifier.
            summary_type: Target summary layer.
            memory_key: Logical bucket key.
            content: Generated summary text.
            period_start: Covered period start.
            period_end: Covered period end.
            source_count: Number of source rows.
            max_chars: Maximum merged length.
            is_significant: Whether the summary is significant.
        """
        existing_row = self.db.get_summary_by_memory_key(
            group_id, summary_type, memory_key
        )
        if existing_row:
            merged_content = await self._merge_rollup_content(
                str(existing_row[1]),
                content,
                summary_type,
                max_chars,
            )
            merged_significant = bool(existing_row[3]) or is_significant
            existing_source_count = int(existing_row[9] or 0)
            merged_period_start = period_start
            if existing_row[5] and str(existing_row[5]) < period_start:
                merged_period_start = str(existing_row[5])
            merged_period_end = period_end
            if existing_row[6] and str(existing_row[6]) > period_end:
                merged_period_end = str(existing_row[6])
            self.db.update_summary_content(
                int(existing_row[0]),
                merged_content,
                merged_significant,
                merged_period_start,
                merged_period_end,
                existing_source_count + source_count,
            )
            return

        self.db.add_summary(
            group_id,
            summary_type,
            content,
            is_significant,
            memory_key=memory_key,
            period_start=period_start,
            period_end=period_end,
            source_count=source_count,
        )

    async def _rollup_paragraph_to_daily(self, group_id: str, now: datetime) -> None:
        """Roll up paragraph memory into daily memory with delayed cleanup.

        Args:
            group_id: Group identifier.
            now: Current scheduler time.
        """
        cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
        rows = self.db.get_rollup_candidates(
            group_id,
            "paragraph",
            period_end=cutoff.strftime("%Y-%m-%d %H:%M:%S"),
        )
        if not rows:
            return

        groups: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
        for row in rows:
            memory_key = str(row[4] or row[2][:10])
            groups[memory_key].append(row)

        chunk_size = max(1, int(self.summary_settings.get("daily_chunk_size", 10)))
        chunk_summary_max_chars = max(
            20,
            int(self.summary_settings.get("daily_chunk_summary_max_chars", 100)),
        )
        daily_summary_max_chars = max(
            40,
            int(self.summary_settings.get("daily_summary_max_chars", 150)),
        )

        for memory_key, day_rows in groups.items():
            source_texts = [
                str(row[1]).strip() for row in day_rows if str(row[1]).strip()
            ]
            if not source_texts:
                continue

            chunks = [
                source_texts[index : index + chunk_size]
                for index in range(0, len(source_texts), chunk_size)
            ]
            intermediate_summaries: list[str] = []
            for chunk in chunks:
                chunk_text = "\n".join(f"- {line}" for line in chunk)
                prompt = (
                    f"请把下面这些段落记忆融合成一段不超过 {chunk_summary_max_chars} 字的日摘要素材。\n"
                    "保留那天的主要事件、情绪和人物线索。\n"
                    f"{chunk_text}"
                )
                compressed = await self._call_compression_text(prompt)
                if compressed:
                    intermediate_summaries.append(compressed)

            if not intermediate_summaries:
                continue

            final_prompt = (
                f"请将下面这些日摘要素材合并成一份不超过 {daily_summary_max_chars} 字的当日记忆。\n"
                "风格像回忆一天聊了什么，允许模糊，但要保留重点。\n"
                + "\n".join(f"- {line}" for line in intermediate_summaries)
            )
            daily_summary = await self._call_compression_text(final_prompt)
            if not daily_summary:
                continue

            period_start = min(str(row[5] or row[2]) for row in day_rows)
            period_end = max(str(row[6] or row[2]) for row in day_rows)
            source_count = sum(max(1, int(row[9] or 1)) for row in day_rows)
            await self._upsert_rollup_summary(
                group_id,
                "daily",
                memory_key,
                daily_summary,
                period_start,
                period_end,
                source_count,
                daily_summary_max_chars,
                is_significant=any(bool(row[3]) for row in day_rows),
            )
            cleanup_after = (
                cutoff + timedelta(days=self.paragraph_cleanup_delay_days)
            ).strftime("%Y-%m-%d %H:%M:%S")
            self.db.mark_summaries_rolled_up(
                [int(row[0]) for row in day_rows],
                now.strftime("%Y-%m-%d %H:%M:%S"),
                cleanup_after,
            )

    async def _rollup_daily_to_month(self, group_id: str, now: datetime) -> None:
        """Roll up daily memory into monthly memory with delayed cleanup.

        Args:
            group_id: Group identifier.
            now: Current scheduler time.
        """
        current_month_start = now.replace(
            day=1,
            hour=self.rollup_hour,
            minute=self.rollup_minute,
            second=0,
            microsecond=0,
        )
        rows = self.db.get_rollup_candidates(
            group_id,
            "daily",
            period_end=current_month_start.strftime("%Y-%m-%d %H:%M:%S"),
        )
        if not rows:
            return

        groups: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
        for row in rows:
            key_source = str(row[4] or row[5] or row[2])[:7]
            groups[key_source].append(row)

        month_summary_max_chars = max(
            60,
            int(self.summary_settings.get("month_summary_max_chars", 220)),
        )
        month_input_char_limit = max(
            200,
            int(self.summary_settings.get("month_input_char_limit", 4000)),
        )

        for memory_key, month_rows in groups.items():
            lines = [str(row[1]).strip() for row in month_rows if str(row[1]).strip()]
            if not lines:
                continue

            prompt = (
                f"请把下面这些日记忆总结成一段不超过 {month_summary_max_chars} 字的月度记忆。\n"
                "重点保留这个月反复出现的话题、人物和重要事件。\n"
                f"{chr(10).join(f'- {line}' for line in lines)[:month_input_char_limit]}"
            )
            month_summary = await self._call_compression_text(prompt)
            if not month_summary:
                continue

            period_start = min(str(row[5] or row[2]) for row in month_rows)
            period_end = max(str(row[6] or row[2]) for row in month_rows)
            source_count = sum(max(1, int(row[9] or 1)) for row in month_rows)
            await self._upsert_rollup_summary(
                group_id,
                "month",
                memory_key,
                month_summary,
                period_start,
                period_end,
                source_count,
                month_summary_max_chars,
                is_significant=any(bool(row[3]) for row in month_rows),
            )
            cleanup_after = self._shift_months(
                current_month_start,
                self.daily_cleanup_delay_months,
            ).strftime("%Y-%m-%d %H:%M:%S")
            self.db.mark_summaries_rolled_up(
                [int(row[0]) for row in month_rows],
                now.strftime("%Y-%m-%d %H:%M:%S"),
                cleanup_after,
            )

    async def _rollup_month_to_year(self, group_id: str, now: datetime) -> None:
        """Roll up monthly memory into yearly memory with delayed cleanup.

        Args:
            group_id: Group identifier.
            now: Current scheduler time.
        """
        current_year_start = datetime(
            now.year,
            1,
            1,
            self.rollup_hour,
            self.rollup_minute,
        )
        rows = self.db.get_rollup_candidates(
            group_id,
            "month",
            period_end=current_year_start.strftime("%Y-%m-%d %H:%M:%S"),
        )
        if not rows:
            return

        groups: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
        for row in rows:
            key_source = str(row[4] or row[5] or row[2])[:4]
            groups[key_source].append(row)

        year_summary_max_chars = max(
            80,
            int(self.summary_settings.get("year_summary_max_chars", 260)),
        )
        year_input_char_limit = max(
            300,
            int(self.summary_settings.get("year_input_char_limit", 5000)),
        )

        for memory_key, year_rows in groups.items():
            lines = [str(row[1]).strip() for row in year_rows if str(row[1]).strip()]
            if not lines:
                continue

            prompt = (
                f"请把下面这些月度记忆总结成一段不超过 {year_summary_max_chars} 字的年度记忆。\n"
                "保留这一年里最有代表性的事件、关系变化和群聊氛围。\n"
                f"{chr(10).join(f'- {line}' for line in lines)[:year_input_char_limit]}"
            )
            year_summary = await self._call_compression_text(prompt)
            if not year_summary:
                continue

            period_start = min(str(row[5] or row[2]) for row in year_rows)
            period_end = max(str(row[6] or row[2]) for row in year_rows)
            source_count = sum(max(1, int(row[9] or 1)) for row in year_rows)
            await self._upsert_rollup_summary(
                group_id,
                "year",
                memory_key,
                year_summary,
                period_start,
                period_end,
                source_count,
                year_summary_max_chars,
                is_significant=any(bool(row[3]) for row in year_rows),
            )
            cleanup_after = datetime(
                current_year_start.year + self.month_cleanup_delay_years,
                1,
                1,
                self.rollup_hour,
                self.rollup_minute,
            ).strftime("%Y-%m-%d %H:%M:%S")
            self.db.mark_summaries_rolled_up(
                [int(row[0]) for row in year_rows],
                now.strftime("%Y-%m-%d %H:%M:%S"),
                cleanup_after,
            )

    async def _rollup_year_to_history(self, group_id: str, now: datetime) -> None:
        """Roll up older yearly memory into long-term history with delayed cleanup.

        Args:
            group_id: Group identifier.
            now: Current scheduler time.
        """
        history_year = now.year - self.year_retention_years
        history_cutoff = datetime(
            history_year,
            1,
            1,
            self.rollup_hour,
            self.rollup_minute,
        )
        rows = self.db.get_rollup_candidates(
            group_id,
            "year",
            period_end=history_cutoff.strftime("%Y-%m-%d %H:%M:%S"),
        )
        if not rows:
            return

        history_summary_max_chars = max(
            100,
            int(self.summary_settings.get("history_summary_max_chars", 320)),
        )
        history_input_char_limit = max(
            300,
            int(self.summary_settings.get("history_input_char_limit", 5000)),
        )
        lines = [str(row[1]).strip() for row in rows if str(row[1]).strip()]
        if not lines:
            return

        oldest_year = min(str(row[4] or row[5] or row[2])[:4] for row in rows)
        newest_year = max(str(row[4] or row[5] or row[2])[:4] for row in rows)
        memory_key = f"{oldest_year}-{newest_year}"
        prompt = (
            f"请把下面这些年度记忆融合成一段不超过 {history_summary_max_chars} 字的历史记忆。\n"
            "像人在回忆很多年前的事，保留重要事件和长期印象，不追求绝对精确。\n"
            f"{chr(10).join(f'- {line}' for line in lines)[:history_input_char_limit]}"
        )
        history_summary = await self._call_compression_text(prompt)
        if not history_summary:
            return

        period_start = min(str(row[5] or row[2]) for row in rows)
        period_end = max(str(row[6] or row[2]) for row in rows)
        source_count = sum(max(1, int(row[9] or 1)) for row in rows)
        await self._upsert_rollup_summary(
            group_id,
            "history",
            memory_key,
            history_summary,
            period_start,
            period_end,
            source_count,
            history_summary_max_chars,
            is_significant=any(bool(row[3]) for row in rows),
        )
        cleanup_after = datetime(
            now.year + self.year_cleanup_delay_years,
            1,
            1,
            self.rollup_hour,
            self.rollup_minute,
        ).strftime("%Y-%m-%d %H:%M:%S")
        self.db.mark_summaries_rolled_up(
            [int(row[0]) for row in rows],
            now.strftime("%Y-%m-%d %H:%M:%S"),
            cleanup_after,
        )

    async def _cleanup_rolled_up_memory(self, group_id: str, now: datetime) -> None:
        """Delete source rows whose delayed cleanup time has arrived.

        Args:
            group_id: Group identifier.
            now: Current scheduler time.
        """
        now_text = now.strftime("%Y-%m-%d %H:%M:%S")
        for summary_type in ("paragraph", "daily", "month", "year"):
            deleted_count = self.db.delete_ready_summaries(
                group_id, summary_type, now_text
            )
            if deleted_count > 0:
                self._log(
                    "info",
                    f"[agentic_memory] Deleted {deleted_count} delayed {summary_type} memories for group {group_id}.",
                )

    async def _cron_daily_compression(self) -> None:
        """Run scheduled five-layer memory compression in the background."""
        while True:
            now = datetime.now()
            next_run = now.replace(
                hour=self.rollup_hour,
                minute=self.rollup_minute,
                second=0,
                microsecond=0,
            )
            if now >= next_run:
                next_run += timedelta(days=1)

            wait_seconds = max(1, int((next_run - now).total_seconds()))
            self._log(
                "info",
                f"Memory compression is sleeping for {wait_seconds / 3600:.2f} hours.",
            )
            await asyncio.sleep(wait_seconds)

            run_time = datetime.now().replace(second=0, microsecond=0)
            try:
                active_groups = self.db.list_group_ids_with_summaries()
                for group_id in active_groups:
                    await self._rollup_paragraph_to_daily(group_id, run_time)

                    if run_time.day == 1:
                        await self._rollup_daily_to_month(group_id, run_time)

                    if run_time.month == 1 and run_time.day == 1:
                        await self._rollup_month_to_year(group_id, run_time)

                    await self._rollup_year_to_history(group_id, run_time)
                    await self._cleanup_rolled_up_memory(group_id, run_time)
            except Exception as exc:
                self._log("error", f"Scheduled memory compression failed: {exc}")

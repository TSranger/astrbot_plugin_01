import asyncio
import json
import random
import re
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import EventMessageType
from astrbot.api.star import Context, Star, register
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
        self.prompt_settings = self.config.get("prompt_settings", {})
        self.reply_dedup_settings = self.config.get("reply_dedup", {})
        self.special_reply_settings = self.config.get("special_replies", {})
        self.proactive_talk_settings = self.config.get("proactive_talk_settings", {})
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

        self.skill_content = self._load_skill(
            self.skill_settings.get("active_skill_file", ""),
        )
        self.router = PluginLLMRouter(self.context, self.config.get("llm_settings", {}))
        self.db = MemoryDBManager(str(self._resolve_db_path()))

        self.message_buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.short_windows: dict[str, deque[dict[str, Any]]] = {}
        self.reply_history: dict[str, deque[str]] = {}
        self.cooldowns: dict[str, dict[str, datetime]] = defaultdict(dict)

        asyncio.create_task(self._cron_daily_compression())

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
            logger.warning(f"Skill file was not found: {skill_file}")
            return "You are a natural, casual group member."
        return skill_file.read_text(encoding="utf-8")

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
            logger.debug(f"Failed to inspect message chain mentions: {exc}")

        message_text = str(getattr(event, "message_str", "")).strip()
        return f"@{bot_id}" in message_text if message_text else False

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
            ("{", "}"),
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

    def _parse_json_response(self, response_text: str) -> dict[str, Any]:
        """Parse model JSON output as safely as possible.

        Args:
            response_text: Raw model text.

        Returns:
            Parsed dictionary or an empty dict.
        """
        cleaned = response_text.strip()
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        try:
            loaded = json.loads(cleaned)
            return loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return {}
            try:
                loaded = json.loads(cleaned[start : end + 1])
                return loaded if isinstance(loaded, dict) else {}
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON response: {cleaned[:200]}")
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
        reply_text = self._sanitize_reply_text(reply_text)
        if not reply_text:
            return False
        if self._is_duplicate_reply(group_id, reply_text, channel):
            logger.info(f"Skipped duplicated reply in group {group_id}: {reply_text}")
            return False

        await event.send(event.plain_result(reply_text))
        self._record_reply_history(group_id, reply_text)
        if cooldown_key:
            self._mark_cooldown(group_id, cooldown_key)
        logger.info(f"[{channel}] Sent reply in group {group_id}: {reply_text}")
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
        recent_summaries_rows = self.db.get_summaries(group_id, "paragraph")
        recent_summaries = [
            row[1] for row in recent_summaries_rows[-self.immediate_summary_count :]
        ]
        if recent_summaries:
            background_str = "【Recent group background】\n- " + "\n- ".join(
                recent_summaries
            )
        else:
            background_str = "【Recent group background】\n- None"

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
            f"8. Do not fabricate old memories that are not present above.\n"
            f"9. Output reply only."
        )

        reply_text = await self.router.text_chat(
            role="chat",
            prompt=prompt,
            system_prompt=self.skill_content,
        )
        reply_text = self._sanitize_reply_text(reply_text)

        if (
            reply_text
            and self._is_duplicate_reply(group_id, reply_text, channel)
            and self.dedup_retry_once
        ):
            retry_prompt = f"{prompt}\n6. Avoid repeating your recent wording."
            retry_text = await self.router.text_chat(
                role="chat",
                prompt=retry_prompt,
                system_prompt=self.skill_content,
            )
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
        return self._parse_json_response(response_text)

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
            logger.error(f"Failed to merge dynamic events: {exc}")
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
                self.db.add_summary(group_id, "paragraph", summary_text, is_significant)

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
                logger.info(
                    f"Proactive reply was skipped. group={group_id}, topic={interject_topic}",
                )
        except Exception as exc:
            logger.error(f"Background memory task failed: {exc}")

    @filter.event_message_type(EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent) -> None:
        """Handle group messages for whitelist, immediate reply, and memory logic.

        Args:
            event: Incoming group message event.
        """
        if not hasattr(event, "message_obj"):
            return
        group_id = str(
            getattr(
                event.message_obj,
                "group_id",
                getattr(event, "session_id", ""),
            ),
        ).strip()
        if not group_id or not self._is_group_allowed(group_id):
            return

        message_text = event.message_str.strip() if event.message_str else ""
        if not message_text:
            return

        sender_name = str(event.get_sender_name()).strip() or "Unknown"
        trigger_state = self._build_trigger_state(event, message_text)

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

        if self._should_trigger_short_window(group_id) and self._is_cooldown_ready(
            group_id,
            "short_window",
            self.short_window_cooldown_seconds,
        ):
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

    async def _compress_daily_memory(self, group_id: str, before_time: str) -> None:
        """Compress paragraph summaries into a daily summary.

        Args:
            group_id: Group identifier.
            before_time: Time cutoff for selecting paragraph summaries.
        """
        paragraphs = self.db.get_summaries(
            group_id, "paragraph", before_time=before_time
        )
        if not paragraphs:
            return

        texts_to_compress = [row[1] for row in paragraphs]
        chunk_size = max(1, int(self.summary_settings.get("daily_chunk_size", 10)))
        chunk_summary_max_chars = max(
            20,
            int(self.summary_settings.get("daily_chunk_summary_max_chars", 100)),
        )
        daily_summary_max_chars = max(
            40,
            int(self.summary_settings.get("daily_summary_max_chars", 150)),
        )

        chunks = [
            texts_to_compress[index : index + chunk_size]
            for index in range(0, len(texts_to_compress), chunk_size)
        ]

        intermediate_summaries: list[str] = []
        for chunk in chunks:
            chunk_text = "\n".join(f"- {text}" for text in chunk)
            prompt = (
                f"请把下面这些段落总结融合成一段不超过 {chunk_summary_max_chars} 字的摘要。\n"
                f"{chunk_text}"
            )
            compressed = await self._call_compression_text(prompt)
            if compressed:
                intermediate_summaries.append(compressed)

        if not intermediate_summaries:
            return

        final_text = "\n".join(f"- {text}" for text in intermediate_summaries)
        final_prompt = (
            f"请将下面这些中间摘要合并成一份不超过 {daily_summary_max_chars} 字的今日群聊总结。\n"
            f"{final_text}"
        )
        daily_summary = await self._call_compression_text(final_prompt)
        if not daily_summary:
            return

        self.db.add_summary(group_id, "daily", daily_summary)
        self.db.delete_summaries(group_id, "paragraph", before_time=before_time)
        logger.info(f"Daily memory compression finished for group {group_id}")

    async def _compress_history_memory(self, group_id: str, before_time: str) -> None:
        """Compress old daily summaries into long-term history memory.

        Args:
            group_id: Group identifier.
            before_time: Time cutoff for selecting daily summaries.
        """
        daily_summaries = self.db.get_summaries(
            group_id, "daily", before_time=before_time
        )
        if not daily_summaries:
            return

        history_summary_max_chars = max(
            50,
            int(self.summary_settings.get("history_summary_max_chars", 200)),
        )
        history_input_char_limit = max(
            200,
            int(self.summary_settings.get("history_input_char_limit", 3000)),
        )
        merged_text = "\n".join(f"- {row[1]}" for row in daily_summaries)
        prompt = (
            f"请把这些一年以前的群聊日总结浓缩成一段不超过 {history_summary_max_chars} 字的老旧历史。"
            "保留最重要的大事和群友特征。\n"
            f"{merged_text[:history_input_char_limit]}"
        )
        history_summary = await self._call_compression_text(prompt)
        if not history_summary:
            return

        self.db.add_summary(group_id, "history", history_summary)
        self.db.delete_summaries(group_id, "daily", before_time=before_time)
        logger.info(f"History memory compression finished for group {group_id}")

    async def _cron_daily_compression(self) -> None:
        """Run scheduled daily and monthly memory compression in the background."""
        while True:
            now = datetime.now()
            next_run = now.replace(hour=4, minute=0, second=0, microsecond=0)
            if now >= next_run:
                next_run += timedelta(days=1)

            wait_seconds = max(1, int((next_run - now).total_seconds()))
            logger.info(
                f"Memory compression is sleeping for {wait_seconds / 3600:.2f} hours.",
            )
            await asyncio.sleep(wait_seconds)

            try:
                with self.db._get_conn() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT DISTINCT group_id FROM Chat_Summaries")
                    active_groups = [str(row[0]) for row in cursor.fetchall()]

                paragraph_retention_days = max(
                    1,
                    int(self.summary_settings.get("paragraph_retention_days", 2)),
                )
                daily_retention_days = max(
                    30,
                    int(self.summary_settings.get("daily_retention_days", 365)),
                )

                paragraph_cutoff = (
                    datetime.now() - timedelta(days=paragraph_retention_days - 1)
                ).replace(hour=0, minute=0, second=0, microsecond=0)
                paragraph_cutoff_text = paragraph_cutoff.strftime("%Y-%m-%d %H:%M:%S")

                daily_cutoff = (
                    datetime.now() - timedelta(days=daily_retention_days)
                ).strftime("%Y-%m-%d %H:%M:%S")

                for group_id in active_groups:
                    await self._compress_daily_memory(group_id, paragraph_cutoff_text)

                    if datetime.now().day == 1:
                        await self._compress_history_memory(group_id, daily_cutoff)
            except Exception as exc:
                logger.error(f"Scheduled memory compression failed: {exc}")

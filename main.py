import asyncio
import json
import random
import os
import re
import yaml
from datetime import datetime, timedelta
from loguru import logger

# AstrBot V4 SDK 导入核心库
from astrbot.api.star import Context, Star, register
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.event.filter import EventMessageType
from astrbot.api.provider import ProviderRequest

# 导入数据库管理器
from .db_manager import MemoryDBManager 

@register("agentic_memory", "YourName", "2.0", "具备长期折叠记忆与主动行为的群聊智能体")
class AgenticMemoryPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        
        # 【修复日志报错】绑定专属 plugin_tag，这样日志就能正常显示在 Web 面板了！
        self.logger = logger.bind(plugin_tag="agentic_memory")
        
        # 1. 初始化滑动窗口缓冲池
        self.message_buffers = {}
        
        # 2. 加载外部配置
        self.config = self._load_config("config.yaml")
        mem_cfg = self.config.get('memory_settings', {})
        
        # 触发阈值与滑动重叠
        self.threshold = mem_cfg.get('buffer_threshold', 50)
        self.overlap = mem_cfg.get('overlap_count', 10)
        
        # 概率控制
        self.base_prob = mem_cfg.get('base_interject_probability', 0.15)
        self.boost_prob = mem_cfg.get('boost_interject_probability', 0.60)
        
        # 事件滚动窗口
        self.max_events = mem_cfg.get('max_dynamic_events', 10)
        
        # 3. 加载人设技能 (用于提取“偏好”和对话)
        self.skill_content = self._load_skill(self.config.get('skill_settings', {}).get('active_skill_file', ''))
        
        # 4. 初始化 SQLite 数据库
        db_path = mem_cfg.get('memory_db_path', 'data/agentic_memory.db')
        self.db = MemoryDBManager(db_path)

        # 5. 启动后台定时任务循环 (每天凌晨 4 点执行压缩)
        asyncio.create_task(self._cron_daily_compression())

    # ==========================================
    #         基础配置加载
    # ==========================================

    def _load_config(self, filepath: str) -> dict:
        curr_dir = os.path.dirname(__file__)
        full_path = os.path.join(curr_dir, filepath)
        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        self.logger.error(f"严重错误: 配置文件 {filepath} 缺失！")
        return {"memory_settings": {}, "skill_settings": {}}

    def _load_skill(self, filepath: str) -> str:
        if not filepath: return "你是一个普通的群友。"
        curr_dir = os.path.dirname(__file__)
        full_path = os.path.join(curr_dir, filepath)
        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        self.logger.warning(f"Skill 文件未找到: {full_path}")
        return "你是一个普通的群友。"

    # ==========================================
    #         事件监听与滑动窗口
    # ==========================================

    @filter.event_message_type(EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        group_id = str(getattr(event.message_obj, 'group_id', event.session_id))
        sender_name = event.get_sender_name()
        message_text = event.message_str.strip()
        
        if not message_text: return

        # 拦截 @机器人 或 提及名字的消息
        bot_names = self.config.get('skill_settings', {}).get('bot_names', [])
        is_mentioned = False
        
        if hasattr(event, 'is_at') and getattr(event, 'is_at'):
            is_mentioned = True
        elif any(name in message_text for name in bot_names):
            is_mentioned = True
        
        if is_mentioned:
            recent_msgs = self.message_buffers.get(group_id, [])[-15:]
            recent_msgs.append({"sender": sender_name, "msg": message_text})
            reply_text = await self._generate_reply("有人@我或提到了我", recent_msgs, group_id)
            if reply_text:
                await event.send(event.plain_result(reply_text))
            return 

        # 日常潜水收集消息
        if group_id not in self.message_buffers:
            self.message_buffers[group_id] = []
            
        self.message_buffers[group_id].append({
            "sender": sender_name,
            "msg": message_text
        })

        # 滑动窗口判定触发
        if len(self.message_buffers[group_id]) >= self.threshold:
            chat_batch = self.message_buffers[group_id].copy()
            self.message_buffers[group_id] = self.message_buffers[group_id][-self.overlap:] if self.overlap > 0 else []
            asyncio.create_task(self._process_memory_task(group_id, chat_batch, event))

    # ==========================================
    #         核心大脑：记忆提纯与决策树
    # ==========================================

    async def _process_memory_task(self, group_id: str, chat_batch: list, event: AstrMessageEvent):
        try:
            chat_text = "\n".join([f"{item['sender']}: {item['msg']}" for item in chat_batch])
            
            llm_result = await self._call_llm_for_analysis(chat_text, len(chat_batch))
            if not llm_result: return

            analysis = llm_result.get("topic_analysis", {})
            summary_text = analysis.get("summary", "无有效总结")
            is_significant = analysis.get("is_significant", False)
            matches_preference = analysis.get("matches_preference", False)
            interject_topic = llm_result.get("interject_topic", "")
            
            self.db.add_summary(group_id, "paragraph", summary_text, is_significant)

            profile_updates = analysis.get("profile_updates", {})
            for user_name, updates in profile_updates.items():
                if not isinstance(updates, dict) or not updates: continue
                user_data = self.db.get_user_profile(group_id, user_name)
                fixed = user_data["fixed_data"]
                fixed.update(updates) 
                self.db.upsert_user_profile(group_id, user_name, fixed, user_data["dynamic_events"])
                self.logger.info(f"更新了群友 {user_name} 的固定档案。")

            should_speak = False

            if is_significant:
                target_events = analysis.get("target_users_events", {})
                for user_name, event_desc in target_events.items():
                    user_data = self.db.get_user_profile(group_id, user_name)
                    fixed = user_data["fixed_data"]
                    events = user_data["dynamic_events"]
                    events.append(event_desc)
                    
                    if len(events) > self.max_events:
                        self.logger.info(f"群友 {user_name} 动态事件超载，执行滚动合并...")
                        oldest_1 = events.pop(0)
                        oldest_2 = events.pop(0)
                        merged_event = await self._merge_old_events(oldest_1, oldest_2)
                        events.insert(0, merged_event)
                        
                    self.db.upsert_user_profile(group_id, user_name, fixed, events)
                
                should_speak = True
                self.logger.success(f"检测到大事。档案已更新，强制触发发言。")

            else:
                current_prob = self.boost_prob if matches_preference else self.base_prob
                if random.random() < current_prob:
                    should_speak = True
                    log_msg = "命中兴趣偏好" if matches_preference else "小事随缘命中"
                    self.logger.info(f"{log_msg} ({current_prob*100}%)，准备搭茬。")

            if should_speak and interject_topic:
                reply_text = await self._generate_reply(interject_topic, chat_batch[-15:], group_id)
                if reply_text:
                    await event.send(event.plain_result(reply_text))

        except Exception as e:
            self.logger.error(f"后台记忆处理任务崩溃: {e}")

    # ==========================================
    #         底层 LLM 接口：分析与压缩
    # ==========================================

    async def _call_llm_for_analysis(self, chat_text: str, message_count: int) -> dict:
        system_prompt = f"""你是一个毫无感情的群聊观测系统。请严格执行以下指令：
1. 【概括】一句话总结这 {message_count} 条消息的核心内容。
2. 【固定档案】提取对话中暴露的群友固定属性（如：性别、年龄、职业、偏好）。必须输出在对话中明确提到的内容，严禁自行推理。
3. 【事件定性】判断对话中是否包含某人的人生大事（结婚、辞职、生病等）。
4. 【偏好匹配】判断当前讨论的话题是否与以下人设的兴趣高度相关。
[机器人人设参考]：
{self.skill_content[:500]} 

请严格输出 JSON 格式（绝不能带有 ```json 标记）：
{{
    "topic_analysis": {{
        "summary": "一句话概括讨论内容",
        "is_significant": false,
        "matches_preference": false,
        "profile_updates": {{
            "群友昵称": {{"属性名": "属性值"}} 
        }},
        "target_users_events": {{
            "群友昵称": "重大事件描述" 
        }}
    }},
    "interject_topic": "提取一个简短的话题切入点供机器人插话，无则留空"
}}"""
        user_prompt = f"【群聊记录】：\n{chat_text}"
        
        try:
            provider = self.context.get_using_provider()
            if not provider: return {}
            
            # 【修复 LLM 请求报错】使用 V4 标准的 contexts 字典列表传递用户消息
            req = ProviderRequest(system_prompt=system_prompt)
            req.contexts = [{"role": "user", "content": user_prompt}]
            res = await provider.text_chat(req)
            
            raw_text = getattr(res, 'completion_text', getattr(res, 'text', ''))
            clean_text = raw_text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except Exception as e:
            self.logger.error(f"分析请求失败: {e}")
            return {}

    async def _merge_old_events(self, event1: str, event2: str) -> str:
        prompt = f"""请将以下两件事合并为一句连贯的总结，用于记录人物的历史背景。字数限制在 20 字以内。
事件1：{event1}
事件2：{event2}
直接输出合并后的句子，不要任何多余文字。"""
        try:
            provider = self.context.get_using_provider()
            req = ProviderRequest(system_prompt="你是一个专门负责缩写句子的 AI 助手。")
            req.contexts = [{"role": "user", "content": prompt}]
            res = await provider.text_chat(req)
            return getattr(res, 'completion_text', getattr(res, 'text', '')).strip()
        except:
            return f"{event1} 且 {event2}" 

    # ==========================================
    #         表层 LLM 接口：生成回复
    # ==========================================

    async def _generate_reply(self, topic: str, recent_context: list, group_id: str) -> str:
        recent_summaries_db = self.db.get_summaries(group_id, "paragraph")
        recent_summaries = [row[1] for row in recent_summaries_db[-3:]] 
        background_str = "【群聊近期背景回顾】：\n- " + "\n- ".join(recent_summaries) if recent_summaries else "暂无背景。"

        involved_users = set([item['sender'] for item in recent_context])
        archives_str = "【相关群友档案参考】：\n"
        for user in involved_users:
            u_data = self.db.get_user_profile(group_id, user)
            fixed_info = ", ".join([f"{k}:{v}" for k, v in u_data["fixed_data"].items()])
            dynamic_info = "; ".join(u_data["dynamic_events"])
            if fixed_info or dynamic_info:
                archives_str += f"- {user} -> 基础: [{fixed_info}] | 近况: [{dynamic_info}]\n"

        context_str = "\n".join([f"{item['sender']}: {item['msg']}" for item in recent_context])
        
        user_prompt = f"""
{background_str}

{archives_str}

【最新群聊上下文】：
{context_str}

【你的任务】：
大家正在聊或触发话题：“{topic}”。请结合上述档案和背景，严格遵循你的系统人设插一句话。
要求：口语化、简短，若提及往事且不在档案内，请表现出“忘记了”或“不知道”。直接输出回复内容。"""

        try:
            provider = self.context.get_using_provider()
            req = ProviderRequest(system_prompt=self.skill_content)
            req.contexts = [{"role": "user", "content": user_prompt}]
            res = await provider.text_chat(req)
            return getattr(res, 'completion_text', getattr(res, 'text', '')).strip()
        except Exception as e:
            self.logger.error(f"回复生成失败: {e}")
            return ""

    # ==========================================
    #         定时任务 (凌晨4点记忆折叠)
    # ==========================================

    async def _cron_daily_compression(self):
        while True:
            now = datetime.now()
            target = now.replace(hour=4, minute=0, second=0, microsecond=0)
            if now >= target:
                target += timedelta(days=1)
            wait_seconds = (target - now).total_seconds()
            
            self.logger.info(f"记忆压缩任务挂起，将在 {wait_seconds/3600:.2f} 小时后（凌晨4点）执行...")
            await asyncio.sleep(wait_seconds)
            
            self.logger.info("凌晨 4 点到达，开始执行全局记忆压缩任务！")
            try:
                with self.db._get_conn() as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT DISTINCT group_id FROM Chat_Summaries')
                    active_groups = [row[0] for row in cursor.fetchall()]

                yesterday = datetime.now() - timedelta(days=1)
                cutoff_time = yesterday.replace(hour=23, minute=59, second=59).strftime('%Y-%m-%d %H:%M:%S')

                for group_id in active_groups:
                    await self._compress_daily_memory(group_id, cutoff_time)
                    
                    if datetime.now().day == 1:
                        one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d %H:%M:%S')
                        await self._compress_history_memory(group_id, one_year_ago)

            except Exception as e:
                self.logger.error(f"凌晨压缩任务异常: {e}")

    async def _compress_daily_memory(self, group_id: str, cutoff_time: str):
        paragraphs = self.db.get_summaries(group_id, "paragraph", before_time=cutoff_time)
        if not paragraphs: return

        texts_to_compress = [p[1] for p in paragraphs]
        self.logger.info(f"群 {group_id} 共有 {len(texts_to_compress)} 条段落待压缩...")

        chunk_size = 10 
        chunks = [texts_to_compress[i:i + chunk_size] for i in range(0, len(texts_to_compress), chunk_size)]
        
        intermediate_summaries = []
        for i, chunk in enumerate(chunks):
            chunk_text = "\n".join([f"- {text}" for text in chunk])
            prompt = f"请将以下几个群聊小结融合成一段不超过 100 字的核心摘要：\n{chunk_text}"
            res = await self._call_llm_simple(prompt)
            if res: intermediate_summaries.append(res)
            
        if intermediate_summaries:
            final_text = "\n".join([f"- {text}" for text in intermediate_summaries])
            final_prompt = f"请将今天群里的几个主要话题融合为一篇简短的【今日群聊总结】（约 150 字以内）：\n{final_text}"
            daily_summary = await self._call_llm_simple(final_prompt)
            
            if daily_summary:
                self.db.add_summary(group_id, "daily", daily_summary)
                self.db.delete_summaries(group_id, "paragraph", before_time=cutoff_time)
                self.logger.success(f"群 {group_id} 的昨日记忆已折叠并清理完毕。")

    async def _compress_history_memory(self, group_id: str, cutoff_time: str):
        dailies = self.db.get_summaries(group_id, "daily", before_time=cutoff_time)
        if not dailies: return

        texts = [d[1] for d in dailies]
        merged_text = "\n".join([f"- {text}" for text in texts])
        
        prompt = f"请将这个群一年以前的漫长聊天记录，提炼为一份约 200 字左右的【群聊编年史/老旧回忆】。重点保留大事件和群友特质：\n{merged_text[:3000]}..." 
        history_summary = await self._call_llm_simple(prompt)
        
        if history_summary:
            self.db.add_summary(group_id, "history", history_summary)
            self.db.delete_summaries(group_id, "daily", before_time=cutoff_time)
            self.logger.success(f"群 {group_id} 的年度老旧记忆已封存。")

    async def _call_llm_simple(self, prompt: str) -> str:
        try:
            provider = self.context.get_using_provider()
            req = ProviderRequest(system_prompt="你是一个擅长文本压缩的 AI。")
            req.contexts = [{"role": "user", "content": prompt}]
            res = await provider.text_chat(req)
            return getattr(res, 'completion_text', getattr(res, 'text', '')).strip()
        except Exception as e:
            self.logger.error(f"文本压缩请求失败: {e}")
            return ""

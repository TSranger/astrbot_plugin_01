import asyncio
import json
import random
import os
import re
import yaml
from datetime import datetime, timedelta

# 【修复 1】导入 AstrBot 框架原生魔改过的 logger，解决 Web 面板不显示日志的问题
from astrbot.api import logger

# AstrBot V4 SDK 导入核心库
from astrbot.api.star import Context, Star, register
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.event.filter import EventMessageType

# 导入数据库管理器
from .db_manager import MemoryDBManager 

@register("agentic_memory", "YourName", "2.0", "具备长期折叠记忆与主动行为的群聊智能体")
class AgenticMemoryPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        
        # 1. 初始化滑动窗口缓冲池
        self.message_buffers = {}
        # 用于记录每个群上次触发后保留的消息数（用于精确控制触发阈值）
        self.last_overlap_counts = {}
        
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
        logger.error(f"严重错误: 配置文件 {filepath} 缺失！")
        return {"memory_settings": {}, "skill_settings": {}}

    def _load_skill(self, filepath: str) -> str:
        if not filepath: return "你是一个普通的群友。"
        curr_dir = os.path.dirname(__file__)
        full_path = os.path.join(curr_dir, filepath)
        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        logger.warning(f"Skill 文件未找到: {full_path}")
        return "你是一个普通的群友。"

    # ==========================================
    #         事件监听与滑动窗口
    # ==========================================

    # 【修复6-新方案】使用 EventMessageType.ALL 拦截所有消息，检查 At segment
    @filter.event_message_type(filter.EventMessageType.ALL, priority=9999)
    async def intercept_at_message(self, event: AstrMessageEvent):
        """用极高优先级拦截所有消息，检查是否 @ 了机器人"""
        # 只处理群消息
        if not hasattr(event, 'message_obj'):
            return
            
        group_id = str(getattr(event.message_obj, 'group_id', getattr(event, 'session_id', '')))
        if not group_id:
            return
        
        sender_name = event.get_sender_name()
        message_text = event.message_str.strip() if event.message_str else ''
        
        if not message_text:
            return
        
        logger.debug(f"[新拦截器] 收到消息: group={group_id}, sender={sender_name}, msg={message_text[:50]}")
        
        # 检查消息链中是否有 @ 机器人的 segment
        is_at_me = False
        try:
            message_chain = event.get_messages()
            bot_qq = str(event.get_self_id())
            
            for segment in message_chain:
                if hasattr(segment, 'type') and segment.type == "At":
                    # 检查是否 @ 了机器人
                    qq = str(segment.data.get("qq", ""))
                    if qq == bot_qq:
                        is_at_me = True
                        logger.info(f"[新拦截器] 检测到 At segment，qq={qq}, bot_qq={bot_qq}")
                        break
        except Exception as e:
            logger.debug(f"[新拦截器] 检查 At segment 失败: {e}")
        
        # 也检查 is_at 属性（如果可用）
        if hasattr(event, 'is_at') and getattr(event, 'is_at'):
            is_at_me = True
            logger.info(f"[新拦截器] is_at=True")
        
        if not is_at_me:
            return  # 不是 @ 我的消息，直接返回
        
        logger.info(f"[新拦截器] 检测到 @ 消息，sender={sender_name}, message={message_text}")
        
        # 获取最近消息上下文
        recent_msgs = self.message_buffers.get(group_id, [])[-15:]
        recent_msgs.append({"sender": sender_name, "msg": message_text})
        
        # 生成符合人设的回复
        reply_text = await self._generate_reply("有人@我或提到了我", recent_msgs, group_id)
        
        if reply_text:
            await event.send(event.plain_result(reply_text))
            logger.info(f"[新拦截器] 机器人回复 {group_id} 群聊：{reply_text}")
        else:
            logger.warning(f"[新拦截器] 回复生成失败")
        
        # 关键：阻止事件继续传播
        if hasattr(event, 'stop_event'):
            event.stop_event()
            logger.info("[新拦截器] 已调用 event.stop_event()")
        elif hasattr(event, 'stop_propagation'):
            event.stop_propagation()
            logger.info("[新拦截器] 已调用 event.stop_propagation()")
        else:
            logger.warning("[新拦截器] 事件对象没有 stop_event 或 stop_propagation 方法！")
        
        return

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
            logger.debug(f"[被@触发] 检测到 @ 或提及，sender={sender_name}, message={message_text}")
            recent_msgs = self.message_buffers.get(group_id, [])[-15:]
            recent_msgs.append({"sender": sender_name, "msg": message_text})
            reply_text = await self._generate_reply("有人@我或提到了我", recent_msgs, group_id)
            if reply_text:
                await event.send(event.plain_result(reply_text))
                logger.info(f"[被@触发] 机器人回复 {group_id} 群聊：{reply_text}")
            # 尝试阻止事件传播，防止框架默认处理
            if hasattr(event, 'stop_propagation'):
                event.stop_propagation()
            return 

        # 日常潜水收集消息
        if group_id not in self.message_buffers:
            self.message_buffers[group_id] = []
            
        self.message_buffers[group_id].append({
            "sender": sender_name,
            "msg": message_text
        })

        # 滑动窗口判定触发
        # 【修复3】计算当前需要的有效消息数：threshold + 上次保留的overlap数
        last_overlap = self.last_overlap_counts.get(group_id, 0)
        effective_threshold = self.threshold + last_overlap
        
        if len(self.message_buffers[group_id]) >= effective_threshold:
            chat_batch = self.message_buffers[group_id].copy()
            # 保留 overlap 条消息到下一轮
            kept_count = min(self.overlap, len(chat_batch)) if self.overlap > 0 else 0
            self.message_buffers[group_id] = chat_batch[-kept_count:] if kept_count > 0 else []
            self.last_overlap_counts[group_id] = kept_count
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
                old_fixed = user_data["fixed_data"].copy()
                fixed = user_data["fixed_data"]
                fixed.update(updates) 
                self.db.upsert_user_profile(group_id, user_name, fixed, user_data["dynamic_events"])
                
                # 【修复4】详细记录档案更新内容
                added_keys = set(updates.keys()) - set(old_fixed.keys())
                changed_keys = {k for k in updates if k in old_fixed and old_fixed[k] != updates[k]}
                if added_keys:
                    logger.info(f"更新群友 [{user_name}] 档案：新增属性 {added_keys}")
                if changed_keys:
                    logger.info(f"更新群友 [{user_name}] 档案：修改属性 {changed_keys}")
                for key, value in updates.items():
                    if key in old_fixed:
                        logger.debug(f"  - {key}: '{old_fixed[key]}' -> '{value}'")
                    else:
                        logger.debug(f"  + {key}: '{value}'")

            should_speak = False

            if is_significant:
                target_events = analysis.get("target_users_events", {})
                for user_name, event_desc in target_events.items():
                    user_data = self.db.get_user_profile(group_id, user_name)
                    fixed = user_data["fixed_data"]
                    events = user_data["dynamic_events"]
                    events.append(event_desc)
                    
                    if len(events) > self.max_events:
                        logger.info(f"群友 {user_name} 动态事件超载，执行滚动合并...")
                        oldest_1 = events.pop(0)
                        oldest_2 = events.pop(0)
                        merged_event = await self._merge_old_events(oldest_1, oldest_2)
                        events.insert(0, merged_event)
                        
                    self.db.upsert_user_profile(group_id, user_name, fixed, events)
                
                should_speak = True
                logger.info(f"[重大事件] 检测到大事。档案已更新，强制触发发言。")

            else:
                current_prob = self.boost_prob if matches_preference else self.base_prob
                if random.random() < current_prob:
                    should_speak = True
                    log_msg = "命中兴趣偏好" if matches_preference else "小事随缘命中"
                    logger.info(f"{log_msg} ({current_prob*100}%)，准备搭茬。")

            if should_speak and interject_topic:
                reply_text = await self._generate_reply(interject_topic, chat_batch[-15:], group_id)
                if reply_text:
                    await event.send(event.plain_result(reply_text))
                    logger.info(f"[主动接话] 机器人回复 {group_id} 群聊：{reply_text}")
                else:
                    logger.warning(f"[主动接话] 命中触发条件但生成回复为空，interject_topic='{interject_topic}'")
            elif should_speak and not interject_topic:
                logger.warning(f"[主动接话] 命中触发条件但无话题切入点 (should_speak=True, interject_topic为空)")

        except Exception as e:
            logger.error(f"后台记忆处理任务崩溃: {e}")

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
    "interject_topic": "必须提取一个简短的话题切入点供机器人插话。如果没有合适的话题，可以留空字符串''。注意：即使话题很普通，也要尽量提取一个切入点，不要总是留空。"
}}"""
        user_prompt = f"【群聊记录】：\n{chat_text}"
        
        try:
            provider = self.context.get_using_provider()
            if not provider: return {}
            
            # 【修复 2】V4 标准文本请求法，并传入 context=[] 切断冗余的历史记忆干扰
            res = await provider.text_chat(
                prompt=user_prompt,
                system_prompt=system_prompt,
                context=[] 
            )
            
            raw_text = getattr(res, 'completion_text', getattr(res, 'text', ''))
            clean_text = raw_text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except Exception as e:
            logger.error(f"分析请求失败: {e}")
            return {}

    async def _merge_old_events(self, event1: str, event2: str) -> str:
        prompt = f"""请将以下两件事合并为一句连贯的总结，用于记录人物的历史背景。字数限制在 20 字以内。
事件1：{event1}
事件2：{event2}
直接输出合并后的句子，不要任何多余文字。"""
        try:
            provider = self.context.get_using_provider()
            res = await provider.text_chat(
                prompt=prompt,
                system_prompt="你是一个专门负责缩写句子的 AI 助手。",
                context=[]
            )
            return getattr(res, 'completion_text', getattr(res, 'text', '')).strip()
        except:
            return f"{event1} 且 {event2}" 

    # ==========================================
    #         表层 LLM 接口：生成回复
    # ==========================================

    async def _generate_reply(self, topic: str, recent_context: list, group_id: str) -> str:
        # 【修复5】增加调试日志
        logger.debug(f"[回复生成] 开始为群 {group_id} 生成回复，话题: {topic}")
        
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
大家正在聊或触发话题："{topic}”。请结合上述档案和背景，严格遵循你的系统人设插一句话。
要求：口语化、简短，若提及往事且不在档案内，请表现出"忘记了"或"不知道"。直接输出回复内容。"""

        try:
            provider = self.context.get_using_provider()
            if not provider:
                logger.error(f"[回复生成] 未找到 LLM 提供者！")
                return ""
            
            logger.debug(f"[回复生成] 正在调用 LLM...")
            res = await provider.text_chat(
                prompt=user_prompt,
                system_prompt=self.skill_content,
                context=[]
            )
            
            reply_text = getattr(res, 'completion_text', getattr(res, 'text', '')).strip()
            
            if not reply_text:
                logger.warning(f"[回复生成] LLM 返回空响应！raw response: {res}")
            else:
                logger.debug(f"[回复生成] LLM 返回: {reply_text[:100]}...")
            
            return reply_text
        except Exception as e:
            logger.error(f"[回复生成] 异常: {type(e).__name__}: {e}")
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
            
            logger.info(f"记忆压缩任务挂起，将在 {wait_seconds/3600:.2f} 小时后（凌晨4点）执行...")
            await asyncio.sleep(wait_seconds)
            
            logger.info("凌晨 4 点到达，开始执行全局记忆压缩任务！")
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
                logger.error(f"凌晨压缩任务异常: {e}")

    async def _compress_daily_memory(self, group_id: str, cutoff_time: str):
        paragraphs = self.db.get_summaries(group_id, "paragraph", before_time=cutoff_time)
        if not paragraphs: return

        texts_to_compress = [p[1] for p in paragraphs]
        logger.info(f"群 {group_id} 共有 {len(texts_to_compress)} 条段落待压缩...")

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
                logger.info(f"[记忆压缩] 群 {group_id} 的昨日记忆已折叠并清理完毕。")

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
            logger.info(f"[记忆压缩] 群 {group_id} 的年度老旧记忆已封存。")

    async def _call_llm_simple(self, prompt: str) -> str:
        try:
            provider = self.context.get_using_provider()
            res = await provider.text_chat(
                prompt=prompt,
                system_prompt="你是一个擅长文本压缩的 AI。",
                context=[]
            )
            return getattr(res, 'completion_text', getattr(res, 'text', '')).strip()
        except Exception as e:
            logger.error(f"文本压缩请求失败: {e}")
            return ""

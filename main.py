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

# 导入我们刚刚写的数据库管理器
from .db_manager import MemoryDBManager

@register("agentic_memory", "YourName", "2.0", "具备长期折叠记忆与主动行为的群聊智能体")
class AgenticMemoryPlugin(Star):
    # V4 中继承自 Star，且 __init__ 需要接收 context 和 config
    def __init__(self, context: Context):
        super().__init__(context)
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
    #         事件监听与滑动窗口 (实时接水管)
    # ==========================================

    # V4 拦截群聊消息的专用装饰器
    @filter.event_message_type(EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        # V4 获取群号的兼容写法
        group_id = str(getattr(event.message_obj, 'group_id', event.session_id))
        sender_name = event.get_sender_name()
        message_text = event.message_str.strip()
        
        if not message_text: return

        # 1. 拦截 @机器人 或 提及名字的消息，直接触发回复（无视滑动窗口）
        bot_names = self.config.get('skill_settings', {}).get('bot_names', [])
        is_mentioned = False
        
        # 兼容检测@或者关键词
        if hasattr(event, 'is_at') and getattr(event, 'is_at'):
            is_mentioned = True
        elif any(name in message_text for name in bot_names):
            is_mentioned = True
        
        if is_mentioned:
            # 被提及，立刻收集上下文（最多最后15条）和最近的段落总结进行回复
            recent_msgs = self.message_buffers.get(group_id, [])[-15:]
            recent_msgs.append({"sender": sender_name, "msg": message_text})
            # 唤醒“沉浸式群聊成员”进行对话
            reply_text = await self._generate_reply("有人@我或提到了我", recent_msgs, group_id)
            if reply_text:
                # V4 发信标准 API
                await event.send(event.plain_result(reply_text))
            return # 被 @ 处理完后，这条消息不再计入日常总结缓冲，防止重复触发

        # 2. 日常潜水收集消息
        if group_id not in self.message_buffers:
            self.message_buffers[group_id] = []
            
        self.message_buffers[group_id].append({
            "sender": sender_name,
            "msg": message_text
        })

        # 3. 滑动窗口判定触发
        if len(self.message_buffers[group_id]) >= self.threshold:
            # 提取送去分析的批次
            chat_batch = self.message_buffers[group_id].copy()
            
            # 窗口滑动：保留重叠部分
            self.message_buffers[group_id] = self.message_buffers[group_id][-self.overlap:] if self.overlap > 0 else []
            
            # 丢进后台分析，不阻塞当前群聊接收
            asyncio.create_task(self._process_memory_task(group_id, chat_batch, event))

    # ==========================================
    #         核心大脑：记忆提纯与决策树
    # ==========================================

    async def _process_memory_task(self, group_id: str, chat_batch: list, event: AstrMessageEvent):
        """后台任务：总结、更新档案、决定是否插话"""
        try:
            chat_text = "\n".join([f"{item['sender']}: {item['msg']}" for item in chat_batch])
            
            # 1. 呼叫“无情的档案管理员”提取 JSON
            llm_result = await self._call_llm_for_analysis(chat_text, len(chat_batch))
            if not llm_result: return

            analysis = llm_result.get("topic_analysis", {})
            summary_text = analysis.get("summary", "无有效总结")
            is_significant = analysis.get("is_significant", False)
            matches_preference = analysis.get("matches_preference", False)
            interject_topic = llm_result.get("interject_topic", "")
            
            # 2. 存储当前段落总结到时间轴 (Level 1)
            self.db.add_summary(group_id, "paragraph", summary_text, is_significant)

            # 3. 档案更新机制 (带原文校验的 UPSERT)
            # a. 固定属性更新
            profile_updates = analysis.get("profile_updates", {})
            for user_name, updates in profile_updates.items():
                if not isinstance(updates, dict) or not updates: continue
                # 从 DB 获取现有档案
                user_data = self.db.get_user_profile(group_id, user_name)
                fixed = user_data["fixed_data"]
                fixed.update(updates) # 合并新属性
                # 写回 DB
                self.db.upsert_user_profile(group_id, user_name, fixed, user_data["dynamic_events"])
                logger.info(f"更新了群友 {user_name} 的固定档案。")

            should_speak = False

            # 4. 事件滚动压缩与插话决策
            if is_significant:
                # 【大事分支】
                target_events = analysis.get("target_users_events", {})
                for user_name, event_desc in target_events.items():
                    user_data = self.db.get_user_profile(group_id, user_name)
                    fixed = user_data["fixed_data"]
                    events = user_data["dynamic_events"]
                    
                    events.append(event_desc)
                    
                    # 滚动压缩逻辑：超过阈值时，触发合并任务
                    if len(events) > self.max_events:
                        logger.info(f"群友 {user_name} 动态事件超载，执行滚动合并...")
                        # 弹出最老的两条记录
                        oldest_1 = events.pop(0)
                        oldest_2 = events.pop(0)
                        # 调用合并工具函数
                        merged_event = await self._merge_old_events(oldest_1, oldest_2)
                        # 将合并后的总结插回头部
                        events.insert(0, merged_event)
                        
                    self.db.upsert_user_profile(group_id, user_name, fixed, events)
                
                should_speak = True
                logger.success(f"检测到大事。档案已更新，强制触发发言。")

            else:
                # 【小事分支】根据偏好加权掷骰子
                current_prob = self.boost_prob if matches_preference else self.base_prob
                if random.random() < current_prob:
                    should_speak = True
                    log_msg = "命中兴趣偏好" if matches_preference else "小事随缘命中"
                    logger.info(f"{log_msg} ({current_prob*100}%)，准备搭茬。")

            # 5. 执行插话动作
            if should_speak and interject_topic:
                reply_text = await self._generate_reply(interject_topic, chat_batch[-15:], group_id)
                if reply_text:
                    await event.send(event.plain_result(reply_text))

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
{self.skill_content[:500]} # 取前500字作为兴趣参考

请严格输出 JSON 格式（绝不能带有 ```json 标记）：
{{
    "topic_analysis": {{
        "summary": "一句话概括讨论内容",
        "is_significant": false, // 是否为重大事件
        "matches_preference": false, // 是否命中了机器人的兴趣爱好
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
            # V4 获取大模型 Provider 的标准方法
            provider = self.context.get_using_provider()
            if not provider: return {}
            
            # V4 的 ProviderRequest 对象
            req = ProviderRequest(system_prompt=system_prompt, text=user_prompt)
            res = await provider.text_chat(req)
            
            # 兼容获取 V4 的返回文本
            raw_text = getattr(res, 'completion_text', getattr(res, 'text', ''))
            clean_text = raw_text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except Exception as e:
            logger.error(f"分析请求失败: {e}")
            return {}

    async def _merge_old_events(self, event1: str, event2: str) -> str:
        """滚动压缩：将两条旧事件合并为一条精简总结"""
        prompt = f"""请将以下两件事合并为一句连贯的总结，用于记录人物的历史背景。字数限制在 20 字以内。
事件1：{event1}
事件2：{event2}
直接输出合并后的句子，不要任何多余文字。"""
        try:
            provider = self.context.get_using_provider()
            req = ProviderRequest(text=prompt)
            res = await provider.text_chat(req)
            return getattr(res, 'completion_text', getattr(res, 'text', '')).strip()
        except:
            return f"{event1} 且 {event2}" # 失败时的 fallback

    # ==========================================
    #         表层 LLM 接口：生成回复
    # ==========================================

    async def _generate_reply(self, topic: str, recent_context: list, group_id: str) -> str:
        # 1. 捞取最近的段落总结 (提供远期背景)
        recent_summaries_db = self.db.get_summaries(group_id, "paragraph")
        # 取最后 3 条段落总结
        recent_summaries = [row[1] for row in recent_summaries_db[-3:]] 
        background_str = "【群聊近期背景回顾】：\n- " + "\n- ".join(recent_summaries) if recent_summaries else "暂无背景。"

        # 2. 捞取参与者的群友档案
        involved_users = set([item['sender'] for item in recent_context])
        archives_str = "【相关群友档案参考】：\n"
        for user in involved_users:
            u_data = self.db.get_user_profile(group_id, user)
            fixed_info = ", ".join([f"{k}:{v}" for k, v in u_data["fixed_data"].items()])
            dynamic_info = "; ".join(u_data["dynamic_events"])
            if fixed_info or dynamic_info:
                archives_str += f"- {user} -> 基础: [{fixed_info}] | 近况: [{dynamic_info}]\n"

        # 3. 组装当前聊天
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
            req = ProviderRequest(
                system_prompt=self.skill_content,
                text=user_prompt
            )
            res = await provider.text_chat(req)
            return getattr(res, 'completion_text', getattr(res, 'text', '')).strip()
        except Exception as e:
            logger.error(f"回复生成失败: {e}")
            return ""

    # ==========================================
    #         定时任务 (凌晨4点记忆折叠)
    # ==========================================

    async def _cron_daily_compression(self):
        """死循环任务，每天凌晨 4 点触发记忆折叠"""
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
                # 获取所有在数据库中有记录的群号 (通过 DISTINCT 过滤)
                with self.db._get_conn() as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT DISTINCT group_id FROM Chat_Summaries')
                    active_groups = [row[0] for row in cursor.fetchall()]

                yesterday = datetime.now() - timedelta(days=1)
                # 设定时间界限：昨天 23:59:59 之前的都算昨天
                cutoff_time = yesterday.replace(hour=23, minute=59, second=59).strftime('%Y-%m-%d %H:%M:%S')

                for group_id in active_groups:
                    await self._compress_daily_memory(group_id, cutoff_time)
                    
                    # 每月 1 号凌晨 4 点，额外执行一次年度老旧历史压缩
                    if datetime.now().day == 1:
                        one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d %H:%M:%S')
                        await self._compress_history_memory(group_id, one_year_ago)

            except Exception as e:
                logger.error(f"凌晨压缩任务异常: {e}")

    async def _compress_daily_memory(self, group_id: str, cutoff_time: str):
        """Level 1 -> Level 2：将段落总结浓缩为【一天总结】"""
        # 1. 捞出所有待压缩的段落总结
        paragraphs = self.db.get_summaries(group_id, "paragraph", before_time=cutoff_time)
        if not paragraphs:
            return

        texts_to_compress = [p[1] for p in paragraphs]
        logger.info(f"群 {group_id} 共有 {len(texts_to_compress)} 条段落待压缩...")

        # 2. Map-Reduce 核心逻辑：防止一次性喂给大模型太多导致 OOM
        chunk_size = 10  # 每 10 个段落合成一个小节
        chunks = [texts_to_compress[i:i + chunk_size] for i in range(0, len(texts_to_compress), chunk_size)]
        
        intermediate_summaries = []
        for i, chunk in enumerate(chunks):
            chunk_text = "\n".join([f"- {text}" for text in chunk])
            prompt = f"请将以下几个群聊小结融合成一段不超过 100 字的核心摘要：\n{chunk_text}"
            res = await self._call_llm_simple(prompt)
            if res: intermediate_summaries.append(res)
            
        # 3. 最终融合 (Reduce)
        if intermediate_summaries:
            final_text = "\n".join([f"- {text}" for text in intermediate_summaries])
            final_prompt = f"请将今天群里的几个主要话题融合为一篇简短的【今日群聊总结】（约 150 字以内）：\n{final_text}"
            daily_summary = await self._call_llm_simple(final_prompt)
            
            if daily_summary:
                # 存入 Level 2
                self.db.add_summary(group_id, "daily", daily_summary)
                # 清理战场的 Level 1 旧数据
                self.db.delete_summaries(group_id, "paragraph", before_time=cutoff_time)
                logger.success(f"群 {group_id} 的昨日记忆已折叠并清理完毕。")

    async def _compress_history_memory(self, group_id: str, cutoff_time: str):
        """Level 2 -> Level 3：将一年前的一天总结浓缩为【老旧历史】"""
        dailies = self.db.get_summaries(group_id, "daily", before_time=cutoff_time)
        if not dailies:
            return

        texts = [d[1] for d in dailies]
        merged_text = "\n".join([f"- {text}" for text in texts])
        
        prompt = f"请将这个群一年以前的漫长聊天记录，提炼为一份约 200 字左右的【群聊编年史/老旧回忆】。重点保留大事件和群友特质：\n{merged_text[:3000]}..." # 截断保护
        history_summary = await self._call_llm_simple(prompt)
        
        if history_summary:
            self.db.add_summary(group_id, "history", history_summary)
            self.db.delete_summaries(group_id, "daily", before_time=cutoff_time)
            logger.success(f"群 {group_id} 的年度老旧记忆已封存。")

    async def _call_llm_simple(self, prompt: str) -> str:
        """纯文本大模型调用封装（专门用于文本压缩，不需要 JSON）"""
        try:
            provider = self.context.get_using_provider()
            req = ProviderRequest(text=prompt)
            res = await provider.text_chat(req)
            return getattr(res, 'completion_text', getattr(res, 'text', '')).strip()
        except Exception as e:
            logger.error(f"文本压缩请求失败: {e}")
            return ""

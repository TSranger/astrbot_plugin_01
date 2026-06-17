'''
数据库模块。用于存储历史记录和群友档案
'''

import sqlite3
import json
import os
from datetime import datetime
from loguru import logger

class MemoryDBManager:
    def __init__(self, db_path="data/agentic_memory.db"):
        self.db_path = db_path
        # 确保数据存放的文件夹存在
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()

    def _get_conn(self):
        # check_same_thread=False 允许在异步的 AstrBot 环境中跨线程调用数据库
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        """初始化数据库表结构"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            # 1. 创建聊天总结时间轴表 (日记本)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Chat_Summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id TEXT NOT NULL,
                    summary_type TEXT NOT NULL,  -- 'paragraph', 'daily', 'history'
                    content TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    is_significant BOOLEAN DEFAULT 0
                )
            ''')
            # 建立联合索引，极大加速凌晨 4 点查询特定群、特定类型总结的速度
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_group_type ON Chat_Summaries(group_id, summary_type)')
            
            # 2. 创建群友档案表 (二维画像)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS User_Profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id TEXT NOT NULL,
                    nickname TEXT NOT NULL,
                    fixed_data TEXT DEFAULT '{}',    -- 存放固定属性的 JSON 字符串
                    dynamic_events TEXT DEFAULT '[]', -- 存放近期大事件的 JSON 字符串
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(group_id, nickname)       -- 确保同一个群的同一个昵称只有一条记录
                )
            ''')
            conn.commit()
            logger.info(f"SQLite 数据库初始化完成: {self.db_path}")

    # ==========================================
    #         聊天总结 (时间轴) 相关操作
    # ==========================================

    def add_summary(self, group_id: str, summary_type: str, content: str, is_significant: bool = False):
        """添加一条新的总结 (段落/一天/历史)"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO Chat_Summaries (group_id, summary_type, content, is_significant) VALUES (?, ?, ?, ?)',
                (group_id, summary_type, content, is_significant)
            )
            conn.commit()

    def get_summaries(self, group_id: str, summary_type: str, before_time: str = None) -> list:
        """获取总结记录 (常用于凌晨4点读取昨天的段落做合并)"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            if before_time:
                cursor.execute(
                    'SELECT id, content, created_at FROM Chat_Summaries WHERE group_id = ? AND summary_type = ? AND created_at < ? ORDER BY created_at ASC',
                    (group_id, summary_type, before_time)
                )
            else:
                cursor.execute(
                    'SELECT id, content, created_at FROM Chat_Summaries WHERE group_id = ? AND summary_type = ? ORDER BY created_at ASC',
                    (group_id, summary_type)
                )
            return cursor.fetchall()

    def delete_summaries(self, group_id: str, summary_type: str, before_time: str = None):
        """删除旧的总结记录 (合并后用来清理战场)"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            if before_time:
                cursor.execute('DELETE FROM Chat_Summaries WHERE group_id = ? AND summary_type = ? AND created_at < ?', 
                               (group_id, summary_type, before_time))
            else:
                cursor.execute('DELETE FROM Chat_Summaries WHERE group_id = ? AND summary_type = ?', 
                               (group_id, summary_type))
            conn.commit()

    # ==========================================
    #         群友档案 (二维画像) 相关操作
    # ==========================================

    def get_user_profile(self, group_id: str, nickname: str) -> dict:
        """获取某个群友的完整档案，自动将 JSON 字符串反序列化为 Python 字典"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT fixed_data, dynamic_events FROM User_Profiles WHERE group_id = ? AND nickname = ?', 
                           (group_id, nickname))
            row = cursor.fetchone()
            if row:
                return {
                    "fixed_data": json.loads(row[0]), 
                    "dynamic_events": json.loads(row[1])
                }
            return {"fixed_data": {}, "dynamic_events": []}

    def upsert_user_profile(self, group_id: str, nickname: str, fixed_data: dict, dynamic_events: list):
        """
        更新群友档案 (插入或更新)。
        无论这个人是第一次被记录，还是更新已有记录，统一调这个方法。
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            # SQLite 神技：ON CONFLICT ... DO UPDATE，遇到主键/唯一约束冲突时自动转为 UPDATE
            cursor.execute('''
                INSERT INTO User_Profiles (group_id, nickname, fixed_data, dynamic_events, last_updated)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(group_id, nickname) DO UPDATE SET
                    fixed_data = excluded.fixed_data,
                    dynamic_events = excluded.dynamic_events,
                    last_updated = CURRENT_TIMESTAMP
            ''', (
                group_id, 
                nickname, 
                json.dumps(fixed_data, ensure_ascii=False), 
                json.dumps(dynamic_events, ensure_ascii=False)
            ))
            conn.commit()

"""插件记忆摘要、用户画像和主动任务状态的数据库模块。"""

import json
import os
import sqlite3
from typing import Any

from loguru import logger


class MemoryDBManager:
    """管理摘要、画像和主动任务状态的 SQLite 持久化。"""

    def __init__(self, db_path: str = "data/agentic_memory.db"):
        """初始化 SQLite 数据库管理器。

        Args:
            db_path: SQLite 数据库文件路径。
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """创建一次操作所用的 SQLite 连接。

        Returns:
            允许跨线程访问的 SQLite 连接。
        """
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self) -> None:
        """初始化表结构并执行轻量级 schema 迁移。"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS Chat_Summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id TEXT NOT NULL,
                    summary_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    memory_key TEXT DEFAULT '',
                    period_start DATETIME DEFAULT NULL,
                    period_end DATETIME DEFAULT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    is_significant BOOLEAN DEFAULT 0,
                    rolled_up_at DATETIME DEFAULT NULL,
                    cleanup_after DATETIME DEFAULT NULL,
                    source_count INTEGER DEFAULT 0
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_group_type ON Chat_Summaries(group_id, summary_type)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_group_type_key ON Chat_Summaries(group_id, summary_type, memory_key)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_group_type_period ON Chat_Summaries(group_id, summary_type, period_start, period_end)"
            )

            existing_columns = {
                row[1]
                for row in cursor.execute(
                    "PRAGMA table_info(Chat_Summaries)"
                ).fetchall()
            }
            required_columns = {
                "memory_key": "TEXT DEFAULT ''",
                "period_start": "DATETIME DEFAULT NULL",
                "period_end": "DATETIME DEFAULT NULL",
                "rolled_up_at": "DATETIME DEFAULT NULL",
                "cleanup_after": "DATETIME DEFAULT NULL",
                "source_count": "INTEGER DEFAULT 0",
            }
            for column_name, column_type in required_columns.items():
                if column_name not in existing_columns:
                    cursor.execute(
                        f"ALTER TABLE Chat_Summaries ADD COLUMN {column_name} {column_type}"
                    )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS User_Profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    nickname TEXT NOT NULL,
                    fixed_data TEXT DEFAULT '{}',
                    dynamic_events TEXT DEFAULT '[]',
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(group_id, nickname)
                )
                """
            )

            existing_user_profile_columns = {
                row[1]
                for row in cursor.execute("PRAGMA table_info(User_Profiles)").fetchall()
            }
            if "user_id" not in existing_user_profile_columns:
                cursor.execute(
                    "ALTER TABLE User_Profiles ADD COLUMN user_id TEXT NOT NULL DEFAULT ''"
                )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS Nickname_History (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    nickname TEXT NOT NULL,
                    first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(group_id, user_id, nickname)
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS Proactive_Task_State (
                    group_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    last_run_date TEXT NOT NULL,
                    last_send_time TEXT DEFAULT NULL,
                    PRIMARY KEY (group_id, task_id)
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS News_Selfie_Task_State (
                    task_id TEXT PRIMARY KEY,
                    last_run_date TEXT NOT NULL,
                    last_run_time TEXT DEFAULT NULL,
                    last_status TEXT DEFAULT ''
                )
                """
            )
            conn.commit()
            logger.info(f"SQLite database initialized: {self.db_path}")

    def add_summary(
        self,
        group_id: str,
        summary_type: str,
        content: str,
        is_significant: bool = False,
        memory_key: str = "",
        period_start: str | None = None,
        period_end: str | None = None,
        rolled_up_at: str | None = None,
        cleanup_after: str | None = None,
        source_count: int = 0,
    ) -> int:
        """插入一条摘要记录。

        Args:
            group_id: 群号。
            summary_type: 摘要层级，例如 ``paragraph`` 或 ``daily``。
            content: 摘要文本。
            is_significant: 是否标记为重要摘要。
            memory_key: 摘要所属的逻辑时间桶。
            period_start: 覆盖区间的开始时间。
            period_end: 覆盖区间的结束时间。
            rolled_up_at: 被继续归纳到上层的时间。
            cleanup_after: 这条记录最早可删除的时间。
            source_count: 参与归纳的源记录数。

        Returns:
            插入后的行 ID。
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO Chat_Summaries (
                    group_id,
                    summary_type,
                    content,
                    memory_key,
                    period_start,
                    period_end,
                    is_significant,
                    rolled_up_at,
                    cleanup_after,
                    source_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    group_id,
                    summary_type,
                    content,
                    memory_key,
                    period_start,
                    period_end,
                    is_significant,
                    rolled_up_at,
                    cleanup_after,
                    source_count,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def get_summaries(
        self,
        group_id: str,
        summary_type: str,
        before_time: str | None = None,
        memory_key: str | None = None,
        limit: int | None = None,
        order_desc: bool = False,
        only_unrolled: bool = False,
    ) -> list[tuple[Any, ...]]:
        """查询某个群、某个层级的摘要。

        Args:
            group_id: 群号。
            summary_type: 摘要层级。
            before_time: 可选的 ``created_at`` 上限。
            memory_key: 可选的精确 ``memory_key`` 过滤。
            limit: 可选行数限制。
            order_desc: 是否按时间倒序。
            only_unrolled: 是否只取尚未向上归纳的记录。

        Returns:
            摘要行列表。
        """
        conditions = ["group_id = ?", "summary_type = ?"]
        params: list[Any] = [group_id, summary_type]

        if before_time:
            conditions.append("created_at < ?")
            params.append(before_time)
        if memory_key is not None:
            conditions.append("memory_key = ?")
            params.append(memory_key)
        if only_unrolled:
            conditions.append("rolled_up_at IS NULL")

        order_clause = "DESC" if order_desc else "ASC"
        sql = (
            "SELECT id, content, created_at, is_significant, memory_key, period_start, "
            "period_end, rolled_up_at, cleanup_after, source_count "
            "FROM Chat_Summaries WHERE "
            + " AND ".join(conditions)
            + f" ORDER BY created_at {order_clause}, id {order_clause}"
        )
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return cursor.fetchall()

    def get_rollup_candidates(
        self,
        group_id: str,
        summary_type: str,
        period_start: str | None = None,
        period_end: str | None = None,
        exclude_memory_keys: list[str] | None = None,
    ) -> list[tuple[Any, ...]]:
        """按时间覆盖区间查询可继续向上归纳的记录。

        Args:
            group_id: 群号。
            summary_type: 源摘要层级。
            period_start: 可选覆盖区间下界。
            period_end: 可选覆盖区间上界。
            exclude_memory_keys: 可选排除的 memory_key。

        Returns:
            按区间和创建时间排序的候选行。
        """
        conditions = ["group_id = ?", "summary_type = ?", "rolled_up_at IS NULL"]
        params: list[Any] = [group_id, summary_type]

        if period_start is not None:
            conditions.append("COALESCE(period_end, created_at) >= ?")
            params.append(period_start)
        if period_end is not None:
            conditions.append("COALESCE(period_start, created_at) < ?")
            params.append(period_end)
        if exclude_memory_keys:
            placeholders = ", ".join("?" for _ in exclude_memory_keys)
            conditions.append(f"COALESCE(memory_key, '') NOT IN ({placeholders})")
            params.extend(exclude_memory_keys)

        sql = (
            "SELECT id, content, created_at, is_significant, memory_key, period_start, "
            "period_end, rolled_up_at, cleanup_after, source_count "
            "FROM Chat_Summaries WHERE "
            + " AND ".join(conditions)
            + " ORDER BY COALESCE(period_start, created_at) ASC, created_at ASC, id ASC"
        )
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return cursor.fetchall()

    def get_summary_by_memory_key(
        self,
        group_id: str,
        summary_type: str,
        memory_key: str,
    ) -> tuple[Any, ...] | None:
        """按 memory_key 取最新的一条摘要。

        Args:
            group_id: 群号。
            summary_type: 摘要层级。
            memory_key: 逻辑 memory_key。

        Returns:
            摘要行或 ``None``。
        """
        rows = self.get_summaries(
            group_id,
            summary_type,
            memory_key=memory_key,
            limit=1,
            order_desc=True,
        )
        return rows[0] if rows else None

    def update_summary_content(
        self,
        summary_id: int,
        content: str,
        is_significant: bool = False,
        period_start: str | None = None,
        period_end: str | None = None,
        source_count: int = 0,
    ) -> None:
        """更新一条已有摘要记录。

        Args:
            summary_id: 要更新的行 ID。
            content: 新摘要文本。
            is_significant: 更新后的重要性标记。
            period_start: 更新后的覆盖开始时间。
            period_end: 更新后的覆盖结束时间。
            source_count: 更新后的源记录数。
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE Chat_Summaries
                SET content = ?,
                    is_significant = ?,
                    period_start = ?,
                    period_end = ?,
                    source_count = ?
                WHERE id = ?
                """,
                (
                    content,
                    is_significant,
                    period_start,
                    period_end,
                    source_count,
                    summary_id,
                ),
            )
            conn.commit()

    def mark_summaries_rolled_up(
        self,
        summary_ids: list[int],
        rolled_up_at: str,
        cleanup_after: str | None = None,
    ) -> None:
        """把源摘要标记为已经完成向上归纳。

        Args:
            summary_ids: 源行 ID 列表。
            rolled_up_at: 归纳时间戳。
            cleanup_after: 最早可删除时间戳。
        """
        if not summary_ids:
            return
        placeholders = ", ".join("?" for _ in summary_ids)
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                UPDATE Chat_Summaries
                SET rolled_up_at = ?,
                    cleanup_after = COALESCE(?, cleanup_after)
                WHERE id IN ({placeholders})
                """,
                [rolled_up_at, cleanup_after, *summary_ids],
            )
            conn.commit()

    def delete_summaries(
        self, group_id: str, summary_type: str, before_time: str | None = None
    ) -> None:
        """按层级和可选时间上限删除摘要。

        Args:
            group_id: 群号。
            summary_type: 摘要层级。
            before_time: 可选的 ``created_at`` 上限。
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            if before_time:
                cursor.execute(
                    "DELETE FROM Chat_Summaries WHERE group_id = ? AND summary_type = ? AND created_at < ?",
                    (group_id, summary_type, before_time),
                )
            else:
                cursor.execute(
                    "DELETE FROM Chat_Summaries WHERE group_id = ? AND summary_type = ?",
                    (group_id, summary_type),
                )
            conn.commit()

    def delete_ready_summaries(
        self,
        group_id: str,
        summary_type: str,
        now_text: str,
    ) -> int:
        """删除已经超过延迟清理时间的记录。

        Args:
            group_id: 群号。
            summary_type: 摘要层级。
            now_text: 当前时间文本。

        Returns:
            删除的行数。
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM Chat_Summaries
                WHERE group_id = ?
                  AND summary_type = ?
                  AND rolled_up_at IS NOT NULL
                  AND cleanup_after IS NOT NULL
                  AND cleanup_after <= ?
                """,
                (group_id, summary_type, now_text),
            )
            conn.commit()
            return int(cursor.rowcount or 0)

    def list_group_ids_with_summaries(self) -> list[str]:
        """列出当前存在摘要数据的所有群。

        Returns:
            去重后的群号列表。
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT DISTINCT group_id FROM Chat_Summaries ORDER BY group_id ASC"
            )
            return [str(row[0]) for row in cursor.fetchall()]

    def search_summaries_by_keywords(
        self,
        group_id: str,
        summary_types: list[str],
        keywords: list[str],
        limit: int,
    ) -> list[tuple[Any, ...]]:
        """按轻量关键词匹配搜索摘要。

        Args:
            group_id: 群号。
            summary_types: 候选摘要层级。
            keywords: 用于 ``LIKE`` 匹配的关键词。
            limit: 最大返回行数。

        Returns:
            匹配到的摘要行。
        """
        if not summary_types or not keywords or limit <= 0:
            return []

        type_placeholders = ", ".join("?" for _ in summary_types)
        keyword_conditions: list[str] = []
        params: list[Any] = [group_id, *summary_types]

        for keyword in keywords:
            keyword_conditions.append("content LIKE ?")
            params.append(f"%{keyword}%")

        sql = (
            "SELECT id, group_id, summary_type, content, created_at, memory_key, period_start, "
            "period_end, is_significant, rolled_up_at, cleanup_after, source_count "
            "FROM Chat_Summaries WHERE group_id = ? AND summary_type IN ("
            + type_placeholders
            + ") AND ("
            + " OR ".join(keyword_conditions)
            + ") ORDER BY is_significant DESC, created_at DESC, id DESC LIMIT ?"
        )
        params.append(limit)

        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return cursor.fetchall()

    def get_neighbor_summaries(
        self,
        group_id: str,
        anchor_start: str,
        anchor_end: str,
        limit: int,
        exclude_ids: list[int] | None = None,
    ) -> list[tuple[Any, ...]]:
        """查询覆盖时间与目标窗口重叠的摘要。

        Args:
            group_id: 群号。
            anchor_start: 窗口开始时间。
            anchor_end: 窗口结束时间。
            limit: 最大返回行数。
            exclude_ids: 可选排除的行 ID。

        Returns:
            锚点窗口附近的相邻摘要行。
        """
        if limit <= 0:
            return []

        conditions = [
            "group_id = ?",
            "COALESCE(period_end, created_at) >= ?",
            "COALESCE(period_start, created_at) <= ?",
        ]
        params: list[Any] = [group_id, anchor_start, anchor_end]

        if exclude_ids:
            placeholders = ", ".join("?" for _ in exclude_ids)
            conditions.append(f"id NOT IN ({placeholders})")
            params.extend(exclude_ids)

        sql = (
            "SELECT id, group_id, summary_type, content, created_at, memory_key, period_start, "
            "period_end, is_significant, rolled_up_at, cleanup_after, source_count "
            "FROM Chat_Summaries WHERE "
            + " AND ".join(conditions)
            + " ORDER BY is_significant DESC, COALESCE(period_start, created_at) DESC, id DESC LIMIT ?"
        )
        params.append(limit)

        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return cursor.fetchall()

    def get_user_profile(self, group_id: str, nickname: str) -> dict[str, Any]:
        """获取一条已解码 JSON 字段的用户画像。

        Args:
            group_id: 群号。
            nickname: 用户昵称。

        Returns:
            包含固定信息和动态记忆的画像字典。
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT fixed_data, dynamic_events FROM User_Profiles WHERE group_id = ? AND nickname = ?",
                (group_id, nickname),
            )
            row = cursor.fetchone()
            if row:
                return {
                    "fixed_data": json.loads(row[0]),
                    "dynamic_events": json.loads(row[1]),
                }
            return {"fixed_data": {}, "dynamic_events": []}

    def get_user_profile_by_user_id(
        self, group_id: str, user_id: str
    ) -> dict[str, Any]:
        """用 user_id 查询用户画像。

        Args:
            group_id: 群号。
            user_id: 用户 QQ 号。

        Returns:
            包含固定信息和动态记忆的画像字典。
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT fixed_data, dynamic_events FROM User_Profiles WHERE group_id = ? AND user_id = ? AND user_id != '' ORDER BY last_updated DESC LIMIT 1",
                (group_id, user_id),
            )
            row = cursor.fetchone()
            if row:
                return {
                    "fixed_data": json.loads(row[0]),
                    "dynamic_events": json.loads(row[1]),
                }
            return {"fixed_data": {}, "dynamic_events": []}

    def record_nickname(self, group_id: str, user_id: str, nickname: str) -> None:
        """记录一条昵称→user_id 映射。

        Args:
            group_id: 群号。
            user_id: 用户 QQ 号。
            nickname: 用户当前昵称。
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO Nickname_History (group_id, user_id, nickname, last_seen)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(group_id, user_id, nickname) DO UPDATE SET
                    last_seen = CURRENT_TIMESTAMP
                """,
                (group_id, user_id, nickname),
            )
            conn.commit()

    def get_user_id_by_nickname(self, group_id: str, nickname: str) -> str | None:
        """从旧昵称反查 user_id。

        Args:
            group_id: 群号。
            nickname: 昵称。

        Returns:
            user_id 或 ``None``。
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT user_id FROM Nickname_History WHERE group_id = ? AND nickname = ? ORDER BY last_seen DESC LIMIT 1",
                (group_id, nickname),
            )
            row = cursor.fetchone()
            if row:
                return str(row[0])

            cursor.execute(
                "SELECT user_id FROM User_Profiles WHERE group_id = ? AND nickname = ? AND user_id != '' LIMIT 1",
                (group_id, nickname),
            )
            row = cursor.fetchone()
            return str(row[0]) if row else None

    def get_nickname_history(self, group_id: str, user_id: str) -> list[dict[str, str]]:
        """查某用户的所有历史昵称。

        Args:
            group_id: 群号。
            user_id: 用户 QQ 号。

        Returns:
            包含 nickname、first_seen、last_seen 的字典列表。
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT nickname, first_seen, last_seen FROM Nickname_History WHERE group_id = ? AND user_id = ? ORDER BY last_seen DESC",
                (group_id, user_id),
            )
            return [
                {
                    "nickname": str(row[0]),
                    "first_seen": str(row[1] or ""),
                    "last_seen": str(row[2] or ""),
                }
                for row in cursor.fetchall()
            ]

    def upsert_user_profile(
        self,
        group_id: str,
        nickname: str,
        fixed_data: dict,
        dynamic_events: list,
        user_id: str = "",
    ) -> None:
        """插入或更新一条用户画像。

        Args:
            group_id: 群号。
            nickname: 用户昵称。
            fixed_data: 稳定画像字段。
            dynamic_events: 最近动态事件。
            user_id: 用户的 QQ 号。
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO User_Profiles (group_id, user_id, nickname, fixed_data, dynamic_events, last_updated)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(group_id, nickname) DO UPDATE SET
                    user_id = COALESCE(NULLIF(excluded.user_id, ''), user_id),
                    fixed_data = excluded.fixed_data,
                    dynamic_events = excluded.dynamic_events,
                    last_updated = CURRENT_TIMESTAMP
                """,
                (
                    group_id,
                    user_id,
                    nickname,
                    json.dumps(fixed_data, ensure_ascii=False),
                    json.dumps(dynamic_events, ensure_ascii=False),
                ),
            )
            conn.commit()

    def get_proactive_task_state(self, group_id: str, task_id: str) -> dict[str, str]:
        """获取主动任务持久化的最后执行状态。

        Args:
            group_id: 群号。
            task_id: 主动任务 ID。

        Returns:
            包含 ``last_run_date`` 和 ``last_send_time`` 的字典。
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT last_run_date, last_send_time FROM Proactive_Task_State "
                "WHERE group_id = ? AND task_id = ?",
                (group_id, task_id),
            )
            row = cursor.fetchone()
            if row:
                return {
                    "last_run_date": str(row[0] or ""),
                    "last_send_time": str(row[1] or ""),
                }
            return {"last_run_date": "", "last_send_time": ""}

    def upsert_proactive_task_state(
        self,
        group_id: str,
        task_id: str,
        last_run_date: str,
        last_send_time: str,
    ) -> None:
        """在成功发送后写入或更新主动任务状态。

        Args:
            group_id: 群号。
            task_id: 主动任务 ID。
            last_run_date: 最近一次成功运行日期，格式为 ``YYYY-MM-DD``。
            last_send_time: 最近一次成功发送的 ISO 时间戳。
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO Proactive_Task_State (group_id, task_id, last_run_date, last_send_time)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(group_id, task_id) DO UPDATE SET
                    last_run_date = excluded.last_run_date,
                    last_send_time = excluded.last_send_time
                """,
                (group_id, task_id, last_run_date, last_send_time),
            )
            conn.commit()

    def get_news_selfie_task_state(self, task_id: str) -> dict[str, str]:
        """获取新闻自拍任务的最近执行状态。

        Args:
            task_id: 任务 ID。

        Returns:
            包含 ``last_run_date``、``last_run_time`` 和 ``last_status`` 的字典。
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT last_run_date, last_run_time, last_status FROM News_Selfie_Task_State WHERE task_id = ?",
                (task_id,),
            )
            row = cursor.fetchone()
            if row:
                return {
                    "last_run_date": str(row[0] or ""),
                    "last_run_time": str(row[1] or ""),
                    "last_status": str(row[2] or ""),
                }
            return {"last_run_date": "", "last_run_time": "", "last_status": ""}

    def upsert_news_selfie_task_state(
        self,
        task_id: str,
        last_run_date: str,
        last_run_time: str | None = None,
        last_status: str = "",
    ) -> None:
        """写入新闻自拍任务的最近执行状态。

        Args:
            task_id: 任务 ID。
            last_run_date: 最近执行日期，格式为 ``YYYY-MM-DD``。
            last_run_time: 最近执行时间文本。
            last_status: 最近执行结果状态。
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO News_Selfie_Task_State (task_id, last_run_date, last_run_time, last_status)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    last_run_date = excluded.last_run_date,
                    last_run_time = excluded.last_run_time,
                    last_status = excluded.last_status
                """,
                (task_id, last_run_date, last_run_time, last_status),
            )
            conn.commit()

    def get_memory_stats(self, group_id: str) -> dict[str, Any]:
        """获取某个群各层记忆的结构化诊断统计。

        Args:
            group_id: 群号。

        Returns:
            包含各层数量、时间范围和归纳比例的字典。
        """
        stats: dict[str, Any] = {}
        summary_types = ["paragraph", "daily", "month", "year", "history"]

        with self._get_conn() as conn:
            cursor = conn.cursor()
            for stype in summary_types:
                cursor.execute(
                    "SELECT COUNT(*), "
                    "COUNT(CASE WHEN rolled_up_at IS NULL THEN 1 END), "
                    "MIN(created_at), MAX(created_at) "
                    "FROM Chat_Summaries WHERE group_id = ? AND summary_type = ?",
                    (group_id, stype),
                )
                row = cursor.fetchone()
                total = int(row[0] or 0)
                unrolled = int(row[1] or 0)
                stats[stype] = {
                    "total": total,
                    "unrolled": unrolled,
                    "rolled": total - unrolled,
                    "earliest": str(row[2] or ""),
                    "latest": str(row[3] or ""),
                }

            cursor.execute(
                "SELECT COUNT(*) FROM User_Profiles WHERE group_id = ?",
                (group_id,),
            )
            stats["user_profile_count"] = int(cursor.fetchone()[0] or 0)

        return stats

    def get_all_user_profiles(self, group_id: str) -> list[dict[str, Any]]:
        """获取某个群的全部用户画像。

        Args:
            group_id: 群号。

        Returns:
            包含 nickname、fixed_data 和 dynamic_events 的画像列表。
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT nickname, fixed_data, dynamic_events, last_updated "
                "FROM User_Profiles WHERE group_id = ? ORDER BY last_updated DESC",
                (group_id,),
            )
            profiles: list[dict[str, Any]] = []
            for row in cursor.fetchall():
                profiles.append(
                    {
                        "nickname": str(row[0]),
                        "fixed_data": json.loads(row[1]),
                        "dynamic_events": json.loads(row[2]),
                        "last_updated": str(row[3] or ""),
                    }
                )
            return profiles

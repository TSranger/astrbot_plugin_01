"""Database module for plugin memory summaries and user profiles."""

import json
import os
import sqlite3
from typing import Any

from loguru import logger


class MemoryDBManager:
    """Manage SQLite persistence for summaries and user profiles."""

    def __init__(self, db_path: str = "data/agentic_memory.db"):
        """Initialize the SQLite database manager.

        Args:
            db_path: SQLite database file path.
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Create one SQLite connection for the current operation.

        Returns:
            SQLite connection with cross-thread access enabled.
        """
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self) -> None:
        """Initialize tables and apply lightweight schema migrations."""
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
                    nickname TEXT NOT NULL,
                    fixed_data TEXT DEFAULT '{}',
                    dynamic_events TEXT DEFAULT '[]',
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(group_id, nickname)
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
        """Insert one summary row.

        Args:
            group_id: Group identifier.
            summary_type: Summary layer such as ``paragraph`` or ``daily``.
            content: Summary text.
            is_significant: Whether the summary is marked significant.
            memory_key: Logical time bucket key for the summary.
            period_start: Covered period start time.
            period_end: Covered period end time.
            rolled_up_at: Rollup timestamp when this row has been aggregated upward.
            cleanup_after: Earliest time when this row may be deleted.
            source_count: Number of source rows used for this summary.

        Returns:
            Inserted row id.
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
        """Fetch summaries for one group and one layer.

        Args:
            group_id: Group identifier.
            summary_type: Summary layer.
            before_time: Optional upper bound for ``created_at``.
            memory_key: Optional exact ``memory_key`` filter.
            limit: Optional row limit.
            order_desc: Whether to sort newest first.
            only_unrolled: Whether to restrict rows not yet rolled upward.

        Returns:
            Summary rows.
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
        """Fetch rows eligible for upward rollup by covered period.

        Args:
            group_id: Group identifier.
            summary_type: Source summary layer.
            period_start: Optional lower bound for the covered period.
            period_end: Optional upper bound for the covered period.
            exclude_memory_keys: Optional memory keys to exclude.

        Returns:
            Candidate rows ordered by period and creation time.
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
        """Fetch the newest summary row for one memory key.

        Args:
            group_id: Group identifier.
            summary_type: Summary layer.
            memory_key: Logical memory key.

        Returns:
            Summary row or ``None``.
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
        """Update one existing summary row.

        Args:
            summary_id: Row id to update.
            content: New summary text.
            is_significant: Updated significance flag.
            period_start: Updated covered period start.
            period_end: Updated covered period end.
            source_count: Updated source count.
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
        """Mark source summaries as already aggregated upward.

        Args:
            summary_ids: Source row ids.
            rolled_up_at: Rollup timestamp.
            cleanup_after: First safe deletion timestamp.
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
        """Delete summaries by layer and optional creation cutoff.

        Args:
            group_id: Group identifier.
            summary_type: Summary layer.
            before_time: Optional upper bound for ``created_at``.
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
        """Delete rows that have already passed their delayed cleanup time.

        Args:
            group_id: Group identifier.
            summary_type: Summary layer.
            now_text: Current timestamp text.

        Returns:
            Deleted row count.
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
        """List all groups that currently have stored summaries.

        Returns:
            Distinct group id list.
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
        """Search summaries by lightweight keyword matching.

        Args:
            group_id: Group identifier.
            summary_types: Candidate summary layers.
            keywords: Keywords used for ``LIKE`` matching.
            limit: Maximum returned rows.

        Returns:
            Matching summary rows.
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
        """Fetch summaries whose covered period overlaps a time window.

        Args:
            group_id: Group identifier.
            anchor_start: Window start timestamp.
            anchor_end: Window end timestamp.
            limit: Maximum returned rows.
            exclude_ids: Optional row ids to exclude.

        Returns:
            Neighbor summary rows near the anchor window.
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
        """Get one user profile with decoded JSON fields.

        Args:
            group_id: Group identifier.
            nickname: User nickname.

        Returns:
            Profile dictionary with fixed and dynamic memory.
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

    def upsert_user_profile(
        self, group_id: str, nickname: str, fixed_data: dict, dynamic_events: list
    ) -> None:
        """Insert or update one user profile.

        Args:
            group_id: Group identifier.
            nickname: User nickname.
            fixed_data: Stable profile fields.
            dynamic_events: Recent dynamic events.
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO User_Profiles (group_id, nickname, fixed_data, dynamic_events, last_updated)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(group_id, nickname) DO UPDATE SET
                    fixed_data = excluded.fixed_data,
                    dynamic_events = excluded.dynamic_events,
                    last_updated = CURRENT_TIMESTAMP
                """,
                (
                    group_id,
                    nickname,
                    json.dumps(fixed_data, ensure_ascii=False),
                    json.dumps(dynamic_events, ensure_ascii=False),
                ),
            )
            conn.commit()

    def get_proactive_task_state(self, group_id: str, task_id: str) -> dict[str, str]:
        """Get the persisted last run date for a proactive task.

        Args:
            group_id: Group identifier.
            task_id: Proactive task identifier.

        Returns:
            Dictionary with ``last_run_date`` and ``last_send_time`` keys.
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
        """Insert or update proactive task run state after a successful send.

        Args:
            group_id: Group identifier.
            task_id: Proactive task identifier.
            last_run_date: Date string (``YYYY-MM-DD``) of last successful run.
            last_send_time: ISO timestamp of the last successful send.
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

    def get_memory_stats(self, group_id: str) -> dict[str, Any]:
        """Get structured diagnostic stats for a group's memory layers.

        Args:
            group_id: Group identifier.

        Returns:
            Dictionary with per-layer counts, time ranges, and rollup ratios.
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
        """Get all user profiles for a group.

        Args:
            group_id: Group identifier.

        Returns:
            List of profile dictionaries with nickname, fixed_data, and dynamic_events.
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

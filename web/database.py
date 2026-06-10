"""
数据库模块 — SQLite 初始化与 CRUD 封装

所有数据库操作集中在这里，路由通过调用这些函数访问数据。
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List

DB_DIR = Path(__file__).parent / "data"
DB_PATH = DB_DIR / "qa.db"


def get_db() -> sqlite3.Connection:
    """获取数据库连接（线程级，每次调用新建连接）"""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """初始化数据库表结构（幂等）"""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            created_at    TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS models (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL REFERENCES users(id),
            name        TEXT    NOT NULL,
            type        TEXT    NOT NULL CHECK(type IN ('retrieval', 'generative')),
            file_path   TEXT,
            description TEXT    DEFAULT '',
            public      INTEGER DEFAULT 0,
            status      TEXT    DEFAULT 'active' CHECK(status IN ('active', 'disabled')),
            created_at  TEXT    DEFAULT (datetime('now')),
            updated_at  TEXT    DEFAULT (datetime('now')),
            downloads   INTEGER DEFAULT 0,
            avg_score   REAL    DEFAULT 0.0
        );

        CREATE TABLE IF NOT EXISTS ratings (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL REFERENCES users(id),
            model_id      INTEGER NOT NULL REFERENCES models(id),
            question      TEXT    NOT NULL,
            answer        TEXT    NOT NULL,
            score         INTEGER NOT NULL CHECK(score BETWEEN 1 AND 5),
            session_id    TEXT,
            created_at    TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS usage_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id    INTEGER NOT NULL REFERENCES models(id),
            caller_id   INTEGER REFERENCES users(id),
            api_path    TEXT,
            latency_ms  INTEGER,
            created_at  TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS official_scores (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id     INTEGER NOT NULL REFERENCES models(id),
            test_set     TEXT    NOT NULL,
            metric       TEXT    NOT NULL,
            score        REAL    NOT NULL,
            created_at   TEXT    DEFAULT (datetime('now'))
        );

        -- 索引
        CREATE INDEX IF NOT EXISTS idx_models_user ON models(user_id);
        CREATE INDEX IF NOT EXISTS idx_models_type ON models(type);
        CREATE INDEX IF NOT EXISTS idx_models_public ON models(public);
        CREATE INDEX IF NOT EXISTS idx_ratings_model ON ratings(model_id);
        CREATE INDEX IF NOT EXISTS idx_ratings_user ON ratings(user_id);
        CREATE INDEX IF NOT EXISTS idx_usage_model ON usage_logs(model_id);
        CREATE INDEX IF NOT EXISTS idx_official_model ON official_scores(model_id);
    """)
    conn.commit()
    conn.close()


# ════════════════════════════════════════════════════════════════════
# 用户 CRUD
# ════════════════════════════════════════════════════════════════════

def create_user(username: str, password_hash: str) -> dict:
    conn = get_db()
    try:
        cur = conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, password_hash),
        )
        conn.commit()
        return {"id": cur.lastrowid, "username": username}
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def get_user_by_username(username: str) -> Optional[dict]:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[dict]:
    conn = get_db()
    row = conn.execute(
        "SELECT id, username, created_at FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# ════════════════════════════════════════════════════════════════════
# 模型 CRUD
# ════════════════════════════════════════════════════════════════════

def create_model(user_id: int, name: str, model_type: str,
                 description: str = "", public: bool = False) -> dict:
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO models (user_id, name, type, description, public) "
        "VALUES (?, ?, ?, ?, ?)",
        (user_id, name, model_type, description, 1 if public else 0),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM models WHERE id = ?", (cur.lastrowid,)).fetchone()
    conn.close()
    return dict(row)


def get_user_models(user_id: int) -> List[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM models WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_public_models(model_type: str = None) -> List[dict]:
    conn = get_db()
    if model_type:
        rows = conn.execute(
            "SELECT m.*, u.username FROM models m "
            "JOIN users u ON m.user_id = u.id "
            "WHERE m.public = 1 AND m.type = ? ORDER BY m.avg_score DESC",
            (model_type,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT m.*, u.username FROM models m "
            "JOIN users u ON m.user_id = u.id "
            "WHERE m.public = 1 ORDER BY m.avg_score DESC",
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_model_by_id(model_id: int) -> Optional[dict]:
    conn = get_db()
    row = conn.execute(
        "SELECT m.*, u.username FROM models m "
        "JOIN users u ON m.user_id = u.id "
        "WHERE m.id = ?", (model_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def update_model_file(model_id: int, file_path: str):
    conn = get_db()
    conn.execute(
        "UPDATE models SET file_path = ?, updated_at = datetime('now') WHERE id = ?",
        (file_path, model_id),
    )
    conn.commit()
    conn.close()


def delete_model(model_id: int) -> bool:
    conn = get_db()
    # 先删除关联数据，避免外键约束冲突
    conn.execute("DELETE FROM ratings WHERE model_id = ?", (model_id,))
    conn.execute("DELETE FROM usage_logs WHERE model_id = ?", (model_id,))
    conn.execute("DELETE FROM official_scores WHERE model_id = ?", (model_id,))
    cur = conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


def update_model(model_id: int, name: str = None, description: str = None, public: bool = None) -> bool:
    """更新模型属性"""
    conn = get_db()
    updates = []
    params = []
    if name is not None:
        updates.append('name = ?')
        params.append(name)
    if description is not None:
        updates.append('description = ?')
        params.append(description)
    if public is not None:
        updates.append('public = ?')
        params.append(1 if public else 0)
    if not updates:
        conn.close()
        return False
    updates.append("updated_at = datetime('now')")
    params.append(model_id)
    cur = conn.execute(
        f'UPDATE models SET {", ".join(updates)} WHERE id = ?',
        params,
    )
    conn.commit()
    ok = cur.rowcount > 0
    conn.close()
    return ok


def increment_downloads(model_id: int):
    conn = get_db()
    conn.execute(
        "UPDATE models SET downloads = downloads + 1 WHERE id = ?",
        (model_id,),
    )
    conn.commit()
    conn.close()


# ════════════════════════════════════════════════════════════════════
# 评分
# ════════════════════════════════════════════════════════════════════

def create_rating(user_id: int, model_id: int, question: str,
                  answer: str, score: int, session_id: str = None) -> dict:
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO ratings (user_id, model_id, question, answer, score, session_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, model_id, question, answer, score, session_id),
    )
    conn.commit()

    # 更新模型平均分
    row = conn.execute(
        "SELECT AVG(score) as avg_s FROM ratings WHERE model_id = ?",
        (model_id,),
    ).fetchone()
    if row and row["avg_s"]:
        conn.execute(
            "UPDATE models SET avg_score = ? WHERE id = ?",
            (round(row["avg_s"], 4), model_id),
        )
    conn.commit()
    conn.close()
    return {"id": cur.lastrowid, "score": score}


def get_model_ratings(model_id: int) -> List[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT r.*, u.username FROM ratings r "
        "JOIN users u ON r.user_id = u.id "
        "WHERE r.model_id = ? ORDER BY r.created_at DESC",
        (model_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ════════════════════════════════════════════════════════════════════
# 排行榜
# ════════════════════════════════════════════════════════════════════

def get_ranking(model_type: str = None, sort_by: str = "avg_score",
                period: str = "all", limit: int = 50) -> List[dict]:
    """排行榜查询"""
    conn = get_db()

    date_filter = ""
    if period == "7d":
        date_filter = "AND m.created_at >= datetime('now', '-7 days')"
    elif period == "30d":
        date_filter = "AND m.created_at >= datetime('now', '-30 days')"

    type_filter = ""
    if model_type and model_type != "all":
        type_filter = "AND m.type = ?"

    # 排序
    order_map = {
        "avg_score": "m.avg_score DESC",
        "downloads": "m.downloads DESC",
        "official": "os.score DESC",
    }
    order_clause = order_map.get(sort_by, "m.avg_score DESC")

    query = f"""
        SELECT m.*, u.username,
               (SELECT AVG(r.score) FROM ratings r WHERE r.model_id = m.id) as rating_avg,
               (SELECT COUNT(*) FROM ratings r WHERE r.model_id = m.id) as rating_count,
               (SELECT COUNT(*) FROM usage_logs ul WHERE ul.model_id = m.id) as call_count
        FROM models m
        JOIN users u ON m.user_id = u.id
        LEFT JOIN official_scores os ON os.model_id = m.id
        WHERE m.public = 1 AND m.status = 'active' {date_filter} {type_filter}
        ORDER BY {order_clause}
        LIMIT ?
    """

    params = []
    if model_type and model_type != "all":
        params.append(model_type)
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ════════════════════════════════════════════════════════════════════
# 使用统计
# ════════════════════════════════════════════════════════════════════

def log_usage(model_id: int, caller_id: int = None, api_path: str = "",
              latency_ms: int = 0):
    conn = get_db()
    conn.execute(
        "INSERT INTO usage_logs (model_id, caller_id, api_path, latency_ms) "
        "VALUES (?, ?, ?, ?)",
        (model_id, caller_id, api_path, latency_ms),
    )
    conn.commit()
    conn.close()

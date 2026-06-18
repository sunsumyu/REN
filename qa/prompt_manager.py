import sqlite3
import os
import hashlib
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "prompts.db")


def _md5(text: str) -> str:
    """计算字符串的 MD5 指纹，用于快速检测提示词是否变化。"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class PromptManager:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """
        Initializes the SQLite database and creates the prompt_versions table if it doesn't exist.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prompt_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt_name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    content_md5 TEXT,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active INTEGER DEFAULT 0
                )
            """)
            # 兼容旧库：若 content_md5 列不存在则补加
            try:
                cursor.execute("ALTER TABLE prompt_versions ADD COLUMN content_md5 TEXT")
                conn.commit()
            except Exception:
                pass  # 列已存在，忽略
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prompt_name_active ON prompt_versions(prompt_name, is_active)")
            conn.commit()
            conn.close()
            logger.info(f"SQLite Prompt Database initialized successfully at: {self.db_path}")
        except Exception as e:
            logger.critical(f"Failed to initialize prompt database: {e}")
            raise e

    def register_prompt(self, name: str, default_content: str, description: str = "Bootstrap Default") -> str:
        """
        【代码优先策略】对比代码中的 _BOOTSTRAP_ 提示词与数据库当前激活版本的 MD5。
        - 若内容相同：直接使用数据库版本（无操作）。
        - 若内容不同（或数据库中无记录）：将代码版本写入数据库为新版本并激活。
        
        这确保：代码中修改提示词后，下次启动会自动以新版本覆盖，无需手动操作。
        """
        code_md5 = _md5(default_content)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            # 查询当前激活版本的内容和 MD5
            cursor.execute(
                "SELECT content, content_md5 FROM prompt_versions WHERE prompt_name = ? AND is_active = 1",
                (name,)
            )
            row = cursor.fetchone()

            if row is None:
                # 数据库中没有任何激活版本，首次注册
                cursor.execute("SELECT COUNT(*) FROM prompt_versions WHERE prompt_name = ?", (name,))
                count = cursor.fetchone()[0]

                if count == 0:
                    # 全新提示词，写入版本 1
                    logger.info(f"[PromptManager] 首次注册提示词 '{name}'，写入版本 1 (MD5={code_md5[:8]})")
                    cursor.execute("""
                        INSERT INTO prompt_versions (prompt_name, version, content, content_md5, description, is_active)
                        VALUES (?, 1, ?, ?, ?, 1)
                    """, (name, default_content, code_md5, description))
                else:
                    # 有历史版本但无激活版，激活最新版
                    logger.warning(f"[PromptManager] '{name}' 无激活版本，自动激活最新历史版本")
                    cursor.execute("""
                        UPDATE prompt_versions 
                        SET is_active = 1 
                        WHERE id = (SELECT id FROM prompt_versions WHERE prompt_name = ? ORDER BY version DESC LIMIT 1)
                    """, (name,))

                conn.commit()
                cursor.execute("SELECT content FROM prompt_versions WHERE prompt_name = ? AND is_active = 1", (name,))
                row = cursor.fetchone()
                return row[0] if row else default_content

            else:
                db_content, db_md5 = row

                # 如果数据库里的 MD5 为空（旧数据），用内容本身计算
                if not db_md5:
                    db_md5 = _md5(db_content)
                    cursor.execute(
                        "UPDATE prompt_versions SET content_md5 = ? WHERE prompt_name = ? AND is_active = 1",
                        (db_md5, name)
                    )
                    conn.commit()

                if code_md5 == db_md5:
                    # 内容一致，直接返回数据库版本（无需任何操作）
                    logger.debug(f"[PromptManager] '{name}' 提示词无变化，使用数据库版本")
                    return db_content
                else:
                    # 代码中的提示词已更新，自动写入新版本并激活
                    cursor.execute("SELECT MAX(version) FROM prompt_versions WHERE prompt_name = ?", (name,))
                    latest_version = cursor.fetchone()[0] or 0
                    new_version = latest_version + 1

                    cursor.execute("UPDATE prompt_versions SET is_active = 0 WHERE prompt_name = ?", (name,))
                    cursor.execute("""
                        INSERT INTO prompt_versions (prompt_name, version, content, content_md5, description, is_active)
                        VALUES (?, ?, ?, ?, ?, 1)
                    """, (name, new_version, default_content, code_md5, f"代码自动更新 (MD5变化: {db_md5[:8]}→{code_md5[:8]})"))
                    conn.commit()
                    logger.info(
                        f"[PromptManager] 检测到 '{name}' 提示词变化，已自动写入版本 {new_version} 并激活 "
                        f"(MD5: {db_md5[:8]}→{code_md5[:8]})"
                    )
                    return default_content

        except Exception as e:
            logger.error(f"Error registering prompt '{name}': {e}")
            return default_content
        finally:
            conn.close()

    def get_prompt(self, name: str) -> str:
        """
        Fetches the active prompt content from the database.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT content FROM prompt_versions WHERE prompt_name = ? AND is_active = 1", (name,))
            row = cursor.fetchone()
            if row is None:
                # If not registered, load the hardcoded fallback from prompts.py (bootstrapping happens dynamically)
                import prompts
                default_content = getattr(prompts, f"_BOOTSTRAP_{name}", None)
                if default_content is None:
                    default_content = getattr(prompts, name, None)

                if default_content:
                    logger.info(f"Dynamic fetch triggered bootstrap for '{name}'")
                    return self.register_prompt(name, default_content)
                raise ValueError(f"Prompt '{name}' not found in database or prompts module defaults")
            return row[0]
        finally:
            conn.close()

    def save_new_version(self, name: str, new_content: str, description: str = "") -> int:
        """
        Saves a new prompt version and sets it as the active version, deactivating the old one.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT MAX(version) FROM prompt_versions WHERE prompt_name = ?", (name,))
            row = cursor.fetchone()
            latest_version = row[0] if row[0] is not None else 0
            new_version = latest_version + 1

            cursor.execute("UPDATE prompt_versions SET is_active = 0 WHERE prompt_name = ?", (name,))
            cursor.execute("""
                INSERT INTO prompt_versions (prompt_name, version, content, content_md5, description, is_active)
                VALUES (?, ?, ?, ?, ?, 1)
            """, (name, new_version, new_content, _md5(new_content), description))

            conn.commit()
            logger.info(f"Saved new version {new_version} for prompt '{name}' (Active)")
            return new_version
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to save new version for '{name}': {e}")
            raise e
        finally:
            conn.close()

    def rollback_to_version(self, name: str, version: int) -> bool:
        """
        Rolls back the active prompt to a specific historical version.
        Returns True if successful, False otherwise.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT id FROM prompt_versions WHERE prompt_name = ? AND version = ?", (name, version))
            row = cursor.fetchone()
            if row is None:
                logger.error(f"Cannot rollback: Version {version} of '{name}' does not exist.")
                return False

            cursor.execute("UPDATE prompt_versions SET is_active = 0 WHERE prompt_name = ?", (name,))
            cursor.execute("UPDATE prompt_versions SET is_active = 1 WHERE prompt_name = ? AND version = ?", (name, version))
            conn.commit()
            logger.info(f"Prompt '{name}' rolled back to Version {version} successfully.")
            return True
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to rollback '{name}' to version {version}: {e}")
            return False
        finally:
            conn.close()

    def list_versions(self, name: str) -> List[Dict[str, Any]]:
        """
        Lists all historical versions of a prompt.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT version, content, content_md5, description, created_at, is_active 
                FROM prompt_versions 
                WHERE prompt_name = ? 
                ORDER BY version DESC
            """, (name,))
            rows = cursor.fetchall()

            history = []
            for r in rows:
                history.append({
                    "version": r[0],
                    "content": r[1],
                    "content_md5": r[2],
                    "description": r[3],
                    "created_at": r[4],
                    "is_active": bool(r[5])
                })
            return history
        finally:
            conn.close()

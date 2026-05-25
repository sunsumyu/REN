import sqlite3
import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "prompts.db")

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
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active INTEGER DEFAULT 0
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prompt_name_active ON prompt_versions(prompt_name, is_active)")
            conn.commit()
            conn.close()
            logger.info(f"SQLite Prompt Database initialized successfully at: {self.db_path}")
        except Exception as e:
            logger.critical(f"Failed to initialize prompt database: {e}")
            raise e

    def register_prompt(self, name: str, default_content: str, description: str = "Bootstrap Default") -> str:
        """
        Registers a default prompt template if no version exists in the database.
        Returns the active content.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            # Check if an active version exists
            cursor.execute("SELECT content FROM prompt_versions WHERE prompt_name = ? AND is_active = 1", (name,))
            row = cursor.fetchone()
            
            if row is None:
                # Check if ANY version exists for this prompt
                cursor.execute("SELECT COUNT(*) FROM prompt_versions WHERE prompt_name = ?", (name,))
                count = cursor.fetchone()[0]
                
                if count == 0:
                    logger.info(f"Self-bootstrapping default prompt for '{name}' (Version 1)")
                    cursor.execute("""
                        INSERT INTO prompt_versions (prompt_name, version, content, description, is_active)
                        VALUES (?, 1, ?, ?, 1)
                    """, (name, default_content, description))
                    conn.commit()
                else:
                    logger.warning(f"No active version found for '{name}' but history exists. Activating latest version.")
                    cursor.execute("""
                        UPDATE prompt_versions 
                        SET is_active = 1 
                        WHERE id = (SELECT id FROM prompt_versions WHERE prompt_name = ? ORDER BY version DESC LIMIT 1)
                    """, (name,))
                    conn.commit()
                
                # Fetch again
                cursor.execute("SELECT content FROM prompt_versions WHERE prompt_name = ? AND is_active = 1", (name,))
                row = cursor.fetchone()
                
            return row[0] if row else default_content
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
                    # In case the bootstrap naming matches exact template name in a fallback attempt
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
            # Get latest version number
            cursor.execute("SELECT MAX(version) FROM prompt_versions WHERE prompt_name = ?", (name,))
            row = cursor.fetchone()
            latest_version = row[0] if row[0] is not None else 0
            new_version = latest_version + 1
            
            # Deactivate current active version
            cursor.execute("UPDATE prompt_versions SET is_active = 0 WHERE prompt_name = ?", (name,))
            
            # Insert the new active version
            cursor.execute("""
                INSERT INTO prompt_versions (prompt_name, version, content, description, is_active)
                VALUES (?, ?, ?, ?, 1)
            """, (name, new_version, new_content, description))
            
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
            # Check if this specific version exists
            cursor.execute("SELECT id FROM prompt_versions WHERE prompt_name = ? AND version = ?", (name, version))
            row = cursor.fetchone()
            if row is None:
                logger.error(f"Cannot rollback: Version {version} of '{name}' does not exist.")
                return False
            
            # Deactivate all versions of this prompt
            cursor.execute("UPDATE prompt_versions SET is_active = 0 WHERE prompt_name = ?", (name,))
            
            # Activate the specific version
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
                SELECT version, content, description, created_at, is_active 
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
                    "description": r[2],
                    "created_at": r[3],
                    "is_active": bool(r[4])
                })
            return history
        finally:
            conn.close()

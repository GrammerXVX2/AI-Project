import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from ai_server.app.config import CHATS_DIR


class HistoryManager:
    def __init__(self):
        self.chats_dir = CHATS_DIR

    def _get_file_path(self, session_id: str):
        return self.chats_dir / f"{session_id}.json"

    def load_history(self, session_id: str) -> List[dict]:
        """Loads message history from JSON file"""
        file_path = self._get_file_path(session_id)
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("messages", [])
            except Exception as e:
                print(f"[HISTORY] Error loading {session_id}: {e}")
                return []
        return []

    def save_history(self, session_id: str, messages: List[dict]):
        """Saves history to JSON"""
        file_path = self._get_file_path(session_id)
        data = {
            "session_id": session_id,
            "last_updated": datetime.now().isoformat(),
            "messages": messages,
        }
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[HISTORY] Error saving {session_id}: {e}")

    def get_all_sessions(self) -> List[Dict]:
        """Returns list of all chat sessions sorted by date"""
        sessions = []
        if self.chats_dir.exists():
            # Sort by modification time (newest first)
            files = sorted(self.chats_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
            for f in files:
                try:
                    with open(f, "r", encoding="utf-8") as file:
                        data = json.load(file)
                        # Use first user message as title or fallback
                        messages = data.get("messages", [])
                        first_msg = next(
                            (m["content"] for m in messages if m["role"] == "user"), "New Chat"
                        )
                        title = first_msg[:30]

                        sessions.append(
                            {
                                "id": data.get("session_id", f.stem),
                                "title": title,
                                "date": data.get("last_updated", ""),
                            }
                        )
                except:
                    continue
        return sessions

    def delete_session(self, session_id: str) -> bool:
        file_path = self._get_file_path(session_id)
        if file_path.exists():
            try:
                os.remove(file_path)
                return True
            except Exception:
                return False
        return False

    # Async wrappers
    async def aload_history(self, session_id: str):
        return await asyncio.to_thread(self.load_history, session_id)

    async def asave_history(self, session_id: str, messages: List[dict]):
        await asyncio.to_thread(self.save_history, session_id, messages)

    async def aget_all_sessions(self):
        return await asyncio.to_thread(self.get_all_sessions)

    async def adelete_session(self, session_id: str):
        return await asyncio.to_thread(self.delete_session, session_id)


# Global instance
history_manager = HistoryManager()

from typing import Optional

from pydantic import BaseModel


class UserRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None
    use_rag: bool = True
    history_limit: int = 5
    max_tokens: int = 512
    temperature: float = 0.7
    repeat_penalty: float = 1.1
    system_prompt: Optional[str] = None


class UpdateMessageRequest(BaseModel):
    content: str

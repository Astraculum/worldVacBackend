import asyncio
from enum import Enum
from typing import Optional

from AgentMatrix.src.llm import LLMClient
from logger import get_logger


class SceneStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class SceneTask:
    def __init__(self, user_id: str, world_id: str, commit_id: str):
        self.user_id = user_id
        self.world_id = world_id
        self.commit_id = commit_id
        self.status = SceneStatus.PENDING
        self.error: Optional[str] = None
        self._task: Optional[asyncio.Task] = None

    def set_task(self, task: asyncio.Task):
        self._task = task
        self.status = SceneStatus.IN_PROGRESS

    def set_completed(self):
        self.status = SceneStatus.COMPLETED
        self._task = None

    def set_failed(self, error: str):
        self.status = SceneStatus.FAILED
        self.error = error
        self._task = None

    def is_completed(self) -> bool:
        return self.status == SceneStatus.COMPLETED

    def is_failed(self) -> bool:
        return self.status == SceneStatus.FAILED

    def is_in_progress(self) -> bool:
        return self.status == SceneStatus.IN_PROGRESS


class SceneTaskManager:
    def __init__(self, fast_chat_llm_client: Optional[LLMClient] = None):
        self._tasks: dict[tuple[str, str, str], SceneTask] = (
            {}
        )  # (user_id, world_id, commit_id) -> task
        self._lock = asyncio.Lock()
        self.fast_chat_llm_client = fast_chat_llm_client

    async def create_task(
        self, user_id: str, world_id: str, commit_id: str
    ) -> SceneTask:
        async with self._lock:
            task = SceneTask(user_id, world_id, commit_id)
            self._tasks[(user_id, world_id, commit_id)] = task
            return task

    async def get_task(
        self, user_id: str, world_id: str, commit_id: str
    ) -> Optional[SceneTask]:
        async with self._lock:
            return self._tasks.get((user_id, world_id, commit_id))

    async def remove_task(self, user_id: str, world_id: str, commit_id: str):
        async with self._lock:
            if (user_id, world_id, commit_id) in self._tasks:
                del self._tasks[(user_id, world_id, commit_id)]


# Global task manager instance
scene_task_manager = SceneTaskManager()

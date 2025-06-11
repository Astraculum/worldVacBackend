import asyncio
from enum import Enum
from typing import Optional
from uuid import UUID

from logger import get_logger


class WorldCreationStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class WorldCreationTask:
    def __init__(self, user_id: str, world_id: str, commit_id: str):
        self.user_id = user_id
        self.world_id = world_id
        self.commit_id = commit_id
        self.status = WorldCreationStatus.PENDING
        self.error: Optional[str] = None
        self._task: Optional[asyncio.Task] = None

    def set_task(self, task: asyncio.Task):
        self._task = task
        self.status = WorldCreationStatus.IN_PROGRESS

    def set_completed(self):
        self.status = WorldCreationStatus.COMPLETED
        self._task = None

    def set_failed(self, error: str):
        self.status = WorldCreationStatus.FAILED
        self.error = error
        self._task = None

    def is_completed(self) -> bool:
        return self.status == WorldCreationStatus.COMPLETED

    def is_failed(self) -> bool:
        return self.status == WorldCreationStatus.FAILED

    def is_in_progress(self) -> bool:
        return self.status == WorldCreationStatus.IN_PROGRESS


class WorldTaskManager:
    def __init__(self):
        self._tasks: dict[tuple[str, str, str], WorldCreationTask] = {}  # (user_id, world_id, commit_id) -> task
        self._lock = asyncio.Lock()

    async def create_task(self, user_id: str, world_id: str, commit_id: str) -> WorldCreationTask:
        async with self._lock:
            task = WorldCreationTask(user_id, world_id, commit_id)
            self._tasks[(user_id, world_id, commit_id)] = task
            return task

    async def get_task(self, user_id: str, world_id: str, commit_id: str) -> Optional[WorldCreationTask]:
        async with self._lock:
            return self._tasks.get((user_id, world_id, commit_id))

    async def remove_task(self, user_id: str, world_id: str, commit_id: str):
        async with self._lock:
            if (user_id, world_id, commit_id) in self._tasks:
                del self._tasks[(user_id, world_id, commit_id)]


# Global task manager instance
world_task_manager = WorldTaskManager() 

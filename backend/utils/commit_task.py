import asyncio
from enum import Enum
from typing import Optional

from logger import get_logger


class CommitStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class CommitTask:
    def __init__(self, user_id: str, world_id: str, commit_id: str):
        self.user_id = user_id
        self.world_id = world_id
        self.commit_id = commit_id
        self.status = CommitStatus.PENDING
        self.error: Optional[str] = None
        self._task: Optional[asyncio.Task] = None

    def set_task(self, task: asyncio.Task):
        self._task = task
        self.status = CommitStatus.IN_PROGRESS

    def set_completed(self):
        self.status = CommitStatus.COMPLETED
        self._task = None

    def set_failed(self, error: str):
        self.status = CommitStatus.FAILED
        self.error = error
        self._task = None

    def is_completed(self) -> bool:
        return self.status == CommitStatus.COMPLETED

    def is_failed(self) -> bool:
        return self.status == CommitStatus.FAILED

    def is_in_progress(self) -> bool:
        return self.status == CommitStatus.IN_PROGRESS


class CommitTaskManager:
    def __init__(self):
        self._tasks: dict[tuple[str, str, str], CommitTask] = {}  # (user_id, world_id, commit_id) -> task
        self._lock = asyncio.Lock()

    async def create_task(self, user_id: str, world_id: str, commit_id: str) -> CommitTask:
        async with self._lock:
            task = CommitTask(user_id, world_id, commit_id)
            self._tasks[(user_id, world_id, commit_id)] = task
            return task

    async def get_task(self, user_id: str, world_id: str, commit_id: str) -> Optional[CommitTask]:
        async with self._lock:
            return self._tasks.get((user_id, world_id, commit_id))

    async def remove_task(self, user_id: str, world_id: str, commit_id: str):
        async with self._lock:
            if (user_id, world_id, commit_id) in self._tasks:
                del self._tasks[(user_id, world_id, commit_id)]


# Global task manager instance
commit_task_manager = CommitTaskManager() 

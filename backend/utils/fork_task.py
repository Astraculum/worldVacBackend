import asyncio
from enum import Enum
from typing import Optional


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ForkTask:
    def __init__(self, user_id: str, world_id: str, commit_id: str, new_world_id: str):
        self.user_id = user_id
        self.world_id = world_id
        self.commit_id = commit_id
        self.new_world_id = new_world_id
        self.status = TaskStatus.PENDING
        self.error: Optional[str] = None
        self._task: Optional[asyncio.Task] = None

    def set_task(self, task: asyncio.Task):
        self._task = task
        self.status = TaskStatus.IN_PROGRESS

    def set_completed(self):
        self.status = TaskStatus.COMPLETED
        self._task = None

    def set_failed(self, error: str):
        self.status = TaskStatus.FAILED
        self.error = error
        self._task = None

    def is_completed(self) -> bool:
        return self.status == TaskStatus.COMPLETED

    def is_failed(self) -> bool:
        return self.status == TaskStatus.FAILED

    def is_in_progress(self) -> bool:
        return self.status == TaskStatus.IN_PROGRESS

class ForkTaskManager:
    def __init__(self):
        self._tasks: dict[tuple[str, str, str], ForkTask] = {}
        self._lock = asyncio.Lock()

    async def create_task(
        self, user_id: str, world_id: str, commit_id: str, new_world_id: str
    ) -> ForkTask:
        task = ForkTask(user_id, world_id, commit_id, new_world_id)
        async with self._lock:
            self._tasks[(user_id, world_id, commit_id)] = task
        return task

    async def get_task(
        self, user_id: str, world_id: str, commit_id: str
    ) -> Optional[ForkTask]:
        async with self._lock:
            return self._tasks.get((user_id, world_id, commit_id))


fork_task_manager = ForkTaskManager()

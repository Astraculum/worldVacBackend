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

    async def create_or_get_task(
        self, user_id: str, world_id: str, commit_id: str
    ) -> SceneTask:
        """创建新任务或获取现有任务。如果任务已存在且未完成，返回现有任务。"""
        async with self._lock:
            key = (user_id, world_id, commit_id)
            existing_task = self._tasks.get(key)
            
            if existing_task:
                if existing_task.is_in_progress():
                    return existing_task
                elif existing_task.is_failed():
                    # 如果任务失败，创建新任务
                    task = SceneTask(user_id, world_id, commit_id)
                    self._tasks[key] = task
                    return task
                elif existing_task.is_completed():
                    # 如果任务已完成，返回现有任务
                    return existing_task
            
            # 如果没有现有任务，创建新任务
            task = SceneTask(user_id, world_id, commit_id)
            self._tasks[key] = task
            return task

    async def get_task(
        self, user_id: str, world_id: str, commit_id: str
    ) -> Optional[SceneTask]:
        """获取任务，如果不存在返回None"""
        async with self._lock:
            return self._tasks.get((user_id, world_id, commit_id))

    async def remove_task(self, user_id: str, world_id: str, commit_id: str):
        """移除任务，如果任务正在进行中会被取消"""
        async with self._lock:
            key = (user_id, world_id, commit_id)
            if key in self._tasks:
                task = self._tasks[key]
                if task.is_in_progress() and task._task:
                    task._task.cancel()
                del self._tasks[key]

    async def cleanup_completed_tasks(self):
        """清理已完成或失败的任务"""
        async with self._lock:
            keys_to_remove = []
            for key, task in self._tasks.items():
                if task.is_completed() or task.is_failed():
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._tasks[key]


# Global task manager instance
scene_task_manager = SceneTaskManager()

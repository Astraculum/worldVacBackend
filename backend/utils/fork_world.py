import asyncio
import traceback
from typing import Literal, Optional

from AgentMatrix.model import CommitIdentifier, WorldIdentifier
from AgentMatrix.src.graph import ForkRelationEntity, Graph
from AgentMatrix.src.llm import LLMClient
from AgentMatrix.src.memory import SentenceEmbedding
from logger import get_logger as get_logger_backend

from .commit_tree import CommitTree
from .fork_task import fork_task_manager


async def background_fork_world(
    world_dict: dict[WorldIdentifier, Graph],
    world_lock: asyncio.Lock,
    commit_tree_lock: asyncio.Lock,
    commit_trees_dict: dict[CommitIdentifier, CommitTree],
    source_graph: Graph,
    user_id: str,
    world_id: str,
    commit_id: str,
    new_user_id: str,
    new_world_id: str,
    llm_client: LLMClient,
    embeddings: Optional[SentenceEmbedding] = None,
    fork_seed_prompt: Optional[str] = None,
    mode: Literal["full", "remodify"] = "remodify",
):
    try:
        # Fork the world using Graph's fork_world method
        new_graph = await Graph.fork_world(
            source_graph=source_graph,
            ask_for_forks_user_id=user_id,
            llm_client=llm_client,
            embeddings=embeddings,
            fork_seed_prompt=fork_seed_prompt,
            mode=mode,
        )
        new_commit_id = await new_graph.generate_world_status_uuid()

        # add `fork from` to new graph
        async with new_graph.fork_from_lock:
            new_graph.fork_from.append(
                ForkRelationEntity(
                    user_id=user_id,
                    world_id=world_id,
                    commit_id=commit_id,
                )
            )

        # add `fork to` to source graph
        async with source_graph.fork_to_lock:
            source_graph.fork_to.append(
                ForkRelationEntity(
                    user_id=new_user_id,
                    world_id=new_world_id,
                    commit_id=new_commit_id,
                )
            )
            # Save the new graph
            from main import save_commit_tree, save_graph

            # Save the updated source graph
            await save_graph(user_id, world_id, commit_id, source_graph)

            await save_graph(new_user_id, new_world_id, new_commit_id, new_graph)

        # Get the source world's commit tree
        source_commit_identifier = CommitIdentifier(user_id=user_id, world_id=world_id)
        source_commit_tree = commit_trees_dict.get(source_commit_identifier, None)
        if mode == "full" and source_commit_tree is None:
            source_commit_tree = None

        # Create new commit tree
        commit_identifier = CommitIdentifier(user_id=new_user_id, world_id=new_world_id)
        async with commit_tree_lock:
            if commit_identifier not in commit_trees_dict:
                commit_trees_dict[commit_identifier] = CommitTree()
            commit_tree = commit_trees_dict[commit_identifier]

            # If source commit tree exists, copy all its commits
            if source_commit_tree:
                # Copy all commits from source tree
                for node in await source_commit_tree.get_all_commits():
                    await commit_tree.add_commit(
                        user_id=user_id,
                        world_id=world_id,
                        commit_id=node.commit_id,
                        graph=source_graph,  # We'll use the source graph for historical commits
                        parent_id=node.parent_id,
                    )

            # Add the new fork commit
            await commit_tree.add_commit(
                user_id=new_user_id,
                world_id=new_world_id,
                commit_id=new_commit_id,
                graph=new_graph,
                parent_id=commit_id,  # Set parent to the commit we forked from
            )

        await save_commit_tree(user_id, new_world_id, commit_tree)

        # Store the new graph in world_dict

        new_world_identifier = WorldIdentifier(
            user_id=new_user_id, world_id=new_world_id, commit_id=new_commit_id
        )
        async with world_lock:
            world_dict[new_world_identifier] = new_graph

        # Mark task as completed
        task = await fork_task_manager.get_task(user_id, world_id, commit_id)
        if task:
            task.set_completed()

    except Exception as e:
        get_logger_backend().error(f"Fork world failed: {e}")
        get_logger_backend().error(traceback.format_exc())
        task = await fork_task_manager.get_task(user_id, world_id, commit_id)
        if task:
            task.set_failed(str(e))

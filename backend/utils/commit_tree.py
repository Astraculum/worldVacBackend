from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

from AgentMatrix.src.graph import Graph


@dataclass
class CommitNode:
    """Represents a single commit in the commit tree."""
    world_id: str
    user_id: str
    commit_id: str
    event_summary: str
    topic: str
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []

    def add_child(self, child_id: str):
        """Add a child commit ID to this node."""
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)

    def to_dict(self):
        """Convert the node to a dictionary for serialization."""
        return {
            "commit_id": self.commit_id,
            "world_id": self.world_id,
            "user_id": self.user_id,
            "event_summary": self.event_summary,
            "topic": self.topic,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids
        }
        
    def to_json(self) -> dict:
        """Convert the node to a JSON object."""
        return {
            "commit_id": self.commit_id,
            "event_summary": self.event_summary,
            "topic": self.topic,
            "parent_id": self.parent_id if self.parent_id is not None else "root",
            "children_ids": self.children_ids
        }

    @classmethod
    def from_dict(cls, data: dict) -> CommitNode:
        """Create a CommitNode from a dictionary."""
        return cls(
            commit_id=data["commit_id"],
            world_id=data["world_id"],
            user_id=data["user_id"],
            event_summary=data["event_summary"],
            topic=data["topic"],
            parent_id=data.get("parent_id"),
            children_ids=data.get("children_ids", [])
        )


class CommitTree:
    """Manages the tree structure of world commits."""
    
    def __init__(self, ):
        self.nodes: dict[str, CommitNode] = {}
        self.root_id: Optional[str] = None
        self._lock = asyncio.Lock()
        
    def to_json(self) -> dict:
        """Convert the tree to a JSON object."""
        return {
            "root_id": self.root_id,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
        }
    @classmethod
    def from_json(cls, data: dict) -> CommitTree:
        """Create a CommitTree from a JSON object."""
        entity = cls()
        entity.root_id = data["root_id"]
        entity.nodes = {k: CommitNode.from_dict(v) for k, v in data["nodes"].items()}
        return entity
    
    async def add_commit(self, world_id: str, user_id: str, commit_id: str, graph: Graph, parent_id: Optional[str] = None) -> CommitNode:
        """Add a new commit to the tree."""
        async with self._lock:
            # Extract metadata from the graph
            event_summary = graph.commit_metadata.event_summary
            topic = graph.commit_metadata.topic
            
            # Create new node
            node = CommitNode(
                world_id=world_id,
                user_id=user_id,
                commit_id=commit_id,
                event_summary=event_summary,
                topic=topic,
                parent_id=parent_id
            )
            
            # Add to tree
            self.nodes[commit_id] = node
            
            # If this is the first commit, set it as root
            if not self.root_id:
                self.root_id = commit_id
            
            # If it has a parent, update the parent's children list
            if parent_id and parent_id in self.nodes:
                self.nodes[parent_id].add_child(commit_id)
            
            return node
    
    async def get_commit(self, commit_id: str) -> Optional[CommitNode]:
        """Get a commit node by ID."""
        async with self._lock:
            return self.nodes.get(commit_id)
    
    async def get_all_commits(self) -> list[CommitNode]:
        """Get all commits in the tree."""
        async with self._lock:
            return list(self.nodes.values())
    
    async def get_latest_commit(self) -> Optional[CommitNode]:
        """Get the latest commit (a leaf node with no children)."""
        async with self._lock:
            if not self.nodes:
                return None
        
        # Find all leaf nodes (nodes with no children)
        leaf_nodes = [node for node in self.nodes.values() if not node.children_ids]
        
        # If there are no leaf nodes, return None
        if not leaf_nodes:
            return None
        
        # For now, just return the first leaf node found
        return leaf_nodes[0]
    
    def delete_commit(self, commit_id: str) -> bool:
        """Delete a commit and its descendants from the tree."""
        if commit_id not in self.nodes:
            return False
        
        # Get the node to delete
        node = self.nodes[commit_id]
        
        # If this is the root node and has children, we can't delete it
        if commit_id == self.root_id and node.children_ids:
            return False
        
        # Remove this node from its parent's children list
        if node.parent_id and node.parent_id in self.nodes:
            parent = self.nodes[node.parent_id]
            if commit_id in parent.children_ids:
                parent.children_ids.remove(commit_id)
        
        # Recursively delete all children
        children_to_delete = node.children_ids.copy()
        for child_id in children_to_delete:
            self.delete_commit(child_id)
        
        # Delete this node
        del self.nodes[commit_id]
        
        # If this was the root, update the root
        if commit_id == self.root_id:
            self.root_id = None
        
        return True
    
    def get_commit_path(self, commit_id: str) -> list[CommitNode]:
        """Get the path from root to the specified commit."""
        if commit_id not in self.nodes:
            return []
        
        path = []
        current_id = commit_id
        
        # Traverse up the tree until we reach the root
        while current_id:
            node = self.nodes.get(current_id)
            if not node:
                break
                
            path.append(node)
            current_id = node.parent_id
        
        # Reverse to get path from root to commit
        return list(reversed(path))

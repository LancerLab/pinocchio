"""
Knowledge models for Pinocchio.

This module defines the models for tracking knowledge items and their versions.
"""

import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class KnowledgeItem(BaseModel):
    """
    Knowledge item version.

    Represents a specific version of a knowledge item.
    """

    version_id: str
    knowledge_id: str
    content: Dict[str, Any]  # Structured knowledge content
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str = ""  # Knowledge source
    confidence: float = 1.0  # Confidence level
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create_new_version(
        cls,
        knowledge_id: str,
        content: Dict[str, Any],
        source: str = "",
        confidence: float = 1.0,
    ) -> "KnowledgeItem":
        """
        Create a new knowledge item version.

        Args:
            knowledge_id: The ID of the knowledge item
            content: The knowledge content
            source: The source of the knowledge
            confidence: The confidence level (0.0 to 1.0)

        Returns:
            A new KnowledgeItem instance
        """
        # Generate a unique version ID based on content and timestamp
        hash_input = f"{knowledge_id}:{str(content)}:{datetime.utcnow().timestamp()}"
        hash_object = hashlib.sha256(hash_input.encode())
        version_id = hash_object.hexdigest()[:12]

        return cls(
            version_id=version_id,
            knowledge_id=knowledge_id,
            content=content,
            source=source,
            confidence=confidence,
        )

    def extract_fragment(self, path: str) -> Any:
        """
        Extract a specific fragment from the knowledge item content using a path expression.

        Args:
            path: Path to the fragment (e.g., "techniques[0].name")

        Returns:
            The extracted fragment

        Raises:
            ValueError: If the path is invalid or the fragment doesn't exist
        """
        parts = path.split(".")
        current = self.content

        for part in parts:
            # Handle array indexing
            if "[" in part and part.endswith("]"):
                key, idx_str = part.split("[", 1)
                idx = int(idx_str[:-1])  # Remove the closing bracket

                if key not in current:
                    raise ValueError(f"Key '{key}' not found in knowledge item content")

                if not isinstance(current[key], list) or idx >= len(current[key]):
                    raise ValueError(f"Invalid array index '{idx}' for key '{key}'")

                current = current[key][idx]
            else:
                if part not in current:
                    raise ValueError(
                        f"Key '{part}' not found in knowledge item content"
                    )
                current = current[part]

        return current


class KnowledgeMemory(BaseModel):
    """
    Knowledge base version management.

    Maintains a history of knowledge item versions.
    """

    knowledge_items: Dict[str, Dict[str, KnowledgeItem]] = Field(
        default_factory=dict
    )  # knowledge_id -> {version_id -> item}
    current_versions: Dict[str, str] = Field(
        default_factory=dict
    )  # knowledge_id -> current_version_id

    def add_item(self, item: KnowledgeItem) -> str:
        """
        Add a new knowledge item version.

        Args:
            item: The KnowledgeItem to add

        Returns:
            The version ID of the added item
        """
        if item.knowledge_id not in self.knowledge_items:
            self.knowledge_items[item.knowledge_id] = {}

        self.knowledge_items[item.knowledge_id][item.version_id] = item
        self.current_versions[item.knowledge_id] = item.version_id
        return item.version_id

    def get_item(
        self, knowledge_id: str, version_id: Optional[str] = None
    ) -> Optional[KnowledgeItem]:
        """
        Get a specific knowledge item version.

        Args:
            knowledge_id: The ID of the knowledge item
            version_id: The version ID to retrieve, or None for the current version

        Returns:
            The requested KnowledgeItem, or None if not found
        """
        if knowledge_id not in self.knowledge_items:
            return None

        if version_id is None:
            if knowledge_id not in self.current_versions:
                return None
            version_id = self.current_versions[knowledge_id]

        return self.knowledge_items[knowledge_id].get(version_id)

    def set_current_version(self, knowledge_id: str, version_id: str) -> bool:
        """
        Set the current version for a knowledge item.

        Args:
            knowledge_id: The ID of the knowledge item
            version_id: The version ID to set as current

        Returns:
            True if successful, False if the knowledge item or version doesn't exist
        """
        if (
            knowledge_id in self.knowledge_items
            and version_id in self.knowledge_items[knowledge_id]
        ):
            self.current_versions[knowledge_id] = version_id
            return True
        return False

    def list_items(self) -> Dict[str, str]:
        """
        List all knowledge items with their current versions.

        Returns:
            A dictionary mapping knowledge IDs to their current version IDs
        """
        return self.current_versions.copy()

    def list_item_versions(self, knowledge_id: str) -> Dict[str, Dict[str, Any]]:
        """
        List all versions of a specific knowledge item.

        Args:
            knowledge_id: The ID of the knowledge item

        Returns:
            A dictionary mapping version IDs to version metadata
        """
        if knowledge_id not in self.knowledge_items:
            return {}

        return {
            version_id: {
                "timestamp": item.timestamp,
                "source": item.source,
                "confidence": item.confidence,
                "is_current": version_id == self.current_versions.get(knowledge_id),
            }
            for version_id, item in self.knowledge_items[knowledge_id].items()
        }

    def search_items(self, query: Dict[str, Any]) -> Dict[str, KnowledgeItem]:
        """
        Search for knowledge items matching the query.

        Args:
            query: A dictionary of key-value pairs to match against item content

        Returns:
            A dictionary mapping knowledge IDs to their current versions that match the query
        """
        results = {}

        for knowledge_id, version_id in self.current_versions.items():
            item = self.knowledge_items[knowledge_id][version_id]

            # Simple matching algorithm - all query keys must match
            matches = True
            for key, value in query.items():
                if key not in item.content or item.content[key] != value:
                    matches = False
                    break

            if matches:
                results[knowledge_id] = item

        return results

    def extract_fragments(
        self, keyword: str, fragment_path: str, limit: int = 5
    ) -> List[Any]:
        """
        Extract specific fragments from knowledge items matching a keyword.

        Args:
            keyword: Keyword to search for in knowledge item content
            fragment_path: Path to the fragment to extract
            limit: Maximum number of items to extract fragments from

        Returns:
            List of extracted fragments
        """
        results = []
        count = 0

        # Simple keyword search in content
        for knowledge_id, version_id in self.current_versions.items():
            if count >= limit:
                break

            item = self.knowledge_items[knowledge_id][version_id]
            content_str = str(item.content).lower()

            if keyword.lower() in content_str:
                try:
                    fragment = item.extract_fragment(fragment_path)
                    results.append(fragment)
                    count += 1
                except ValueError:
                    # Skip items where the fragment path is invalid
                    continue

        return results

    def compose_prompt_context(
        self, query: str, selectors: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Compose a context for prompts by extracting fragments from knowledge items.

        Args:
            query: Query to search for relevant knowledge items
            selectors: Dictionary mapping context keys to fragment paths

        Returns:
            Dictionary with extracted fragments for each context key
        """
        context = {}
        query_lower = query.lower()

        # Process each selector
        for context_key, fragment_path in selectors.items():
            # Try to match the selector to the appropriate knowledge item
            if "algorithm" in context_key.lower() or "code" in context_key.lower():
                # Look for algorithm/code in matrix-multiply
                if "matrix-multiply" in self.knowledge_items:
                    item = self.get_item("matrix-multiply")
                    if item:
                        try:
                            context[context_key] = item.extract_fragment(fragment_path)
                        except ValueError:
                            pass

            elif "optim" in context_key.lower():
                # Look for optimization in matmul-optimization
                if "matmul-optimization" in self.knowledge_items:
                    item = self.get_item("matmul-optimization")
                    if item:
                        try:
                            context[context_key] = item.extract_fragment(fragment_path)
                        except ValueError:
                            pass

            elif "hardware" in context_key.lower():
                # Look for hardware in hardware-info
                if "hardware-info" in self.knowledge_items:
                    item = self.get_item("hardware-info")
                    if item:
                        try:
                            context[context_key] = item.extract_fragment(fragment_path)
                        except ValueError:
                            pass

            # If we haven't found a match yet, try a general search
            if context_key not in context:
                for knowledge_id, version_id in self.current_versions.items():
                    item = self.knowledge_items[knowledge_id][version_id]
                    content_str = str(item.content).lower()

                    if query_lower in content_str:
                        try:
                            fragment = item.extract_fragment(fragment_path)
                            context[context_key] = fragment
                            break
                        except ValueError:
                            continue

        return context

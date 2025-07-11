"""
Knowledge manager for Pinocchio multi-agent system.

This module provides functionality for managing knowledge fragments with
versioning, session tracking, and multi-agent support.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models.knowledge import (
    KnowledgeCategory,
    KnowledgeContentType,
    KnowledgeExtractionResult,
    KnowledgeFragment,
    KnowledgeQuery,
    KnowledgeVersionHistory,
)


class KnowledgeManager:
    """
    Manager for knowledge fragments with versioning and session tracking.

    Provides functionality for storing, retrieving, and managing knowledge
    fragments used by generator, debugger, and evaluator agents.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the knowledge manager.

        Args:
            storage_path: Path for storing knowledge fragments
        """
        self.storage_path = (
            Path(storage_path) if storage_path else Path("knowledge_storage")
        )
        self.storage_path.mkdir(exist_ok=True)

        # In-memory storage for active fragments
        self.fragments: Dict[str, KnowledgeFragment] = {}
        self.version_histories: Dict[str, KnowledgeVersionHistory] = {}
        self.session_fragments: Dict[str, List[str]] = {}

    def add_fragment(self, fragment: KnowledgeFragment) -> str:
        """
        Add a new knowledge fragment.

        Args:
            fragment: The knowledge fragment to add

        Returns:
            The fragment ID
        """
        # Store fragment
        self.fragments[fragment.fragment_id] = fragment

        # Add to session tracking
        if fragment.session_id:
            if fragment.session_id not in self.session_fragments:
                self.session_fragments[fragment.session_id] = []
            self.session_fragments[fragment.session_id].append(fragment.fragment_id)

        # Add to version history
        if fragment.session_id is not None:
            if fragment.session_id not in self.version_histories:
                self.version_histories[fragment.session_id] = KnowledgeVersionHistory(
                    session_id=fragment.session_id
                )
            self.version_histories[fragment.session_id].add_fragment(fragment)

        # Persist to storage
        self._persist_fragment(fragment)

        return fragment.fragment_id

    def get_fragment(self, fragment_id: str) -> Optional[KnowledgeFragment]:
        """
        Get a knowledge fragment by ID.

        Args:
            fragment_id: The fragment ID

        Returns:
            The knowledge fragment or None if not found
        """
        return self.fragments.get(fragment_id)

    def get_session_fragments(self, session_id: str) -> List[KnowledgeFragment]:
        """
        Get all fragments for a session.

        Args:
            session_id: The session ID

        Returns:
            List of knowledge fragments for the session
        """
        fragment_ids = self.session_fragments.get(session_id, [])
        return [self.fragments[fid] for fid in fragment_ids if fid in self.fragments]

    def search_fragments(self, query: KnowledgeQuery) -> KnowledgeExtractionResult:
        """
        Search for knowledge fragments based on query criteria.

        Args:
            query: The search query

        Returns:
            Knowledge extraction result
        """
        result = KnowledgeExtractionResult(
            session_id=query.session_id, agent_type=query.agent_type
        )

        candidates = []

        # Filter by session if specified
        if query.session_id:
            candidates = self.get_session_fragments(query.session_id)
        else:
            candidates = list(self.fragments.values())

        # Apply filters
        for fragment in candidates:
            if not self._matches_query(fragment, query):
                continue

            # Apply keyword search if specified
            if query.keywords and not self._matches_keywords(fragment, query.keywords):
                continue

            result.add_fragment(fragment)

            # Apply limit
            if result.total_count >= query.limit:
                break

        return result

    def extract_agent_knowledge(
        self, session_id: str, agent_type: str
    ) -> KnowledgeExtractionResult:
        """
        Extract knowledge fragments for a specific agent in a session.

        Args:
            session_id: The session ID
            agent_type: The agent type (generator/debugger/evaluator)

        Returns:
            Knowledge extraction result for the agent
        """
        query = KnowledgeQuery(session_id=session_id, agent_type=agent_type, limit=50)
        return self.search_fragments(query)

    def extract_optimization_knowledge(
        self, session_id: str
    ) -> List[KnowledgeFragment]:
        """
        Extract optimization knowledge fragments for a session.

        Args:
            session_id: The session ID

        Returns:
            List of optimization knowledge fragments
        """
        query = KnowledgeQuery(
            session_id=session_id, category=KnowledgeCategory.OPTIMIZATION, limit=20
        )
        result = self.search_fragments(query)
        return result.fragments

    def extract_domain_knowledge(self, session_id: str) -> List[KnowledgeFragment]:
        """
        Extract domain knowledge fragments for a session.

        Args:
            session_id: The session ID

        Returns:
            List of domain knowledge fragments
        """
        query = KnowledgeQuery(
            session_id=session_id, category=KnowledgeCategory.DOMAIN, limit=20
        )
        result = self.search_fragments(query)
        return result.fragments

    def extract_dsl_knowledge(self, session_id: str) -> List[KnowledgeFragment]:
        """
        Extract DSL knowledge fragments for a session.

        Args:
            session_id: The session ID

        Returns:
            List of DSL knowledge fragments
        """
        query = KnowledgeQuery(
            session_id=session_id, category=KnowledgeCategory.DSL, limit=20
        )
        result = self.search_fragments(query)
        return result.fragments

    def get_version_history(
        self, session_id: Optional[str] = None
    ) -> Optional[KnowledgeVersionHistory]:
        """
        Get version history for a session.

        Args:
            session_id: Session ID

        Returns:
            Version history or None if not found
        """
        # Get version history
        if session_id is not None:
            if session_id not in self.version_histories:
                return None
            return self.version_histories[session_id]
        else:
            # Return None for fragments without session
            return None

    def create_fragment_from_content(
        self,
        session_id: Optional[str],
        agent_type: Optional[str],
        category: KnowledgeCategory,
        title: str,
        content: Any,
        content_type: KnowledgeContentType,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeFragment:
        """
        Create a new knowledge fragment from content.

        Args:
            session_id: Session ID this fragment belongs to
            agent_type: Type of agent this fragment is for
            category: Knowledge category
            title: Fragment title
            content: Fragment content
            content_type: Type of content
            metadata: Additional metadata

        Returns:
            The created knowledge fragment
        """
        fragment = KnowledgeFragment.create_fragment(
            session_id=session_id,
            agent_type=agent_type,
            category=category,
            title=title,
            content=content,
            content_type=content_type,
            metadata=metadata,
        )

        self.add_fragment(fragment)
        return fragment

    def _matches_query(
        self, fragment: KnowledgeFragment, query: KnowledgeQuery
    ) -> bool:
        """
        Check if a fragment matches the query criteria.

        Args:
            fragment: The knowledge fragment
            query: The search query

        Returns:
            True if the fragment matches the query
        """
        # Check session ID
        if query.session_id and fragment.session_id != query.session_id:
            return False

        # Check agent type
        if query.agent_type and fragment.agent_type != query.agent_type:
            return False

        # Check category
        if query.category and fragment.category != query.category:
            return False

        # Check content type
        if query.content_type and fragment.content_type != query.content_type:
            return False

        return True

    def _matches_keywords(
        self, fragment: KnowledgeFragment, keywords: List[str]
    ) -> bool:
        """
        Check if a fragment matches the keywords.

        Args:
            fragment: The knowledge fragment
            keywords: List of keywords to search for

        Returns:
            True if the fragment matches any keyword
        """
        if not keywords:
            return True

        # Search in title and content
        search_text = f"{fragment.title} {str(fragment.content)}".lower()

        for keyword in keywords:
            if keyword.lower() in search_text:
                return True

        return False

    def _persist_fragment(self, fragment: KnowledgeFragment) -> None:
        """
        Persist a fragment to storage.

        Args:
            fragment: The knowledge fragment to persist
        """
        fragment_file = self.storage_path / f"{fragment.fragment_id}.json"

        with open(fragment_file, "w") as f:
            json.dump(fragment.model_dump(), f, indent=2, default=str)

    def load_from_storage(self) -> None:
        """
        Load knowledge fragments from storage.
        """
        for fragment_file in self.storage_path.glob("*.json"):
            try:
                with open(fragment_file, "r") as f:
                    data = json.load(f)
                    fragment = KnowledgeFragment(**data)
                    self.fragments[fragment.fragment_id] = fragment

                    # Rebuild session tracking
                    if fragment.session_id:
                        if fragment.session_id not in self.session_fragments:
                            self.session_fragments[fragment.session_id] = []
                        self.session_fragments[fragment.session_id].append(
                            fragment.fragment_id
                        )

                    # Rebuild version history
                    if fragment.session_id is not None:
                        if fragment.session_id not in self.version_histories:
                            self.version_histories[
                                fragment.session_id
                            ] = KnowledgeVersionHistory(session_id=fragment.session_id)
                        self.version_histories[fragment.session_id].add_fragment(
                            fragment
                        )

            except Exception as e:
                print(f"Error loading fragment from {fragment_file}: {e}")

    def clear_session(self, session_id: str) -> None:
        """
        Clear all fragments for a session.

        Args:
            session_id: The session ID to clear
        """
        fragment_ids = self.session_fragments.get(session_id, [])

        for fragment_id in fragment_ids:
            if fragment_id in self.fragments:
                del self.fragments[fragment_id]

        if session_id in self.session_fragments:
            del self.session_fragments[session_id]

        if session_id in self.version_histories:
            del self.version_histories[session_id]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get knowledge manager statistics.

        Returns:
            Dictionary with statistics
        """
        total_fragments = len(self.fragments)
        total_sessions = len(self.session_fragments)
        total_versions = len(self.version_histories)

        # Calculate statistics
        category_counts: Dict[str, int] = {}
        content_type_counts: Dict[str, int] = {}

        for fragment in self.fragments.values():
            category = (
                fragment.category.value
                if hasattr(fragment.category, "value")
                else fragment.category
            )
            category_counts[category] = category_counts.get(category, 0) + 1

            content_type = (
                fragment.content_type.value
                if hasattr(fragment.content_type, "value")
                else fragment.content_type
            )
            content_type_counts[content_type] = (
                content_type_counts.get(content_type, 0) + 1
            )

        return {
            "total_fragments": total_fragments,
            "total_sessions": total_sessions,
            "total_versions": total_versions,
            "category_counts": category_counts,
            "content_type_counts": content_type_counts,
        }

    def get_fragments_by_session(
        self, session_id: Optional[str] = None
    ) -> List[KnowledgeFragment]:
        """
        Get fragments by session ID.

        Args:
            session_id: Session ID to filter by

        Returns:
            List of knowledge fragments
        """
        if session_id is None:
            return list(self.fragments.values())

        return [
            fragment
            for fragment in self.fragments.values()
            if fragment.session_id == session_id
        ]

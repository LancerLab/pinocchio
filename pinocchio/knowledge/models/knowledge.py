"""
Knowledge models for Pinocchio multi-agent system.

This module defines the models for managing knowledge fragments with versioning,
session tracking, and multi-agent support.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class KnowledgeCategory(str, Enum):
    """Knowledge fragment categories."""

    DOMAIN = "domain"
    OPTIMIZATION = "optimization"
    DSL = "dsl"
    OTHER = "other"


class KnowledgeContentType(str, Enum):
    """Knowledge content types."""

    JSON = "json"
    MARKDOWN = "markdown"
    TEXT = "text"


class KnowledgeType(str, Enum):
    """Knowledge fragment types for workflow integration."""

    OPTIMIZATION_TECHNIQUE = "optimization_technique"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    # Extend more types as needed


class KnowledgeFragment(BaseModel):
    """Knowledge fragment with versioning and session tracking."""

    fragment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    agent_type: Optional[str] = None  # generator/debugger/evaluator
    category: KnowledgeCategory
    title: str
    content: Any  # Can be structured JSON or Markdown text
    content_type: KnowledgeContentType
    version: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    parent_version: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration for KnowledgeFragment."""

        use_enum_values = True

    @classmethod
    def create_fragment(
        cls,
        session_id: Optional[str],
        agent_type: Optional[str],
        category: KnowledgeCategory,
        title: str,
        content: Any,
        content_type: KnowledgeContentType,
        parent_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "KnowledgeFragment":
        """
        Create a new knowledge fragment.

        Args:
            session_id: Session ID this fragment belongs to
            agent_type: Type of agent this fragment is for
            category: Knowledge category
            title: Fragment title
            content: Fragment content
            content_type: Type of content
            parent_version: Parent version ID if this is a new version
            metadata: Additional metadata

        Returns:
            A new KnowledgeFragment instance
        """
        # Generate version based on content and timestamp
        version = (
            f"v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )

        return cls(
            session_id=session_id,
            agent_type=agent_type,
            category=category,
            title=title,
            content=content,
            content_type=content_type,
            version=version,
            parent_version=parent_version,
            metadata=metadata or {},
        )

    def extract_structured_data(self) -> Dict[str, Any]:
        """
        Extract structured data from the fragment content.

        Returns:
            Structured data dictionary
        """
        if self.content_type == KnowledgeContentType.JSON:
            return self.content if isinstance(self.content, dict) else {}
        elif self.content_type == KnowledgeContentType.MARKDOWN:
            # Extract code blocks, headers, etc.
            return {"markdown_content": self.content}
        else:
            return {"text_content": self.content}


class KnowledgeVersionHistory(BaseModel):
    """Knowledge version history for tracking fragment evolution."""

    session_id: Optional[str] = None
    fragments: List[KnowledgeFragment] = Field(default_factory=list)
    version_chain: List[str] = Field(default_factory=list)

    def add_fragment(self, fragment: KnowledgeFragment) -> None:
        """
        Add a new fragment to the version history.

        Args:
            fragment: The knowledge fragment to add
        """
        self.fragments.append(fragment)
        self.version_chain.append(fragment.version)

    def get_latest_fragment(self) -> Optional[KnowledgeFragment]:
        """
        Get the latest fragment in the version history.

        Returns:
            The latest fragment or None if empty
        """
        return self.fragments[-1] if self.fragments else None

    def get_fragment_by_version(self, version: str) -> Optional[KnowledgeFragment]:
        """
        Get a specific fragment by version.

        Args:
            version: Version string to search for

        Returns:
            The fragment with the specified version or None
        """
        for fragment in self.fragments:
            if fragment.version == version:
                return fragment
        return None


class KnowledgeQuery(BaseModel):
    """Query model for searching knowledge fragments."""

    session_id: Optional[str] = None
    agent_type: Optional[str] = None
    category: Optional[KnowledgeCategory] = None
    keywords: List[str] = Field(default_factory=list)
    content_type: Optional[KnowledgeContentType] = None
    limit: int = 10

    class Config:
        """Pydantic configuration for KnowledgeQuery."""

        use_enum_values = True


class KnowledgeExtractionResult(BaseModel):
    """Result of knowledge fragment extraction."""

    session_id: Optional[str] = None
    agent_type: Optional[str] = None
    fragments: List[KnowledgeFragment] = Field(default_factory=list)
    total_count: int = 0
    extraction_time: datetime = Field(default_factory=datetime.utcnow)

    def add_fragment(self, fragment: KnowledgeFragment) -> None:
        """
        Add a fragment to the extraction result.

        Args:
            fragment: The knowledge fragment to add
        """
        self.fragments.append(fragment)
        self.total_count += 1

    def get_fragments_by_category(
        self, category: KnowledgeCategory
    ) -> List[KnowledgeFragment]:
        """
        Get fragments filtered by category.

        Args:
            category: Category to filter by

        Returns:
            List of fragments in the specified category
        """
        return [f for f in self.fragments if f.category == category]

    def get_structured_knowledge(self) -> Dict[str, Any]:
        """
        Get structured knowledge organized by category.

        Returns:
            Dictionary with knowledge organized by category
        """
        structured: Dict[str, Any] = {}
        for fragment in self.fragments:
            category = (
                fragment.category.value
                if hasattr(fragment.category, "value")
                else fragment.category
            )
            if category not in structured:
                structured[category] = []
            content_type = (
                fragment.content_type.value
                if hasattr(fragment.content_type, "value")
                else fragment.content_type
            )
            structured[category].append(
                {
                    "title": fragment.title,
                    "content": fragment.content,
                    "content_type": content_type,
                    "version": fragment.version,
                    "metadata": fragment.metadata,
                }
            )
        return structured

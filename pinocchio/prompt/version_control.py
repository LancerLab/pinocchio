"""
Pinocchio prompt version control system.

This module provides version control functionality for prompt templates,
including branching, merging, and version history management.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import PromptTemplate


class VersionStatus(Enum):
    """Version status enumeration."""

    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class VersionInfo:
    """Version information for templates."""

    version_id: str
    parent_version_id: Optional[str] = None
    branch_name: str = "main"
    status: VersionStatus = VersionStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    description: str = ""
    change_log: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "parent_version_id": self.parent_version_id,
            "branch_name": self.branch_name,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "description": self.description,
            "change_log": self.change_log,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VersionInfo":
        """Create from dictionary."""
        # Convert string enum back to enum object
        if "status" in data and isinstance(data["status"], str):
            data["status"] = VersionStatus(data["status"])

        # Convert datetime string back to datetime object
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])

        return cls(**data)


@dataclass
class BranchInfo:
    """Branch information."""

    branch_name: str
    head_version_id: str
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    description: str = ""
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "branch_name": self.branch_name,
            "head_version_id": self.head_version_id,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "description": self.description,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BranchInfo":
        """Create from dictionary."""
        # Convert datetime string back to datetime object
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])

        return cls(**data)


class VersionControl:
    """
    Version control system for prompt templates.

    Handles versioning, branching, merging, and history management.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize version control system.

        Args:
            storage_path: Path to store version control data
        """
        self.storage_path = (
            Path(storage_path) if storage_path else Path("./version_control")
        )
        self.storage_path.mkdir(exist_ok=True)

        # Version tracking: {template_name: {version_id: VersionInfo}}
        self.versions: Dict[str, Dict[str, VersionInfo]] = {}

        # Branch tracking: {template_name: {branch_name: BranchInfo}}
        self.branches: Dict[str, Dict[str, BranchInfo]] = {}

        # Load existing version control data
        self._load_version_data()

    def create_version(
        self,
        template: PromptTemplate,
        branch_name: str = "main",
        description: str = "",
        created_by: str = "",
        status: VersionStatus = VersionStatus.DRAFT,
    ) -> VersionInfo:
        """
        Create a new version of a template.

        Args:
            template: The template to version
            branch_name: Branch name for the version
            description: Version description
            created_by: Creator identifier
            status: Version status

        Returns:
            Version information
        """
        if template.template_name not in self.versions:
            self.versions[template.template_name] = {}

        # Find parent version (latest in branch)
        parent_version_id = None
        if (
            template.template_name in self.branches
            and branch_name in self.branches[template.template_name]
        ):
            parent_version_id = self.branches[template.template_name][
                branch_name
            ].head_version_id

        version_info = VersionInfo(
            version_id=template.version_id,
            parent_version_id=parent_version_id,
            branch_name=branch_name,
            status=status,
            created_by=created_by,
            description=description,
        )

        self.versions[template.template_name][template.version_id] = version_info

        # Update branch head
        if template.template_name not in self.branches:
            self.branches[template.template_name] = {}

        self.branches[template.template_name][branch_name] = BranchInfo(
            branch_name=branch_name,
            head_version_id=template.version_id,
            created_by=created_by,
            description=description,
        )

        self._save_version_data()
        return version_info

    def get_version_info(
        self, template_name: str, version_id: str
    ) -> Optional[VersionInfo]:
        """
        Get version information.

        Args:
            template_name: Name of the template
            version_id: Version ID

        Returns:
            Version information or None if not found
        """
        return self.versions.get(template_name, {}).get(version_id)

    def list_versions(
        self, template_name: str, branch_name: Optional[str] = None
    ) -> List[VersionInfo]:
        """
        List versions of a template.

        Args:
            template_name: Name of the template
            branch_name: Filter by branch name (optional)

        Returns:
            List of version information
        """
        if template_name not in self.versions:
            return []

        versions = list(self.versions[template_name].values())

        if branch_name is not None:
            versions = [v for v in versions if v.branch_name == branch_name]

        # Sort by creation time (newest first)
        versions.sort(key=lambda v: v.created_at, reverse=True)
        return versions

    def list_branches(self, template_name: str) -> List[BranchInfo]:
        """
        List branches for a template.

        Args:
            template_name: Name of the template

        Returns:
            List of branch information
        """
        if template_name not in self.branches:
            return []

        return list(self.branches[template_name].values())

    def create_branch(
        self,
        template_name: str,
        branch_name: str,
        from_version_id: str,
        created_by: str = "",
        description: str = "",
    ) -> bool:
        """
        Create a new branch from a specific version.

        Args:
            template_name: Name of the template
            branch_name: Name of the new branch
            from_version_id: Version ID to branch from
            created_by: Creator identifier
            description: Branch description

        Returns:
            True if successful, False otherwise
        """
        if template_name not in self.versions:
            return False

        if from_version_id not in self.versions[template_name]:
            return False

        if template_name not in self.branches:
            self.branches[template_name] = {}

        if branch_name in self.branches[template_name]:
            return False  # Branch already exists

        self.branches[template_name][branch_name] = BranchInfo(
            branch_name=branch_name,
            head_version_id=from_version_id,
            created_by=created_by,
            description=description,
        )

        self._save_version_data()
        return True

    def merge_branch(
        self,
        template_name: str,
        source_branch: str,
        target_branch: str,
        merge_strategy: str = "fast-forward",
    ) -> bool:
        """
        Merge one branch into another.

        Args:
            template_name: Name of the template
            source_branch: Source branch name
            target_branch: Target branch name
            merge_strategy: Merge strategy ("fast-forward", "merge", "rebase")

        Returns:
            True if successful, False otherwise
        """
        if template_name not in self.branches:
            return False

        if (
            source_branch not in self.branches[template_name]
            or target_branch not in self.branches[template_name]
        ):
            return False

        source_head = self.branches[template_name][source_branch].head_version_id
        # target_head = self.branches[template_name][target_branch].head_version_id

        # For now, implement simple fast-forward merge
        if merge_strategy == "fast-forward":
            # Update target branch head to source branch head
            self.branches[template_name][target_branch].head_version_id = source_head
            self._save_version_data()
            return True

        # Other merge strategies would require more complex logic
        return False

    def update_version_status(
        self, template_name: str, version_id: str, status: VersionStatus
    ) -> bool:
        """
        Update version status.

        Args:
            template_name: Name of the template
            version_id: Version ID
            status: New status

        Returns:
            True if successful, False otherwise
        """
        if template_name not in self.versions:
            return False

        if version_id not in self.versions[template_name]:
            return False

        self.versions[template_name][version_id].status = status
        self._save_version_data()
        return True

    def add_change_log(self, template_name: str, version_id: str, change: str) -> bool:
        """
        Add a change log entry to a version.

        Args:
            template_name: Name of the template
            version_id: Version ID
            change: Change description

        Returns:
            True if successful, False otherwise
        """
        if template_name not in self.versions:
            return False

        if version_id not in self.versions[template_name]:
            return False

        self.versions[template_name][version_id].change_log.append(change)
        self._save_version_data()
        return True

    def get_version_history(
        self, template_name: str, version_id: str
    ) -> List[VersionInfo]:
        """
        Get version history (ancestors) for a specific version.

        Args:
            template_name: Name of the template
            version_id: Version ID

        Returns:
            List of ancestor versions (oldest first)
        """
        if template_name not in self.versions:
            return []

        history = []
        current_version_id: Optional[str] = version_id

        while current_version_id is not None:
            if current_version_id not in self.versions[template_name]:
                break

            version_info = self.versions[template_name][current_version_id]
            history.append(version_info)
            current_version_id = version_info.parent_version_id

        # Reverse to get oldest first
        history.reverse()
        return history

    def get_latest_version(
        self, template_name: str, branch_name: str = "main"
    ) -> Optional[VersionInfo]:
        """
        Get the latest version in a branch.

        Args:
            template_name: Name of the template
            branch_name: Branch name

        Returns:
            Latest version information or None if not found
        """
        if template_name not in self.branches:
            return None

        if branch_name not in self.branches[template_name]:
            return None

        head_version_id = self.branches[template_name][branch_name].head_version_id
        return self.get_version_info(template_name, head_version_id)

    def delete_branch(self, template_name: str, branch_name: str) -> bool:
        """
        Delete a branch.

        Args:
            template_name: Name of the template
            branch_name: Branch name to delete

        Returns:
            True if successful, False otherwise
        """
        if template_name not in self.branches:
            return False

        if branch_name not in self.branches[template_name]:
            return False

        # Don't allow deleting main branch
        if branch_name == "main":
            return False

        del self.branches[template_name][branch_name]
        self._save_version_data()
        return True

    def archive_version(self, template_name: str, version_id: str) -> bool:
        """
        Archive a version.

        Args:
            template_name: Name of the template
            version_id: Version ID to archive

        Returns:
            True if successful, False otherwise
        """
        return self.update_version_status(
            template_name, version_id, VersionStatus.ARCHIVED
        )

    def get_version_diff(
        self, template_name: str, version_id_1: str, version_id_2: str
    ) -> Dict[str, Any]:
        """
        Get differences between two versions.

        Args:
            template_name: Name of the template
            version_id_1: First version ID
            version_id_2: Second version ID

        Returns:
            Dictionary containing differences
        """
        # This is a placeholder - implement actual diff logic
        # For now, return basic information
        return {
            "template_name": template_name,
            "version_1": version_id_1,
            "version_2": version_id_2,
            "differences": "Diff logic not implemented",
        }

    def _save_versions_data(self) -> None:
        """Save versions data to storage."""
        versions_file = self.storage_path / "versions.json"
        versions_data = {}
        for template_name, versions in self.versions.items():
            versions_data[template_name] = {
                version_id: version_info.to_dict()
                for version_id, version_info in versions.items()
            }

        with open(versions_file, "w") as f:
            json.dump(versions_data, f, indent=2)

    def _save_branches_data(self) -> None:
        """Save branches data to storage."""
        branches_file = self.storage_path / "branches.json"
        branches_data = {}
        for template_name, branches in self.branches.items():
            branches_data[template_name] = {
                branch_name: branch_info.to_dict()
                for branch_name, branch_info in branches.items()
            }

        with open(branches_file, "w") as f:
            json.dump(branches_data, f, indent=2)

    def _save_version_data(self) -> None:
        """Save version control data to storage."""
        self._save_versions_data()
        self._save_branches_data()

    def _load_version_data(self) -> None:
        """Load version control data from storage."""
        # Load versions
        versions_file = self.storage_path / "versions.json"
        if versions_file.exists():
            try:
                with open(versions_file, "r") as f:
                    versions_data = json.load(f)

                for template_name, versions in versions_data.items():
                    self.versions[template_name] = {}
                    for version_id, version_info_data in versions.items():
                        version_info = VersionInfo.from_dict(version_info_data)
                        self.versions[template_name][version_id] = version_info
            except Exception as e:
                print(f"Error loading versions: {e}")

        # Load branches
        branches_file = self.storage_path / "branches.json"
        if branches_file.exists():
            try:
                with open(branches_file, "r") as f:
                    branches_data = json.load(f)

                for template_name, branches in branches_data.items():
                    self.branches[template_name] = {}
                    for branch_name, branch_info_data in branches.items():
                        branch_info = BranchInfo.from_dict(branch_info_data)
                        self.branches[template_name][branch_name] = branch_info
            except Exception as e:
                print(f"Error loading branches: {e}")

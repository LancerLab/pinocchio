"""
Tests for the version control system.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from pinocchio.prompt import AgentType, PromptTemplate, PromptType
from pinocchio.prompt.version_control import (
    BranchInfo,
    VersionControl,
    VersionInfo,
    VersionStatus,
)
from tests.utils import (
    assert_session_valid,
    assert_task_valid,
    create_mock_llm_client,
    create_test_session,
    create_test_task,
)


class TestVersionInfo:
    """Tests for VersionInfo model."""

    def test_version_info_creation(self):
        """Test VersionInfo creation."""
        version_info = VersionInfo(
            version_id="test-version",
            parent_version_id="parent-version",
            branch_name="main",
            status=VersionStatus.DRAFT,
            created_by="test_user",
            description="Test version",
            change_log=["Initial version"],
            tags=["test"],
        )

        assert version_info.version_id == "test-version"
        assert version_info.parent_version_id == "parent-version"
        assert version_info.branch_name == "main"
        assert version_info.status == VersionStatus.DRAFT
        assert version_info.created_by == "test_user"
        assert version_info.description == "Test version"
        assert "Initial version" in version_info.change_log
        assert "test" in version_info.tags

    def test_version_info_to_dict(self):
        """Test VersionInfo to_dict conversion."""
        version_info = VersionInfo(
            version_id="test-version", branch_name="main", status=VersionStatus.APPROVED
        )

        result = version_info.to_dict()
        assert result["version_id"] == "test-version"
        assert result["branch_name"] == "main"
        assert result["status"] == "approved"

    def test_version_info_from_dict(self):
        """Test VersionInfo from_dict creation."""
        data = {
            "version_id": "test-version",
            "branch_name": "main",
            "status": "draft",
            "created_by": "test_user",
            "description": "Test version",
            "change_log": ["Initial version"],
            "tags": ["test"],
        }

        version_info = VersionInfo.from_dict(data)
        assert version_info.version_id == "test-version"
        assert version_info.branch_name == "main"
        assert version_info.status == VersionStatus.DRAFT
        assert version_info.created_by == "test_user"


class TestBranchInfo:
    """Tests for BranchInfo model."""

    def test_branch_info_creation(self):
        """Test BranchInfo creation."""
        branch_info = BranchInfo(
            branch_name="feature-branch",
            head_version_id="head-version",
            created_by="test_user",
            description="Feature branch",
            is_active=True,
        )

        assert branch_info.branch_name == "feature-branch"
        assert branch_info.head_version_id == "head-version"
        assert branch_info.created_by == "test_user"
        assert branch_info.description == "Feature branch"
        assert branch_info.is_active is True

    def test_branch_info_to_dict(self):
        """Test BranchInfo to_dict conversion."""
        branch_info = BranchInfo(branch_name="main", head_version_id="head-version")

        result = branch_info.to_dict()
        assert result["branch_name"] == "main"
        assert result["head_version_id"] == "head-version"
        assert result["is_active"] is True

    def test_branch_info_from_dict(self):
        """Test BranchInfo from_dict creation."""
        data = {
            "branch_name": "main",
            "head_version_id": "head-version",
            "created_by": "test_user",
            "description": "Main branch",
            "is_active": True,
        }

        branch_info = BranchInfo.from_dict(data)
        assert branch_info.branch_name == "main"
        assert branch_info.head_version_id == "head-version"
        assert branch_info.created_by == "test_user"
        assert branch_info.description == "Main branch"
        assert branch_info.is_active is True


class TestVersionControl:
    """Tests for VersionControl system."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def version_control(self, temp_dir):
        """Create a VersionControl instance."""
        return VersionControl(storage_path=temp_dir)

    @pytest.fixture
    def sample_template(self):
        """Create a sample template for testing."""
        return PromptTemplate.create_new_version(
            template_name="test_template",
            content="Generate {{task}}",
            agent_type=AgentType.GENERATOR,
            prompt_type=PromptType.CODE_GENERATION,
        )

    def test_version_control_creation(self, temp_dir):
        """Test VersionControl creation."""
        vc = VersionControl(temp_dir)
        assert vc.storage_path == Path(temp_dir)
        assert isinstance(vc.versions, dict)
        assert isinstance(vc.branches, dict)

    def test_create_version(self, version_control, sample_template):
        """Test creating a version."""
        version_control.create_version(
            template=sample_template,
            branch_name="main",
            description="Initial version",
            created_by="test_user",
            status=VersionStatus.DRAFT,
        )

        assert (
            version_control.get_version_info(
                sample_template.template_name, sample_template.version_id
            ).version_id
            == sample_template.version_id
        )
        assert (
            version_control.get_version_info(
                sample_template.template_name, sample_template.version_id
            ).branch_name
            == "main"
        )
        assert (
            version_control.get_version_info(
                sample_template.template_name, sample_template.version_id
            ).status
            == VersionStatus.DRAFT
        )
        assert (
            version_control.get_version_info(
                sample_template.template_name, sample_template.version_id
            ).created_by
            == "test_user"
        )
        assert (
            version_control.get_version_info(
                sample_template.template_name, sample_template.version_id
            ).description
            == "Initial version"
        )

    def test_get_version_info(self, version_control, sample_template):
        """Test getting version information."""
        version_control.create_version(sample_template)

        retrieved = version_control.get_version_info(
            sample_template.template_name, sample_template.version_id
        )

        assert retrieved is not None
        assert retrieved.version_id == sample_template.version_id

    def test_list_versions(self, version_control, sample_template):
        """Test listing versions."""
        # Create multiple versions
        version_control.create_version(sample_template, branch_name="main")

        template2 = PromptTemplate.create_new_version(
            template_name=sample_template.template_name, content="Updated content"
        )
        version_control.create_version(template2, branch_name="main")

        versions = version_control.list_versions(sample_template.template_name)
        assert len(versions) == 2

        # Test filtering by branch
        versions_main = version_control.list_versions(
            sample_template.template_name, branch_name="main"
        )
        assert len(versions_main) == 2

    def test_list_branches(self, version_control, sample_template):
        """Test listing branches."""
        # Create versions in different branches
        version_control.create_version(sample_template, branch_name="main")

        template2 = PromptTemplate.create_new_version(
            template_name=sample_template.template_name, content="Feature content"
        )
        version_control.create_version(template2, branch_name="feature")

        branches = version_control.list_branches(sample_template.template_name)
        assert len(branches) == 2

        branch_names = [b.branch_name for b in branches]
        assert "main" in branch_names
        assert "feature" in branch_names

    def test_create_branch(self, version_control, sample_template):
        """Test creating a new branch."""
        # Create initial version
        version_control.create_version(sample_template)

        # Create new branch
        success = version_control.create_branch(
            template_name=sample_template.template_name,
            branch_name="feature",
            from_version_id=sample_template.version_id,
            created_by="test_user",
            description="Feature branch",
        )

        assert success is True

        # Verify branch exists
        branches = version_control.list_branches(sample_template.template_name)
        branch_names = [b.branch_name for b in branches]
        assert "feature" in branch_names

    def test_create_branch_from_nonexistent_version(self, version_control):
        """Test creating branch from nonexistent version."""
        success = version_control.create_branch(
            template_name="nonexistent",
            branch_name="feature",
            from_version_id="nonexistent-version",
        )

        assert success is False

    def test_create_duplicate_branch(self, version_control, sample_template):
        """Test creating duplicate branch."""
        # Create initial version and branch
        version_control.create_version(sample_template)
        version_control.create_branch(
            template_name=sample_template.template_name,
            branch_name="feature",
            from_version_id=sample_template.version_id,
        )

        # Try to create duplicate branch
        success = version_control.create_branch(
            template_name=sample_template.template_name,
            branch_name="feature",
            from_version_id=sample_template.version_id,
        )

        assert success is False

    def test_merge_branch(self, version_control, sample_template):
        """Test merging branches."""
        # Create versions in different branches
        version_control.create_version(sample_template, branch_name="main")

        template2 = PromptTemplate.create_new_version(
            template_name=sample_template.template_name, content="Feature content"
        )
        version_control.create_version(template2, branch_name="feature")

        # Merge feature into main
        success = version_control.merge_branch(
            template_name=sample_template.template_name,
            source_branch="feature",
            target_branch="main",
        )

        assert success is True

    def test_merge_nonexistent_branches(self, version_control):
        """Test merging nonexistent branches."""
        success = version_control.merge_branch(
            template_name="nonexistent", source_branch="source", target_branch="target"
        )

        assert success is False

    def test_update_version_status(self, version_control, sample_template):
        """Test updating version status."""
        version_control.create_version(sample_template)

        success = version_control.update_version_status(
            template_name=sample_template.template_name,
            version_id=sample_template.version_id,
            status=VersionStatus.APPROVED,
        )

        assert success is True

        # Verify status is updated
        version_info = version_control.get_version_info(
            sample_template.template_name, sample_template.version_id
        )
        assert version_info.status == VersionStatus.APPROVED

    def test_add_change_log(self, version_control, sample_template):
        """Test adding change log entry."""
        version_control.create_version(sample_template)

        success = version_control.add_change_log(
            template_name=sample_template.template_name,
            version_id=sample_template.version_id,
            change="Updated template content",
        )

        assert success is True

        # Verify change log is added
        version_info = version_control.get_version_info(
            sample_template.template_name, sample_template.version_id
        )
        assert "Updated template content" in version_info.change_log

    def test_get_version_history(self, version_control, sample_template):
        """Test getting version history."""
        # Create version chain
        version_control.create_version(sample_template)

        template2 = PromptTemplate.create_new_version(
            template_name=sample_template.template_name,
            content="Updated content",
            parent_version_id=version_control.get_version_info(
                sample_template.template_name, sample_template.version_id
            ).version_id,
        )
        version2 = version_control.create_version(template2)

        template3 = PromptTemplate.create_new_version(
            template_name=sample_template.template_name,
            content="Final content",
            parent_version_id=version2.version_id,
        )
        version3 = version_control.create_version(template3)

        # Get history
        history = version_control.get_version_history(
            sample_template.template_name, version3.version_id
        )

        assert len(history) == 3
        assert (
            history[0].version_id
            == version_control.get_version_info(
                sample_template.template_name, sample_template.version_id
            ).version_id
        )
        assert history[1].version_id == version2.version_id
        assert history[2].version_id == version3.version_id

    def test_get_latest_version(self, version_control, sample_template):
        """Test getting latest version in branch."""
        version_control.create_version(sample_template, branch_name="main")

        template2 = PromptTemplate.create_new_version(
            template_name=sample_template.template_name, content="Updated content"
        )
        version_control.create_version(template2, branch_name="main")

        latest = version_control.get_latest_version(
            sample_template.template_name, branch_name="main"
        )

        assert latest is not None
        assert latest.version_id == template2.version_id

    def test_delete_branch(self, version_control, sample_template):
        """Test deleting a branch."""
        # Create branch
        version_control.create_version(sample_template, branch_name="feature")

        # Delete branch
        success = version_control.delete_branch(
            template_name=sample_template.template_name, branch_name="feature"
        )

        assert success is True

        # Verify branch is deleted
        branches = version_control.list_branches(sample_template.template_name)
        branch_names = [b.branch_name for b in branches]
        assert "feature" not in branch_names

    def test_delete_main_branch(self, version_control, sample_template):
        """Test that main branch cannot be deleted."""
        version_control.create_version(sample_template, branch_name="main")

        success = version_control.delete_branch(
            template_name=sample_template.template_name, branch_name="main"
        )

        assert success is False

    def test_archive_version(self, version_control, sample_template):
        """Test archiving a version."""
        version_control.create_version(sample_template)

        success = version_control.archive_version(
            template_name=sample_template.template_name,
            version_id=sample_template.version_id,
        )

        assert success is True

        # Verify version is archived
        version_info = version_control.get_version_info(
            sample_template.template_name, sample_template.version_id
        )
        assert version_info.status == VersionStatus.ARCHIVED

    def test_get_version_diff(self, version_control, sample_template):
        """Test getting version differences."""
        version_control.create_version(sample_template)

        template2 = PromptTemplate.create_new_version(
            template_name=sample_template.template_name, content="Different content"
        )
        version_control.create_version(template2)

        diff = version_control.get_version_diff(
            template_name=sample_template.template_name,
            version_id_1=sample_template.version_id,
            version_id_2=template2.version_id,
        )

        assert diff["template_name"] == sample_template.template_name
        assert diff["version_1"] == sample_template.version_id
        assert diff["version_2"] == template2.version_id

    def test_persistence(self, temp_dir):
        """Test version control persistence."""
        # Create version control and add data
        vc1 = VersionControl(temp_dir)
        template = PromptTemplate.create_new_version("test", "Content")
        vc1.create_version(template, branch_name="main")

        # Create new instance
        vc2 = VersionControl(temp_dir)

        # Verify data is loaded
        versions = vc2.list_versions("test")
        assert len(versions) == 1

        branches = vc2.list_branches("test")
        assert len(branches) == 1
        assert branches[0].branch_name == "main"

    def test_edge_cases(self, version_control):
        """Test edge cases."""
        # Get version info for nonexistent template
        version_info = version_control.get_version_info("nonexistent", "nonexistent")
        assert version_info is None

        # List versions for nonexistent template
        versions = version_control.list_versions("nonexistent")
        assert len(versions) == 0

        # List branches for nonexistent template
        branches = version_control.list_branches("nonexistent")
        assert len(branches) == 0

        # Update status for nonexistent version
        success = version_control.update_version_status(
            "nonexistent", "nonexistent", VersionStatus.APPROVED
        )
        assert success is False

        # Add change log for nonexistent version
        success = version_control.add_change_log("nonexistent", "nonexistent", "change")
        assert success is False

        # Get latest version for nonexistent branch
        latest = version_control.get_latest_version("nonexistent", "nonexistent")
        assert latest is None

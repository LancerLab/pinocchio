"""Unit tests for Coordinator."""

import shutil
import tempfile
from unittest.mock import AsyncMock, Mock

import pytest

from pinocchio.coordinator import Coordinator
from pinocchio.llm.mock_client import MockLLMClient
from tests.utils import create_mock_llm_client


class TestCoordinator:
    """Test Coordinator functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        return create_mock_llm_client(response_delay_ms=1, failure_rate=0.0)

    @pytest.fixture
    def coordinator(self, mock_llm_client, temp_dir):
        """Create coordinator with mocks."""
        return Coordinator(llm_client=mock_llm_client, sessions_dir=temp_dir)

    def test_coordinator_initialization(self, coordinator):
        """Test coordinator initialization."""
        assert coordinator.task_planner is not None
        assert coordinator.task_executor is not None
        assert coordinator.total_sessions == 0
        assert coordinator.successful_sessions == 0
        assert coordinator.current_session is None

    def test_stats_tracking(self, coordinator):
        """Test statistics tracking."""
        initial_stats = coordinator.get_stats()

        assert initial_stats["total_sessions"] == 0
        assert initial_stats["successful_sessions"] == 0
        assert initial_stats["success_rate"] == 0.0

    def test_stats_reset(self, coordinator):
        """Test statistics reset."""
        # Manually set some stats
        coordinator.total_sessions = 5
        coordinator.successful_sessions = 3

        coordinator.reset_stats()

        assert coordinator.total_sessions == 0
        assert coordinator.successful_sessions == 0

    def test_session_history_empty(self, coordinator):
        """Test session history when no sessions exist."""
        history = coordinator.get_session_history()
        assert isinstance(history, list)
        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_load_nonexistent_session(self, coordinator):
        """Test loading a session that doesn't exist."""
        result = await coordinator.load_session("nonexistent_session")
        assert result is None

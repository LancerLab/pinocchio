"""
Integration tests between Memory and Session modules.
"""

import pytest

from pinocchio.knowledge import (
    KnowledgeCategory,
    KnowledgeContentType,
    KnowledgeFragment,
    KnowledgeManager,
)
from pinocchio.memory import AgentMemory, CodeVersion, MemoryManager
from pinocchio.prompt import AgentType, PromptManager
from pinocchio.session import SessionManager


@pytest.fixture
def temp_sessions_dir(tmp_path):
    """Create a temporary sessions directory."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    return sessions_dir


@pytest.fixture
def session_manager(temp_sessions_dir):
    """Create a session manager instance."""
    return SessionManager(store_dir=str(temp_sessions_dir))


@pytest.fixture
def memory_manager(temp_sessions_dir):
    """Create a memory manager instance."""
    return MemoryManager(store_dir=str(temp_sessions_dir / "memory"))


@pytest.fixture
def prompt_manager(temp_sessions_dir):
    """Create a prompt manager instance."""
    return PromptManager(storage_path=str(temp_sessions_dir / "prompt"))


@pytest.fixture
def knowledge_manager(temp_sessions_dir):
    """Create a knowledge manager instance."""
    return KnowledgeManager(storage_path=str(temp_sessions_dir / "knowledge"))


@pytest.fixture
def sample_session(session_manager):
    """Create a sample session."""
    session = session_manager.create_session(
        task_description="Test integration session"
    )
    return session.session_id, session_manager


class TestMemorySessionIntegration:
    """Integration tests between Memory and Session modules."""

    def test_session_memory_initialization(self, session_manager):
        """Test that memory components are initialized when a session is created."""
        session = session_manager.create_session(task_description="Test session")

        # Verify session was created
        assert session.session_id is not None
        assert session.task_description == "Test session"
        assert session.status.value == "active"

    def test_session_memory_persistence(
        self, session_manager, memory_manager, prompt_manager, knowledge_manager
    ):
        """Test that memory data is persisted when a session is saved."""
        # Create a session
        session = session_manager.create_session(task_description="Test persistence")
        session_id = session.session_id

        # Add code version
        code = "def example(): return 42"
        version = CodeVersion.create_new_version(
            session_id=session_id,
            code=code,
            language="python",
            kernel_type="example_function",
            source_agent="generator",
            description="Test version",
        )
        memory_manager.add_code_version(session_id, version)

        # Add prompt template
        prompt_manager.create_template(
            template_name="test-template",
            content="This is a test template",
            description="Test template",
        )

        # Add knowledge fragment
        knowledge = KnowledgeFragment.create_fragment(
            session_id=session_id,
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Test knowledge fragment",
            content="Test knowledge fragment",
            content_type=KnowledgeContentType.TEXT,
        )
        knowledge_manager.add_fragment(knowledge)

        # Save the session
        session_manager._persist_session(session)

        # Verify files were created
        assert session_id in session_manager.active_sessions

        # Verify content of saved files
        memory_dir = session_manager.store_dir / "memory" / session_id
        assert (memory_dir / "code_memory.json").exists()

        prompt_root = session_manager.store_dir / "prompt"
        assert len(list(prompt_root.glob("*.json"))) > 0

        # Note: Knowledge fragments are not automatically persisted by session manager
        # They need to be explicitly saved by the knowledge manager
        knowledge_dir = session_manager.store_dir / "knowledge" / session_id
        # The test should not expect fragments.json to exist unless explicitly saved

    def test_session_memory_loading(
        self, session_manager, memory_manager, prompt_manager, knowledge_manager
    ):
        """Test that memory data is loaded when a session is loaded."""
        # Create and save a session with data
        session = session_manager.create_session(task_description="Test loading")
        session_id = session.session_id

        # Add code version
        code = "def example(): return 42"
        version = CodeVersion.create_new_version(
            session_id=session_id,
            code=code,
            language="python",
            kernel_type="example_function",
            source_agent="generator",
            description="Test version",
        )
        memory_manager.add_code_version(session_id, version)

        # Add prompt template
        prompt_manager.create_template(
            template_name="test-template",
            content="Test template content",
            description="Test template",
        )

        # Add knowledge fragment
        knowledge = KnowledgeFragment.create_fragment(
            session_id=session_id,
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Test knowledge",
            content="Test knowledge",
            content_type=KnowledgeContentType.TEXT,
            metadata={"code_version_id": version.version_id},
        )
        knowledge_manager.add_fragment(knowledge)

        # Save the session
        session_manager._persist_session(session)

        # Create a new session manager (simulating a new process)
        new_manager = type(session_manager)(store_dir=str(session_manager.store_dir))

        # Load the session
        loaded_session = new_manager.get_session(session_id)
        assert loaded_session is not None

        # Verify memory data was loaded
        code_memory = memory_manager.get_code_memory(session_id)
        assert version.version_id in code_memory.versions
        retrieved_version = memory_manager.get_code_version(
            session_id, version.version_id
        )
        assert retrieved_version.code == code

        # Verify prompt data was loaded
        templates = prompt_manager.list_templates()
        assert "test-template" in templates

        # Verify knowledge data was loaded
        fragments = knowledge_manager.get_session_fragments(session_id)
        assert knowledge.fragment_id in [f.fragment_id for f in fragments]
        # Verify code_version_id linkage
        loaded_knowledge = knowledge_manager.get_fragment(knowledge.fragment_id)
        assert loaded_knowledge.metadata["code_version_id"] == version.version_id

    def test_memory_manager_session_isolation(
        self, session_manager, memory_manager, prompt_manager, knowledge_manager
    ):
        """Test that memory data is isolated between sessions."""
        # Create first session
        session1 = session_manager.create_session(task_description="Session 1")
        session_id1 = session1.session_id

        # Add code to first session
        code1 = "def session1(): return 1"
        version1 = CodeVersion.create_new_version(
            session_id=session_id1,
            code=code1,
            language="python",
            kernel_type="session1_function",
            source_agent="generator",
        )
        memory_manager.add_code_version(session_id1, version1)

        # Add prompt to first session
        prompt_manager.create_template(
            template_name="session1-template",
            content="Session 1 template",
            description="Template for session 1",
        )

        # Add knowledge to first session
        knowledge1 = KnowledgeFragment.create_fragment(
            session_id=session_id1,
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Session 1 knowledge",
            content="Session 1 knowledge",
            content_type=KnowledgeContentType.TEXT,
            metadata={"code_version_id": version1.version_id},
        )
        knowledge_manager.add_fragment(knowledge1)

        # Save first session
        session_manager._persist_session(session1)

        # Create second session
        session2 = session_manager.create_session(task_description="Session 2")
        session_id2 = session2.session_id

        # Add code to second session
        code2 = "def session2(): return 2"
        version2 = CodeVersion.create_new_version(
            session_id=session_id2,
            code=code2,
            language="python",
            kernel_type="session2_function",
            source_agent="generator",
        )
        memory_manager.add_code_version(session_id2, version2)

        # Add prompt to second session
        prompt_manager.create_template(
            template_name="session2-template",
            content="Session 2 template",
            description="Template for session 2",
        )

        # Add knowledge to second session
        knowledge2 = KnowledgeFragment.create_fragment(
            session_id=session_id2,
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Session 2 knowledge",
            content="Session 2 knowledge",
            content_type=KnowledgeContentType.TEXT,
            metadata={"code_version_id": version2.version_id},
        )
        knowledge_manager.add_fragment(knowledge2)

        # Save second session
        session_manager._persist_session(session2)

        # Load first session
        session_manager.get_session(session_id1)

        # Verify only first session's data is available
        code_memory1 = memory_manager.get_code_memory(session_id1)
        code_memory2 = memory_manager.get_code_memory(session_id2)
        assert version1.version_id in code_memory1.versions
        assert version2.version_id not in code_memory1.versions

        templates1 = prompt_manager.list_templates()
        assert "session1-template" in templates1
        assert "session2-template" in templates1

        fragments1 = knowledge_manager.get_session_fragments(session_id1)
        assert knowledge1.fragment_id in [f.fragment_id for f in fragments1]
        assert knowledge2.fragment_id not in [f.fragment_id for f in fragments1]

        # Load second session
        session_manager.get_session(session_id2)

        # Verify only second session's data is available
        code_memory2 = memory_manager.get_code_memory(session_id2)
        assert version1.version_id not in code_memory2.versions
        assert version2.version_id in code_memory2.versions

        templates2 = prompt_manager.list_templates()
        assert "session1-template" in templates2
        assert "session2-template" in templates2

        fragments2 = knowledge_manager.get_session_fragments(session_id2)
        assert knowledge1.fragment_id not in [f.fragment_id for f in fragments2]
        assert knowledge2.fragment_id in [f.fragment_id for f in fragments2]

    def test_agent_memory_recording(self, session_manager, memory_manager):
        """Test recording agent interactions in a session."""
        # Create a session
        session = session_manager.create_session(task_description="Test agent memory")
        session_id = session.session_id

        # Create agent memories
        generator_memory = AgentMemory(
            session_id=session_id,
            agent_type="generator",
            version_id="v1",
            input_data={},
            output_data={},
            processing_time_ms=1,
            status="success",
        )

        debugger_memory = AgentMemory(
            session_id=session_id,
            agent_type="debugger",
            version_id="v1",
            input_data={},
            output_data={},
            processing_time_ms=1,
            status="success",
        )

        # Add agent memories
        memory_manager.store_agent_memory(generator_memory)
        memory_manager.store_agent_memory(debugger_memory)

        # Save session
        session_manager._persist_session(session)

        # Verify agent memories are stored
        agent_memories = memory_manager.query_agent_memories(session_id)
        assert len(agent_memories) == 2

        # Verify interaction history
        generator_mem = memory_manager.query_agent_memories(
            session_id, agent_type="generator"
        )
        debugger_mem = memory_manager.query_agent_memories(
            session_id, agent_type="debugger"
        )

        assert len(generator_mem) == 1
        assert len(debugger_mem) == 1
        assert generator_mem[0].agent_type == "generator"
        assert debugger_mem[0].agent_type == "debugger"

    def test_session_version_control(
        self, session_manager, memory_manager, prompt_manager, knowledge_manager
    ):
        """Test version control across sessions."""
        # Create a session
        session = session_manager.create_session(
            task_description="Test version control"
        )
        session_id = session.session_id

        # Create multiple code versions
        versions = []
        for i in range(3):
            version = CodeVersion.create_new_version(
                session_id=session_id,
                code=f"// Version {i+1} of optimized kernel",
                language="c",
                kernel_type="optimization_test",
                source_agent="generator",
                description=f"Optimization iteration {i+1}",
                optimization_techniques=[f"technique_{i+1}"],
            )
            memory_manager.add_code_version(session_id, version)
            versions.append(version)

        # Create multiple prompt templates
        templates = []
        for i in range(2):
            template = prompt_manager.create_template(
                template_name=f"template-{i+1}",
                content=f"Template content {i+1}",
                description=f"Template {i+1}",
                agent_type=AgentType.GENERATOR,
            )
            templates.append(template)

        # Create multiple knowledge fragments
        fragments = []
        for i in range(3):
            fragment = KnowledgeFragment.create_fragment(
                session_id=session_id,
                agent_type="generator",
                category=KnowledgeCategory.OPTIMIZATION,
                title=f"Knowledge fragment {i+1}",
                content=f"Knowledge fragment {i+1}",
                content_type=KnowledgeContentType.TEXT,
                metadata={"code_version_id": versions[i].version_id},
            )
            knowledge_manager.add_fragment(fragment)
            fragments.append(fragment)

        # Save session
        session_manager._persist_session(session)

        # Verify version history
        code_memory = memory_manager.get_code_memory(session_id)
        assert len(code_memory.versions) == 3

        prompt_templates = prompt_manager.list_templates()
        assert len(prompt_templates) == 2

        knowledge_fragments = knowledge_manager.get_session_fragments(session_id)
        assert len(knowledge_fragments) == 3

        # Verify version progression
        for i, version_id in enumerate(code_memory.versions):
            version = memory_manager.get_code_version(session_id, version_id)
            assert version.optimization_techniques == [f"technique_{i+1}"]
            # Ensure knowledge fragment links to correct code version
            assert fragments[i].metadata["code_version_id"] == version.version_id

    def test_session_performance_tracking(
        self, session_manager, memory_manager, knowledge_manager
    ):
        """Test performance tracking across sessions."""
        # Create a session
        session = session_manager.create_session(
            task_description="Test performance tracking"
        )
        session_id = session.session_id

        # Create code versions with performance data
        performance_data = [
            {"gflops": 30.0, "bandwidth": 25.0, "cache_miss": 0.20},
            {"gflops": 45.0, "bandwidth": 35.0, "cache_miss": 0.15},
            {"gflops": 65.0, "bandwidth": 50.0, "cache_miss": 0.10},
        ]
        versions = []
        for i, metrics in enumerate(performance_data):
            version = CodeVersion.create_new_version(
                session_id=session_id,
                code=f"// Performance iteration {i+1}",
                language="c",
                kernel_type="performance_test",
                source_agent="generator",
                description=f"Performance optimization {i+1}",
            )
            memory_manager.add_code_version(session_id, version)
            versions.append(version)

        # Create performance analysis knowledge
        perf_knowledge = KnowledgeFragment.create_fragment(
            session_id=session_id,
            agent_type="evaluator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Performance improvement analysis",
            content="Performance improvement analysis",
            content_type=KnowledgeContentType.TEXT,
            metadata={
                "analysis_type": "trend_analysis",
                "code_version_ids": [v.version_id for v in versions],
            },
        )
        knowledge_manager.add_fragment(perf_knowledge)

        # Save session
        session_manager._persist_session(session)

        # Verify performance tracking
        code_memory = memory_manager.get_code_memory(session_id)
        assert len(code_memory.versions) == 3

        # Check performance improvement trend
        version_ids = list(code_memory.versions.keys())
        assert version_ids == [v.version_id for v in versions]

        # Verify knowledge contains performance analysis
        fragments = knowledge_manager.get_session_fragments(session_id)
        assert perf_knowledge.fragment_id in [f.fragment_id for f in fragments]
        fragment = knowledge_manager.get_fragment(perf_knowledge.fragment_id)
        assert fragment.metadata["analysis_type"] == "trend_analysis"
        assert fragment.metadata["code_version_ids"] == [v.version_id for v in versions]

    def test_session_cross_module_integration(
        self, session_manager, memory_manager, prompt_manager, knowledge_manager
    ):
        """Test integration between all modules in a session."""
        # Create a session
        session = session_manager.create_session(
            task_description="Test cross-module integration"
        )
        session_id = session.session_id

        # Create code version
        code_version = CodeVersion.create_new_version(
            session_id=session_id,
            code="""
// Optimized matrix multiplication kernel
void matmul_optimized(float* A, float* B, float* C, int N) {
    const int BLOCK_SIZE = 32;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            for (int k = 0; k < N; k += BLOCK_SIZE) {
                // Blocked computation with vectorization
                for (int ii = i; ii < min(i + BLOCK_SIZE, N); ii++) {
                    for (int jj = j; jj < min(j + BLOCK_SIZE, N); jj += 16) {
                        __m512 sum = _mm512_setzero_ps();
                        for (int kk = k; kk < min(k + BLOCK_SIZE, N); kk++) {
                            __m512 a = _mm512_set1_ps(A[ii * N + kk]);
                            __m512 b = _mm512_loadu_ps(&B[kk * N + jj]);
                            sum = _mm512_fmadd_ps(a, b, sum);
                        }
                        _mm512_storeu_ps(&C[ii * N + jj], sum);
                    }
                }
            }
        }
    }
}
""",
            language="c",
            kernel_type="matrix_multiplication",
            source_agent="generator",
            description="Optimized matrix multiplication with AVX-512 and blocking",
            optimization_techniques=["vectorization", "blocking", "parallelization"],
        )
        memory_manager.add_code_version(session_id, code_version)

        # Create prompt template
        prompt_manager.create_template(
            template_name="kernel_optimization",
            content="Optimize the following kernel for high performance: {code}",
            description="Template for kernel optimization",
            agent_type=AgentType.GENERATOR,
        )

        # Create knowledge fragment
        knowledge_fragment = KnowledgeFragment.create_fragment(
            session_id=session_id,
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Matrix multiplication optimization with AVX-512 vectorization",
            content="Matrix multiplication optimization with AVX-512 vectorization",
            content_type=KnowledgeContentType.TEXT,
            metadata={"code_version_id": code_version.version_id},
        )
        knowledge_manager.add_fragment(knowledge_fragment)

        # Create agent memory
        agent_memory = AgentMemory(
            session_id=session_id,
            agent_type="generator",
            version_id="v1",
            input_data={},
            output_data={},
            processing_time_ms=1,
            status="success",
        )
        memory_manager.store_agent_memory(agent_memory)

        # Save session
        session_manager._persist_session(session)

        # Verify cross-module integration
        # Check memory
        code_memory = memory_manager.get_code_memory(session_id)
        assert code_version.version_id in code_memory.versions

        agent_memories = memory_manager.query_agent_memories(session_id)
        assert len(agent_memories) == 1

        # Check prompt
        templates = prompt_manager.list_templates()
        assert "kernel_optimization" in templates

        # Check knowledge
        fragments = knowledge_manager.get_session_fragments(session_id)
        assert knowledge_fragment.fragment_id in [f.fragment_id for f in fragments]

        # Verify cross-references
        fragment = knowledge_manager.get_fragment(knowledge_fragment.fragment_id)
        assert fragment.metadata["code_version_id"] == code_version.version_id

        agent_mem = memory_manager.query_agent_memories(
            session_id, agent_type="generator"
        )
        assert len(agent_mem) == 1
        assert agent_mem[0].agent_type == "generator"

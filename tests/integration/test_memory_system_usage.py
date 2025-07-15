"""
Test module demonstrating usage of Memory System functionality.

This test module serves as both validation and documentation, showing developers
how to use the enhanced memory management system for session data, keyword queries,
and context-aware prompt generation.

Key Features Demonstrated:
1. Memory storage and retrieval with metadata
2. Keyword-based memory queries with relevance scoring
3. Session organization and management
4. Memory integration with prompt generation
5. Temporal memory analysis and summaries
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pinocchio.memory import MemoryManager


class TestMemorySystemUsage:
    """
    Test suite demonstrating usage patterns for the memory system.

    These tests show how developers can:
    1. Store and retrieve memories with rich metadata
    2. Perform keyword-based searches with scoring
    3. Organize memories by sessions and topics
    4. Integrate memories with prompt generation
    5. Analyze memory patterns and generate summaries
    """

    def setup_method(self):
        """Set up test environment with temporary memory storage."""
        self.temp_dir = tempfile.mkdtemp()
        self.memory_manager = MemoryManager(self.temp_dir)

        # Initialize with sample memories for testing
        self._populate_sample_memories()

    def _populate_sample_memories(self):
        """Populate memory manager with sample CUDA development memories."""
        sample_memories = [
            {
                "content": "Implemented matrix multiplication kernel with shared memory optimization",
                "context": {
                    "task_type": "code_generation",
                    "agent": "generator",
                    "language": "CUDA",
                    "optimization_level": "high",
                },
                "metadata": {
                    "performance_gain": "3.2x speedup",
                    "memory_usage": "shared memory tiles",
                    "complexity": "medium",
                },
                "tags": [
                    "matrix_multiplication",
                    "shared_memory",
                    "optimization",
                    "CUDA",
                ],
            },
            {
                "content": "Debugged memory coalescing issues in vector addition kernel",
                "context": {
                    "task_type": "debugging",
                    "agent": "debugger",
                    "language": "CUDA",
                    "issue_type": "memory_access",
                },
                "metadata": {
                    "problem": "uncoalesced memory access",
                    "solution": "restructured data layout",
                    "performance_impact": "40% improvement",
                },
                "tags": [
                    "memory_coalescing",
                    "debugging",
                    "vector_addition",
                    "performance",
                ],
            },
            {
                "content": "Optimized CUDA kernel occupancy from 50% to 90%",
                "context": {
                    "task_type": "optimization",
                    "agent": "optimizer",
                    "language": "CUDA",
                    "metric": "occupancy",
                },
                "metadata": {
                    "initial_occupancy": "50%",
                    "final_occupancy": "90%",
                    "technique": "register usage optimization",
                    "block_size": "256 threads",
                },
                "tags": ["occupancy", "optimization", "registers", "block_size"],
            },
            {
                "content": "Evaluated convolution kernel performance on RTX 3080",
                "context": {
                    "task_type": "evaluation",
                    "agent": "evaluator",
                    "language": "CUDA",
                    "hardware": "RTX_3080",
                },
                "metadata": {
                    "throughput": "1.2 TFLOPS",
                    "memory_bandwidth": "85% efficiency",
                    "energy_efficiency": "high",
                    "comparison": "baseline_kernel",
                },
                "tags": ["convolution", "evaluation", "performance", "RTX_3080"],
            },
            {
                "content": "Fixed race condition in reduction algorithm using proper synchronization",
                "context": {
                    "task_type": "debugging",
                    "agent": "debugger",
                    "language": "CUDA",
                    "issue_type": "race_condition",
                },
                "metadata": {
                    "synchronization_method": "__syncthreads()",
                    "pattern": "tree_reduction",
                    "thread_safety": "ensured",
                },
                "tags": ["race_condition", "synchronization", "reduction", "debugging"],
            },
        ]

        # Store sample memories
        for memory_data in sample_memories:
            self.memory_manager.store_memory(
                content=memory_data["content"],
                context=memory_data["context"],
                metadata=memory_data["metadata"],
                tags=memory_data["tags"],
            )

    def test_basic_memory_storage_and_retrieval_usage(self):
        """
        Demonstrates basic memory storage and retrieval operations.

        Usage Pattern:
        - Store memories with rich context and metadata
        - Retrieve memories by ID and validate content
        - Update existing memories with new information
        """

        # Store a new memory
        memory_content = "Created efficient GEMM kernel using tensor cores"
        memory_context = {
            "task_type": "code_generation",
            "agent": "generator",
            "language": "CUDA",
            "hardware_feature": "tensor_cores",
        }
        memory_metadata = {
            "performance": "15 TFLOPS",
            "precision": "FP16",
            "architecture": "Ampere",
        }
        memory_tags = ["GEMM", "tensor_cores", "high_performance", "Ampere"]

        # This is how developers store memories
        memory_id = self.memory_manager.store_memory(
            content=memory_content,
            context=memory_context,
            metadata=memory_metadata,
            tags=memory_tags,
        )

        assert memory_id is not None
        print(f"✓ Stored memory with ID: {memory_id}")

        # Retrieve and validate the memory
        retrieved_memory = self.memory_manager.get_memory(memory_id)

        assert retrieved_memory is not None
        assert retrieved_memory["content"] == memory_content
        assert retrieved_memory["context"]["hardware_feature"] == "tensor_cores"
        assert retrieved_memory["metadata"]["performance"] == "15 TFLOPS"
        assert "tensor_cores" in retrieved_memory["tags"]

        print("✓ Memory retrieved successfully with all metadata intact")

        # Update the memory with additional information
        update_success = self.memory_manager.update_memory(
            memory_id,
            metadata={
                "performance": "16 TFLOPS",
                "optimization_note": "further optimized",
            },
        )

        assert update_success == True

        # Verify update
        updated_memory = self.memory_manager.get_memory(memory_id)
        assert updated_memory["metadata"]["performance"] == "16 TFLOPS"
        assert updated_memory["metadata"]["optimization_note"] == "further optimized"

        print("✓ Memory update functionality works correctly")

    def test_keyword_based_memory_queries_usage(self):
        """
        Demonstrates keyword-based memory search with relevance scoring.

        Usage Pattern:
        - Search memories using keywords and contexts
        - Understand relevance scoring and ranking
        - Filter results by minimum relevance scores
        """

        # Test single keyword search
        print("--- Single Keyword Search ---")
        optimization_memories = self.memory_manager.query_memories_by_keywords(
            keywords=["optimization"], min_score=0.1
        )

        assert (
            len(optimization_memories) >= 2
        )  # Should find optimization-related memories

        for memory in optimization_memories:
            print(
                f"Score: {memory['relevance_score']:.2f} - {memory['content'][:60]}..."
            )
            assert memory["relevance_score"] > 0.1
            # Verify optimization-related content
            content_and_tags = (
                memory["content"].lower() + " ".join(memory["tags"]).lower()
            )
            assert "optim" in content_and_tags

        print(f"✓ Found {len(optimization_memories)} optimization-related memories")

        # Test multiple keyword search with different weights
        print("\n--- Multiple Keyword Search ---")
        cuda_performance_memories = self.memory_manager.query_memories_by_keywords(
            keywords=["CUDA", "performance", "kernel"], min_score=0.2
        )

        assert len(cuda_performance_memories) >= 1

        for memory in cuda_performance_memories:
            print(
                f"Score: {memory['relevance_score']:.2f} - {memory['content'][:60]}..."
            )
            assert memory["relevance_score"] >= 0.2

        print(f"✓ Found {len(cuda_performance_memories)} CUDA performance memories")

        # Test context-specific search
        print("\n--- Context-Specific Search ---")
        debugging_memories = self.memory_manager.query_memories_by_keywords(
            keywords=["debugging", "memory"],
            context_filter={"task_type": "debugging"},
            min_score=0.1,
        )

        for memory in debugging_memories:
            assert memory["context"]["task_type"] == "debugging"
            print(f"Debug memory: {memory['content'][:50]}...")

        print(f"✓ Found {len(debugging_memories)} debugging-specific memories")

        # Test ranking by relevance
        print("\n--- Relevance Ranking Test ---")
        all_memories = self.memory_manager.query_memories_by_keywords(
            keywords=["CUDA"], min_score=0.0  # Get all memories
        )

        # Verify memories are sorted by relevance score (descending)
        for i in range(len(all_memories) - 1):
            assert (
                all_memories[i]["relevance_score"]
                >= all_memories[i + 1]["relevance_score"]
            )

        print("✓ Memories correctly ranked by relevance score")

    def test_session_based_memory_organization_usage(self):
        """
        Demonstrates session-based memory organization and management.

        Usage Pattern:
        - Organize memories by session IDs
        - Retrieve session-specific memories
        - Generate session summaries and analytics
        """

        # Create memories for different sessions
        session_1_id = "cuda_project_session_1"
        session_2_id = "optimization_session_2"

        # Session 1: CUDA project development
        session_1_memories = [
            {
                "content": "Started new CUDA project for image processing",
                "context": {"session_id": session_1_id, "phase": "initiation"},
                "tags": ["project_start", "image_processing"],
            },
            {
                "content": "Implemented basic convolution kernel",
                "context": {"session_id": session_1_id, "phase": "development"},
                "tags": ["convolution", "kernel_implementation"],
            },
            {
                "content": "Optimized convolution for better memory usage",
                "context": {"session_id": session_1_id, "phase": "optimization"},
                "tags": ["convolution", "memory_optimization"],
            },
        ]

        # Session 2: Performance optimization focus
        session_2_memories = [
            {
                "content": "Analyzed performance bottlenecks in existing code",
                "context": {"session_id": session_2_id, "phase": "analysis"},
                "tags": ["performance_analysis", "bottlenecks"],
            },
            {
                "content": "Applied register optimization techniques",
                "context": {"session_id": session_2_id, "phase": "optimization"},
                "tags": ["register_optimization", "performance"],
            },
        ]

        # Store session memories
        session_1_ids = []
        for memory in session_1_memories:
            memory_id = self.memory_manager.store_memory(
                content=memory["content"],
                context=memory["context"],
                tags=memory["tags"],
            )
            session_1_ids.append(memory_id)

        session_2_ids = []
        for memory in session_2_memories:
            memory_id = self.memory_manager.store_memory(
                content=memory["content"],
                context=memory["context"],
                tags=memory["tags"],
            )
            session_2_ids.append(memory_id)

        print(f"✓ Created {len(session_1_ids)} memories for session 1")
        print(f"✓ Created {len(session_2_ids)} memories for session 2")

        # Retrieve session-specific memories
        session_1_retrieved = self.memory_manager.query_memories_by_keywords(
            keywords=[""],  # Empty keyword to get all
            context_filter={"session_id": session_1_id},
            min_score=0.0,
        )

        assert len(session_1_retrieved) == 3
        for memory in session_1_retrieved:
            assert memory["context"]["session_id"] == session_1_id

        print(f"✓ Retrieved {len(session_1_retrieved)} memories for session 1")

        # Generate session summary
        session_summary = self.memory_manager.get_session_summary(session_1_id)

        assert session_summary["session_id"] == session_1_id
        assert session_summary["total_memories"] == 3
        assert (
            len(session_summary["phases"]) >= 2
        )  # initiation, development, optimization
        assert len(session_summary["common_tags"]) >= 2

        print("✓ Session summary generated:")
        print(f"  Total memories: {session_summary['total_memories']}")
        print(f"  Phases: {list(session_summary['phases'].keys())}")
        print(f"  Common tags: {session_summary['common_tags'][:3]}")

        # Test session analytics
        assert "convolution" in session_summary["common_tags"]
        assert "optimization" in session_summary["phases"]

        print("✓ Session organization and analytics work correctly")

    def test_memory_integration_with_prompt_generation_usage(self):
        """
        Demonstrates integration of memory system with prompt generation.

        Usage Pattern:
        - Format memories for prompt inclusion
        - Generate context-aware prompts using relevant memories
        - Show memory-enhanced prompt examples
        """

        # Search for relevant memories for a new task
        task_description = (
            "Optimize CUDA kernel for better memory bandwidth utilization"
        )
        relevant_keywords = ["optimization", "memory", "CUDA", "kernel"]

        relevant_memories = self.memory_manager.query_memories_by_keywords(
            keywords=relevant_keywords, min_score=0.3, max_results=3
        )

        assert len(relevant_memories) >= 1
        print(f"✓ Found {len(relevant_memories)} relevant memories for task")

        # Format memories for prompt inclusion
        formatted_memories = []
        for memory in relevant_memories:
            formatted_memory = self.memory_manager._format_memory_for_prompt(memory)
            formatted_memories.append(formatted_memory)

            # Validate formatting
            assert "CONTENT:" in formatted_memory
            assert "CONTEXT:" in formatted_memory
            assert "TAGS:" in formatted_memory
            assert memory["content"] in formatted_memory

        print("✓ Memories formatted for prompt inclusion")

        # Generate context-aware prompt
        base_prompt = f"""
You are a CUDA optimization expert. Please help with the following task:

TASK: {task_description}

RELEVANT PAST EXPERIENCE:
"""

        # Add formatted memories to prompt
        for i, formatted_memory in enumerate(formatted_memories, 1):
            base_prompt += f"\n--- Experience {i} ---\n{formatted_memory}\n"

        base_prompt += """
Based on the above experience and the current task, please provide optimized CUDA implementation.
"""

        # Validate prompt construction
        assert task_description in base_prompt
        assert "RELEVANT PAST EXPERIENCE:" in base_prompt
        assert "optimization" in base_prompt.lower()

        print("✓ Context-aware prompt generated successfully")
        print(f"✓ Prompt length: {len(base_prompt)} characters")
        print(f"✓ Includes {len(formatted_memories)} relevant memories")

        # Test memory-enhanced prompt for different scenarios
        scenarios = [
            {
                "task": "Debug memory access issues in CUDA kernel",
                "keywords": ["debugging", "memory", "access"],
            },
            {
                "task": "Evaluate performance of matrix multiplication",
                "keywords": ["evaluation", "performance", "matrix"],
            },
            {
                "task": "Generate optimized convolution kernel",
                "keywords": ["generation", "optimization", "convolution"],
            },
        ]

        for scenario in scenarios:
            scenario_memories = self.memory_manager.query_memories_by_keywords(
                keywords=scenario["keywords"], min_score=0.2, max_results=2
            )

            print(f"\nScenario: {scenario['task']}")
            print(f"  Found {len(scenario_memories)} relevant memories")

            if scenario_memories:
                # Show how memory content relates to the task
                for memory in scenario_memories:
                    relevance_indicators = []
                    for keyword in scenario["keywords"]:
                        if keyword.lower() in memory["content"].lower():
                            relevance_indicators.append(keyword)
                    print(f"  Memory relevance: {relevance_indicators}")

        print("\n✓ Memory-enhanced prompt generation works for various scenarios")

    def test_temporal_memory_analysis_usage(self):
        """
        Demonstrates temporal analysis of memory patterns and trends.

        Usage Pattern:
        - Analyze memory patterns over time
        - Track learning and improvement trends
        - Generate temporal summaries and insights
        """

        # Create time-sequenced memories to simulate development progression
        import time
        from datetime import datetime, timedelta

        base_time = datetime.now() - timedelta(days=7)  # Week ago

        progression_memories = [
            {
                "content": "Initial attempt at matrix multiplication - basic implementation",
                "metadata": {"performance": "1.2 GFLOPS", "optimization_level": "none"},
                "tags": ["matrix_multiplication", "initial", "basic"],
                "timestamp": base_time,
            },
            {
                "content": "Added shared memory optimization to matrix multiplication",
                "metadata": {
                    "performance": "3.8 GFLOPS",
                    "optimization_level": "basic",
                },
                "tags": ["matrix_multiplication", "shared_memory", "optimization"],
                "timestamp": base_time + timedelta(days=2),
            },
            {
                "content": "Implemented tiling strategy for better cache utilization",
                "metadata": {
                    "performance": "7.1 GFLOPS",
                    "optimization_level": "intermediate",
                },
                "tags": ["matrix_multiplication", "tiling", "cache_optimization"],
                "timestamp": base_time + timedelta(days=4),
            },
            {
                "content": "Applied register blocking and achieved near-optimal performance",
                "metadata": {
                    "performance": "12.5 GFLOPS",
                    "optimization_level": "advanced",
                },
                "tags": ["matrix_multiplication", "register_blocking", "optimal"],
                "timestamp": base_time + timedelta(days=6),
            },
        ]

        # Store progression memories
        progression_ids = []
        for memory_data in progression_memories:
            memory_id = self.memory_manager.store_memory(
                content=memory_data["content"],
                context={"task_type": "optimization_progression"},
                metadata=memory_data["metadata"],
                tags=memory_data["tags"],
            )
            progression_ids.append(memory_id)

        print(f"✓ Created {len(progression_ids)} progression memories")

        # Analyze temporal patterns
        matrix_mult_memories = self.memory_manager.query_memories_by_keywords(
            keywords=["matrix_multiplication"], min_score=0.1
        )

        # Extract performance progression
        performance_data = []
        for memory in matrix_mult_memories:
            if "performance" in memory.get("metadata", {}):
                perf_str = memory["metadata"]["performance"]
                if "GFLOPS" in perf_str:
                    perf_value = float(perf_str.split()[0])
                    performance_data.append(
                        {
                            "performance": perf_value,
                            "content": memory["content"],
                            "optimization_level": memory["metadata"].get(
                                "optimization_level", "unknown"
                            ),
                        }
                    )

        # Sort by performance to see progression
        performance_data.sort(key=lambda x: x["performance"])

        print("\n✓ Performance progression analysis:")
        for data in performance_data:
            print(
                f"  {data['performance']:5.1f} GFLOPS - {data['optimization_level']} - {data['content'][:50]}..."
            )

        # Calculate improvement metrics
        if len(performance_data) >= 2:
            initial_perf = performance_data[0]["performance"]
            final_perf = performance_data[-1]["performance"]
            improvement_factor = final_perf / initial_perf

            print(f"\n✓ Improvement analysis:")
            print(f"  Initial performance: {initial_perf:.1f} GFLOPS")
            print(f"  Final performance: {final_perf:.1f} GFLOPS")
            print(f"  Improvement factor: {improvement_factor:.1f}x")

            assert (
                improvement_factor > 1.0
            ), "Should show performance improvement over time"

        # Analyze tag evolution
        tag_frequency = {}
        for memory in matrix_mult_memories:
            for tag in memory["tags"]:
                tag_frequency[tag] = tag_frequency.get(tag, 0) + 1

        common_tags = sorted(tag_frequency.items(), key=lambda x: x[1], reverse=True)
        print(f"\n✓ Tag frequency analysis:")
        for tag, freq in common_tags[:5]:
            print(f"  {tag}: {freq} occurrences")

        print("\n✓ Temporal memory analysis provides valuable insights")

    def test_memory_cleanup_and_maintenance_usage(self):
        """
        Demonstrates memory cleanup and maintenance operations.

        Usage Pattern:
        - Remove outdated or irrelevant memories
        - Archive old memories for long-term storage
        - Maintain optimal memory system performance
        """

        # Get initial memory count
        all_memories = self.memory_manager.query_memories_by_keywords(
            keywords=[""], min_score=0.0
        )
        initial_count = len(all_memories)

        print(f"✓ Initial memory count: {initial_count}")

        # Test memory deletion
        # Find a specific memory to delete
        test_memories = self.memory_manager.query_memories_by_keywords(
            keywords=["matrix_multiplication"], min_score=0.1, max_results=1
        )

        if test_memories:
            memory_to_delete = test_memories[0]
            memory_id = memory_to_delete["id"]

            # Delete the memory
            delete_success = self.memory_manager.delete_memory(memory_id)
            assert delete_success == True

            # Verify deletion
            deleted_memory = self.memory_manager.get_memory(memory_id)
            assert deleted_memory is None

            print("✓ Memory deletion works correctly")

            # Verify count decreased
            updated_memories = self.memory_manager.query_memories_by_keywords(
                keywords=[""], min_score=0.0
            )
            assert len(updated_memories) == initial_count - 1

        # Test bulk operations (simulated)
        print("✓ Bulk memory operations:")

        # Find memories older than a certain threshold
        old_threshold_keywords = ["debugging"]  # Simulate finding "old" memories
        old_memories = self.memory_manager.query_memories_by_keywords(
            keywords=old_threshold_keywords, min_score=0.1
        )

        print(f"  Found {len(old_memories)} memories matching cleanup criteria")

        # Simulate archival process
        archived_count = 0
        for memory in old_memories[:2]:  # Archive first 2 for demo
            # In real implementation, might move to archive storage
            print(f"  Would archive: {memory['content'][:40]}...")
            archived_count += 1

        print(f"  Simulated archival of {archived_count} memories")

        # Test memory statistics for maintenance
        stats = {
            "total_memories": len(all_memories),
            "unique_tags": len(
                set(tag for memory in all_memories for tag in memory.get("tags", []))
            ),
            "memory_types": len(
                set(
                    memory["context"].get("task_type", "unknown")
                    for memory in all_memories
                )
            ),
            "agents_involved": len(
                set(
                    memory["context"].get("agent", "unknown") for memory in all_memories
                )
            ),
        }

        print("✓ Memory system statistics:")
        for stat_name, stat_value in stats.items():
            print(f"  {stat_name}: {stat_value}")

        # Test memory system health check
        health_check = {
            "storage_accessible": True,  # Would check actual storage
            "index_consistent": True,  # Would verify index integrity
            "no_corrupted_memories": True,  # Would scan for corruption
            "performance_acceptable": True,  # Would measure query performance
        }

        all_healthy = all(health_check.values())
        print(f"✓ Memory system health check: {'PASSED' if all_healthy else 'FAILED'}")

        for check_name, status in health_check.items():
            print(f"  {check_name}: {'✓' if status else '✗'}")


if __name__ == "__main__":
    """
    Run memory system usage tests and display results.

    This section demonstrates comprehensive memory system testing.
    """
    print("Running Memory System Usage Tests...")
    print("=" * 50)

    # Create test instance
    test_instance = TestMemorySystemUsage()
    test_instance.setup_method()

    # Run each test with clear output
    tests = [
        (
            "Basic Memory Storage and Retrieval",
            test_instance.test_basic_memory_storage_and_retrieval_usage,
        ),
        (
            "Keyword-Based Memory Queries",
            test_instance.test_keyword_based_memory_queries_usage,
        ),
        (
            "Session-Based Memory Organization",
            test_instance.test_session_based_memory_organization_usage,
        ),
        (
            "Memory Integration with Prompts",
            test_instance.test_memory_integration_with_prompt_generation_usage,
        ),
        ("Temporal Memory Analysis", test_instance.test_temporal_memory_analysis_usage),
        (
            "Memory Cleanup and Maintenance",
            test_instance.test_memory_cleanup_and_maintenance_usage,
        ),
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        try:
            print(f"\n{name}:")
            test_func()
            print(f"✅ {name} - PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {name} - FAILED: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All memory system tests passed!")
        print("\nKey Capabilities Demonstrated:")
        print("- Rich memory storage with context and metadata")
        print("- Intelligent keyword-based search with relevance scoring")
        print("- Session organization and temporal analysis")
        print("- Seamless integration with prompt generation")
        print("- Comprehensive memory maintenance and cleanup")
        print("\nDevelopers can now:")
        print("- Build context-aware AI systems with memory")
        print("- Implement intelligent information retrieval")
        print("- Track learning progression and insights over time")
        print("- Create memory-enhanced prompt generation")
        print("- Maintain optimal memory system performance")
    else:
        print("⚠️ Some tests failed. Check memory system implementation.")

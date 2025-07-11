"""
Tests for the knowledge models.
"""

from pinocchio.knowledge.models import KnowledgeItem, KnowledgeMemory


class TestKnowledgeModels:
    """Tests for the knowledge models."""

    def test_knowledge_item(self):
        """Test KnowledgeItem."""
        item = KnowledgeItem.create_new_version(
            knowledge_id="matrix-multiply",
            content={"description": "Matrix multiplication algorithm"},
            source="textbook",
            confidence=0.9,
        )

        assert item.knowledge_id == "matrix-multiply"
        assert item.content["description"] == "Matrix multiplication algorithm"
        assert item.source == "textbook"
        assert item.confidence == 0.9
        assert item.version_id is not None

    def test_knowledge_memory(self):
        """Test KnowledgeMemory operations."""
        memory = KnowledgeMemory()

        # Add an item
        item1 = KnowledgeItem.create_new_version(
            knowledge_id="matrix-multiply", content={"description": "Basic algorithm"}
        )

        version_id = memory.add_item(item1)
        assert version_id == item1.version_id

        # Get the item
        retrieved = memory.get_item("matrix-multiply")
        assert retrieved.content["description"] == "Basic algorithm"

        # Add another version
        item2 = KnowledgeItem.create_new_version(
            knowledge_id="matrix-multiply",
            content={"description": "Optimized algorithm"},
            source="research paper",
        )

        memory.add_item(item2)

        # List items
        items = memory.list_items()
        assert "matrix-multiply" in items

        # List item versions
        versions = memory.list_item_versions("matrix-multiply")
        assert len(versions) == 2

        # Search items
        results = memory.search_items({"description": "Optimized algorithm"})
        assert "matrix-multiply" in results

    def test_knowledge_memory_advanced_search(self):
        """Test advanced search in KnowledgeMemory."""
        memory = KnowledgeMemory()

        # Add multiple items
        item1 = KnowledgeItem.create_new_version(
            knowledge_id="matrix-multiply",
            content={
                "description": "Matrix multiplication algorithm",
                "complexity": "O(n^3)",
                "type": "numerical",
            },
        )
        memory.add_item(item1)

        item2 = KnowledgeItem.create_new_version(
            knowledge_id="quick-sort",
            content={
                "description": "Quick sort algorithm",
                "complexity": "O(n log n)",
                "type": "sorting",
            },
        )
        memory.add_item(item2)

        item3 = KnowledgeItem.create_new_version(
            knowledge_id="merge-sort",
            content={
                "description": "Merge sort algorithm",
                "complexity": "O(n log n)",
                "type": "sorting",
            },
        )
        memory.add_item(item3)

        # Search by complexity
        results = memory.search_items({"complexity": "O(n log n)"})
        assert len(results) == 2
        assert "quick-sort" in results
        assert "merge-sort" in results
        assert "matrix-multiply" not in results

        # Search by type
        results = memory.search_items({"type": "sorting"})
        assert len(results) == 2
        assert "quick-sort" in results
        assert "merge-sort" in results

        # Search with multiple criteria
        results = memory.search_items(
            {"type": "sorting", "description": "Quick sort algorithm"}
        )
        assert len(results) == 1
        assert "quick-sort" in results

        # Search with no matches
        results = memory.search_items({"type": "nonexistent"})
        assert len(results) == 0

        # Test the specific case where key exists but value doesn't match
        # This should trigger the break in the search_items method
        item5 = KnowledgeItem.create_new_version(
            knowledge_id="break-test",
            content={"key1": "value1", "key2": "value2", "key3": "value3"},
        )
        memory.add_item(item5)

        # This search should match the first key but fail on the second key
        # triggering the break statement in the loop
        results = memory.search_items(
            {
                "key1": "value1",
                "key2": "wrong_value",  # This will cause the match to fail
            }
        )
        assert "break-test" not in results

    def test_knowledge_memory_edge_cases(self):
        """Test edge cases for KnowledgeMemory."""
        memory = KnowledgeMemory()

        # Get nonexistent item
        item = memory.get_item("nonexistent")
        assert item is None

        # Get nonexistent version
        item = memory.get_item("nonexistent", "nonexistent-version")
        assert item is None

        # Set nonexistent item version
        result = memory.set_current_version("nonexistent", "nonexistent-version")
        assert result is False

        # List versions of nonexistent item
        versions = memory.list_item_versions("nonexistent")
        assert versions == {}

        # Test getting an item that exists but has no current version set
        # First create a situation where this can happen
        test_memory = KnowledgeMemory()
        test_item = KnowledgeItem.create_new_version(
            knowledge_id="test-item", content={"test": "value"}
        )
        # Manually add to knowledge_items but not to current_versions
        if "test-item" not in test_memory.knowledge_items:
            test_memory.knowledge_items["test-item"] = {}
        test_memory.knowledge_items["test-item"][test_item.version_id] = test_item
        # Note: we don't set current_versions["test-item"]

        # Now try to get the item
        retrieved = test_memory.get_item("test-item")
        assert retrieved is None

    def test_search_items_condition_coverage(self):
        """Test specifically to cover the condition in search_items method."""
        memory = KnowledgeMemory()

        # Add a test item
        item = KnowledgeItem.create_new_version(
            knowledge_id="test-item", content={"key1": "value1", "key2": "value2"}
        )
        memory.add_item(item)

        # Test case 1: key not in item.content
        # This should trigger the first part of the condition
        results = memory.search_items({"nonexistent_key": "any_value"})
        assert "test-item" not in results

        # Test case 2: key in item.content but value doesn't match
        # This should trigger the second part of the condition
        results = memory.search_items({"key1": "wrong_value"})
        assert "test-item" not in results

        # Positive test case for comparison
        results = memory.search_items({"key1": "value1"})
        assert "test-item" in results

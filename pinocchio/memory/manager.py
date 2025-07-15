"""
Memory manager for Pinocchio multi-agent system.

Manages session-isolated code versions, agent memories, performance metrics, and optimization history.
"""
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pinocchio.session.context import get_current_session

from ..utils.file_utils import ensure_directory, safe_read_json, safe_write_json
from ..utils.temp_utils import cleanup_temp_files, create_temp_file
from .models.agent_memories import (
    BaseAgentMemory,
    DebuggerMemory,
    EvaluatorMemory,
    GeneratorMemory,
)
from .models.code import CodeVersion
from .models.optimization import OptimizationHistory
from .models.performance import PerformanceHistory, PerformanceMetrics

logger = logging.getLogger(__name__)

AGENT_TYPES = ["generator", "debugger", "optimizer", "evaluator"]


class MemoryManager:
    """Memory manager for session-isolated agent memories, code versions, performance, and optimization."""

    def __init__(self, store_dir: str = "./memories"):
        """Initialize memory manager with storage directory."""
        self.store_dir = Path(store_dir)
        # Use utils function to ensure directory exists
        ensure_directory(self.store_dir)
        for agent_type in AGENT_TYPES:
            ensure_directory(self.store_dir / agent_type)
        self._session_cache: Dict[
            str, Dict[str, Any]
        ] = {}  # session_id -> {code, perf, opt, memories}
        logger.info(f"MemoryManager initialized with store_dir: {self.store_dir}")

    def _session_path(self, session_id: str) -> Path:
        p = self.store_dir / session_id
        # Use utils function to ensure directory exists
        ensure_directory(p)
        ensure_directory(p / "memories")
        logger.debug(f"Session path resolved: {p}")
        return p

    def _get_agent_memory_dir(self, agent_type: str) -> Path:
        p = self.store_dir / agent_type
        ensure_directory(p)
        return p

    def save_agent_memory(self, agent_type: str, memory_data: dict, session_id: str):
        """Save agent memory to file."""
        dir_path = self._get_agent_memory_dir(agent_type)
        file_path = dir_path / f"{session_id}_{agent_type}_memory.json"
        safe_write_json(memory_data, file_path)
        return file_path

    def store_agent_memory(self, memory: BaseAgentMemory) -> str:
        """Store an agent memory record, each agent's memory in its own directory."""
        session_path = self._session_path(memory.session_id)
        agent_mem_dir = session_path / "memories" / memory.agent_type
        agent_mem_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{memory.id}.json"
        fpath = agent_mem_dir / fname
        logger.info(f"Storing agent memory: {fname} at {fpath}")
        # Use utils function for safe JSON writing
        from ..utils.file_utils import safe_write_json

        success = safe_write_json(memory.model_dump(), fpath)
        if not success:
            logger.error(f"Failed to store agent memory {memory.id} at {fpath}")
            raise RuntimeError(f"Failed to store agent memory {memory.id}")
        # Cache
        self._session_cache.setdefault(memory.session_id, {}).setdefault(
            "memories", []
        ).append(memory)
        logger.debug(
            f"Agent memory cached for session {memory.session_id}, id {memory.id}"
        )
        return memory.id

    def log_generator_interaction(
        self,
        session_id: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        processing_time_ms: int,
        generation_strategy: str,
        optimization_techniques: List[str],
        hyperparameters: Dict[str, Any],
        knowledge_fragments: Dict[str, Any],
        status: str = "success",
        error_details: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str]:
        """Log generator interaction for memory manager."""
        print(f"Starting log_generator_interaction for session: {session_id}")
        # Create code version
        code = output_data.get("code", "")
        language = output_data.get("language", "")
        kernel_type = output_data.get("kernel_type", "")
        code_version = CodeVersion.create_new_version(
            session_id=session_id,
            code=code,
            language=language,
            kernel_type=kernel_type,
            source_agent="generator",
            description="Generator output",
            optimization_techniques=optimization_techniques,
            hyperparameters=hyperparameters,
        )
        print(f"Created code_version with ID: {code_version.version_id}")
        session_obj = get_current_session()
        self.add_code_version(session_id, code_version, session_obj=session_obj)
        print(
            f"After add_code_version, current versions: {list(self.get_code_memory(session_id).versions.keys())}"
        )
        # Create memory
        memory = GeneratorMemory(
            session_id=session_id,
            agent_type="generator",
            version_id=code_version.version_id,
            input_data=input_data,
            output_data=output_data,
            processing_time_ms=processing_time_ms,
            status=status,
            error_details=error_details,
            code_version_id=code_version.version_id,
            generation_strategy=generation_strategy,
            optimization_techniques=optimization_techniques,
            hyperparameters=hyperparameters,
            kernel_type=kernel_type,
            language=language,
            comments=output_data.get("comments", []),
            knowledge_fragments=knowledge_fragments,
        )
        memory_id = self.store_agent_memory(memory)
        print(
            f"About to return: code_version.version_id={code_version.version_id}, memory_id={memory_id}"
        )
        return memory_id, code_version.version_id

    def log_debugger_interaction(
        self,
        session_id: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        processing_time_ms: int,
        compilation_status: str,
        runtime_status: str,
        performance_metrics: Dict[str, Any],
        errors: List[str],
        warnings: List[str],
        execution_log: List[str],
        modified_code: Optional[str] = None,
        status: str = "success",
        error_details: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Optional[str]]:
        """Log a debugger agent interaction and code version (if code modified)."""
        # If code is modified, create new code version
        code_version_id = None
        if modified_code:
            language = output_data.get("language", "")
            kernel_type = output_data.get("kernel_type", "")
            code_version = CodeVersion.create_new_version(
                session_id=session_id,
                code=modified_code,
                language=language,
                kernel_type=kernel_type,
                source_agent="debugger",
                optimization_techniques=output_data.get("optimization_techniques", []),
                hyperparameters=output_data.get("hyperparameters", {}),
                description="Debugger modified code",
            )
            session_obj = get_current_session()
            self.add_code_version(session_id, code_version, session_obj=session_obj)
            code_version_id = code_version.version_id
        # Create memory
        memory = DebuggerMemory(
            session_id=session_id,
            agent_type="debugger",
            version_id=code_version_id or "",
            input_data=input_data,
            output_data=output_data,
            processing_time_ms=processing_time_ms,
            status=status,
            error_details=error_details,
            code_version_id=code_version_id,
            compilation_status=compilation_status,
            runtime_status=runtime_status,
            performance_metrics=performance_metrics,
            modified_code=modified_code,
            errors=errors,
            warnings=warnings,
            execution_log=execution_log,
            preserved_optimization_techniques=output_data.get(
                "optimization_techniques", []
            ),
            preserved_hyperparameters=output_data.get("hyperparameters", {}),
        )
        memory_id = self.store_agent_memory(memory)
        return memory_id, code_version_id

    def log_evaluator_interaction(
        self,
        session_id: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        processing_time_ms: int,
        current_optimization_techniques: List[str],
        current_hyperparameters: Dict[str, Any],
        optimization_suggestions: Dict[str, Any],
        performance_analysis: Dict[str, Any],
        next_iteration_prompt: str,
        status: str = "success",
        error_details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log an evaluator agent interaction."""
        session_obj = get_current_session()
        memory = EvaluatorMemory(
            session_id=session_id,
            agent_type="evaluator",
            version_id=output_data.get("version_id", ""),
            input_data=input_data,
            output_data=output_data,
            processing_time_ms=processing_time_ms,
            status=status,
            error_details=error_details,
            code_version_id=output_data.get("code_version_id", None),
            current_optimization_techniques=current_optimization_techniques,
            current_hyperparameters=current_hyperparameters,
            optimization_suggestions=optimization_suggestions,
            performance_analysis=performance_analysis,
            next_iteration_prompt=next_iteration_prompt,
            bottlenecks=output_data.get("bottlenecks", []),
            target_performance=output_data.get("target_performance", {}),
        )
        return self.store_agent_memory(memory)

    def add_code_version(
        self,
        session_id: str,
        code_version: CodeVersion,
        session_obj: Optional[Any] = None,
    ) -> str:
        """Add a new code version to the session (prefer session_obj if provided)."""
        logger.info(
            f"Adding code version: {code_version.version_id} for session {session_id}"
        )
        if session_obj is None:
            session_obj = get_current_session()
        if session_obj is not None:
            session_obj.add_code_version(code_version)
            logger.debug(
                f"Code version {code_version.version_id} added to session.code_versions."
            )
        else:
            logger.error("No session context available for adding code version.")
        return code_version.version_id

    def get_code_memory(
        self, session_id: str, session_obj: Optional[Any] = None
    ) -> Any:
        """Get the code memory for a session (prefer session_obj if provided)."""
        logger.info(f"Retrieving code memory for session {session_id}")
        if session_obj is None:
            session_obj = get_current_session()
        if session_obj is not None:
            return session_obj.code_memory
        if (
            session_id in self._session_cache
            and "code_memory" in self._session_cache[session_id]
        ):
            return self._session_cache[session_id]["code_memory"]  # type: ignore[no-any-return]
        session_path = self._session_path(session_id)
        fpath = session_path / "code_memory.json"
        if fpath.exists():
            with open(fpath, "r") as f:
                data = json.load(f)
            code_memory = CodeVersion.model_validate(data)  # Changed to CodeVersion
            logger.debug(f"Loading code memory from {fpath}")
        else:
            code_memory = CodeVersion(session_id=session_id)  # Changed to CodeVersion
            logger.debug(
                f"No code memory file found, creating new for session {session_id}"
            )
        self._session_cache.setdefault(session_id, {})["code_memory"] = code_memory
        return code_memory

    def get_current_code(
        self, session_id: str, session_obj: Optional[Any] = None
    ) -> Optional[str]:
        """Get the current code for a session (prefer session_obj if provided)."""
        logger.info(f"Getting current code for session {session_id}")
        if session_obj is None:
            session_obj = get_current_session()
        if session_obj is not None:
            return session_obj.get_latest_code()
        return None

    def get_code_version(
        self,
        session_id: str,
        version_id: Optional[str] = None,
        session_obj: Optional[Any] = None,
    ) -> Optional[CodeVersion]:
        """Get a specific code version or current version if not specified (prefer session_obj if provided)."""
        logger.info(f"Getting code version {version_id} for session {session_id}")
        if session_obj is None:
            session_obj = get_current_session()
        if session_obj is not None:
            if version_id is None:
                if session_obj.current_code_version_id:
                    return session_obj.code_versions.get(
                        session_obj.current_code_version_id
                    )
                return None
            return session_obj.code_versions.get(version_id)
        return None

    def add_performance_metrics(
        self,
        session_id: str,
        code_version_id: str,
        agent_type: str,
        execution_time_ms: float,
        memory_usage_mb: float,
        cache_miss_rate: Optional[float] = None,
        cpu_utilization: Optional[float] = None,
        throughput: Optional[float] = None,
        latency: Optional[float] = None,
        power_consumption: Optional[float] = None,
    ) -> str:
        """Add a performance metrics record."""
        logger.info(
            f"Adding performance metrics for session {session_id}, code_version {code_version_id}, agent {agent_type}"
        )
        session_path = self._session_path(session_id)
        fpath = session_path / "performance_history.json"
        perf_history = self.get_performance_history(session_id)
        metrics = PerformanceMetrics(
            execution_time_ms=execution_time_ms,
            memory_usage_mb=memory_usage_mb,
            cache_miss_rate=cache_miss_rate,
            cpu_utilization=cpu_utilization,
            throughput=throughput,
            latency=latency,
            power_consumption=power_consumption,
            session_id=session_id,
            code_version_id=code_version_id,
            agent_type=agent_type,
        )
        perf_history.add_metrics(metrics)
        # Use utils function for safe JSON writing
        success = safe_write_json(perf_history.model_dump(), fpath)
        if not success:
            raise RuntimeError(
                f"Failed to save performance history for session {session_id}"
            )
        self._session_cache.setdefault(session_id, {})[
            "performance_history"
        ] = perf_history
        return metrics.timestamp.isoformat()

    def get_performance_history(self, session_id: str) -> PerformanceHistory:
        """Get performance history for a session."""
        logger.info(f"Retrieving performance history for session {session_id}")
        if (
            session_id in self._session_cache
            and "performance_history" in self._session_cache[session_id]
        ):
            return self._session_cache[session_id]["performance_history"]  # type: ignore
        session_path = self._session_path(session_id)
        fpath = session_path / "performance_history.json"
        data = safe_read_json(fpath)
        if data is not None:
            perf_history = PerformanceHistory.model_validate(data)
        else:
            perf_history = PerformanceHistory(session_id=session_id)
        self._session_cache.setdefault(session_id, {})[
            "performance_history"
        ] = perf_history
        return perf_history

    def update_optimization_history(
        self,
        session_id: str,
        techniques: List[str],
        hyperparameters: Dict[str, Any],
        performance_impact: Dict[str, float],
    ) -> None:
        """Update optimization history with new iteration data."""
        logger.info(f"Updating optimization history for session {session_id}")
        session_path = self._session_path(session_id)
        fpath = session_path / "optimization_history.json"
        opt_history = self.get_optimization_history(session_id)
        opt_history.add_iteration(techniques, hyperparameters, performance_impact)
        # Use utils function for safe JSON writing
        success = safe_write_json(opt_history.model_dump(), fpath)
        if not success:
            raise RuntimeError(
                f"Failed to save optimization history for session {session_id}"
            )
        self._session_cache.setdefault(session_id, {})[
            "optimization_history"
        ] = opt_history

    def get_optimization_history(self, session_id: str) -> OptimizationHistory:
        """Get optimization history for a session."""
        logger.info(f"Retrieving optimization history for session {session_id}")
        if (
            session_id in self._session_cache
            and "optimization_history" in self._session_cache[session_id]
        ):
            return self._session_cache[session_id]["optimization_history"]  # type: ignore[no-any-return]
        session_path = self._session_path(session_id)
        fpath = session_path / "optimization_history.json"
        data = safe_read_json(fpath)
        if data is not None:
            opt_history = OptimizationHistory.model_validate(data)
        else:
            opt_history = OptimizationHistory(session_id=session_id)
        self._session_cache.setdefault(session_id, {})[
            "optimization_history"
        ] = opt_history
        return opt_history

    def get_optimization_summary(self, session_id: str) -> Dict[str, Any]:
        """Get optimization summary for a session."""
        logger.info(f"Getting optimization summary for session {session_id}")
        return self.get_optimization_history(session_id).get_optimization_summary()

    def query_agent_memories(
        self,
        session_id: str,
        agent_type: Optional[str] = None,
        filter_func: Optional[Callable[[BaseAgentMemory], bool]] = None,
        limit: int = 10,
    ) -> List[BaseAgentMemory]:
        """Query agent memories with optional filtering."""
        logger.info(
            f"Querying agent memories for session {session_id}, agent_type={agent_type}, limit={limit}"
        )
        session_path = self._session_path(session_id)
        memories_dir = session_path / "memories"
        results = []
        # Recursively traverse all agent subdirectories and memory files under memories
        for agent_dir in memories_dir.iterdir():
            if agent_dir.is_dir():
                for f in sorted(agent_dir.glob("*.json"), reverse=True):
                    data = safe_read_json(f)
                    if data is not None:
                        memory = BaseAgentMemory.model_validate(data)
                        if agent_type and memory.agent_type != agent_type:
                            continue
                        if filter_func and not filter_func(memory):
                            continue
                        results.append(memory)
                        if len(results) >= limit:
                            return results
        return results

    def get_agent_memories(self, session_id: str) -> List[BaseAgentMemory]:
        """Get all agent memories for a session."""
        logger.info(f"Getting all agent memories for session {session_id}")
        return self.query_agent_memories(session_id, limit=100)

    def export_logs(self, session_id: str, output_file: Optional[str] = None) -> str:
        """Export session logs to JSON file."""
        logger.info(f"Exporting logs for session {session_id} to {output_file}")
        session_path = self._session_path(session_id)
        export_path = output_file or str(session_path / f"export_{session_id}.json")
        export_data: Dict[str, Any] = {
            "code_memory": {},
            "performance_history": {},
            "optimization_history": {},
            "memories": [],
        }

        # Load code memory
        code_memory_file = session_path / "code_memory.json"
        code_memory_data = safe_read_json(code_memory_file)
        if code_memory_data is not None:
            export_data["code_memory"] = code_memory_data

        # Load performance history
        perf_history_file = session_path / "performance_history.json"
        perf_history_data = safe_read_json(perf_history_file)
        if perf_history_data is not None:
            export_data["performance_history"] = perf_history_data

        # Load optimization history
        opt_history_file = session_path / "optimization_history.json"
        opt_history_data = safe_read_json(opt_history_file)
        if opt_history_data is not None:
            export_data["optimization_history"] = opt_history_data

        # Load memories
        memories_dir = session_path / "memories"
        if memories_dir.exists():
            for memory_path in sorted(memories_dir.glob("*.json")):
                memory_data = safe_read_json(memory_path)
                if memory_data is not None:
                    export_data["memories"].append(memory_data)

        # Use utils function for safe JSON writing
        success = safe_write_json(export_data, export_path)
        if not success:
            raise RuntimeError(f"Failed to export logs to {export_path}")
        return export_path

    def query_memories_by_keywords(
        self,
        session_id: str,
        keywords: List[str],
        agent_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Query agent memories by keywords for prompt integration.

        Args:
            session_id: Session ID to query
            keywords: List of keywords to search for
            agent_type: Optional agent type filter
            limit: Maximum number of results

        Returns:
            List of relevant memory records formatted for prompt use
        """
        logger.info(
            f"Querying memories by keywords: {keywords} for session {session_id}"
        )

        # Get all memories for the session
        memories = self.query_agent_memories(session_id, agent_type, limit=100)

        # Score memories based on keyword relevance
        scored_memories = []
        for memory in memories:
            score = self._calculate_keyword_score(memory, keywords)
            if score > 0:
                scored_memories.append((memory, score))

        # Sort by score and take top results
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        top_memories = scored_memories[:limit]

        # Format for prompt use
        results = []
        for memory, score in top_memories:
            formatted_memory = self._format_memory_for_prompt(memory, score)
            results.append(formatted_memory)

        logger.info(f"Found {len(results)} relevant memories")
        return results

    def _calculate_keyword_score(
        self, memory: BaseAgentMemory, keywords: List[str]
    ) -> float:
        """Calculate relevance score for memory based on keywords."""
        score = 0.0

        # Convert keywords to lowercase for matching
        keywords_lower = [kw.lower() for kw in keywords]

        # Search in different fields with different weights
        search_fields = [
            (str(memory.input_data), 0.3),
            (str(memory.output_data), 0.4),
            (memory.agent_type, 0.1),
            (str(memory.error_details) if memory.error_details else "", 0.2),
        ]

        # Add agent-specific fields
        if hasattr(memory, "optimization_techniques"):
            search_fields.append((str(memory.optimization_techniques), 0.3))
        if hasattr(memory, "errors"):
            search_fields.append((str(memory.errors), 0.3))
        if hasattr(memory, "language"):
            search_fields.append((memory.language, 0.2))
        if hasattr(memory, "kernel_type"):
            search_fields.append((memory.kernel_type, 0.2))

        for text, weight in search_fields:
            text_lower = text.lower()
            for keyword in keywords_lower:
                if keyword in text_lower:
                    score += weight
                    # Bonus for exact matches
                    if f" {keyword} " in f" {text_lower} ":
                        score += weight * 0.5

        return score

    def _format_memory_for_prompt(
        self, memory: BaseAgentMemory, score: float
    ) -> Dict[str, Any]:
        """Format memory record for prompt manager use."""
        base_info = {
            "agent_type": memory.agent_type,
            "timestamp": memory.timestamp.isoformat(),
            "relevance_score": score,
            "session_id": memory.session_id,
            "memory_id": memory.id,
        }

        # Add agent-specific information
        if isinstance(memory, GeneratorMemory):
            base_info.update(
                {
                    "type": "generation",
                    "language": memory.language,
                    "kernel_type": memory.kernel_type,
                    "optimization_techniques": memory.optimization_techniques,
                    "hyperparameters": memory.hyperparameters,
                    "code_snippet": memory.output_data.get("code", "")[:200] + "..."
                    if len(memory.output_data.get("code", "")) > 200
                    else memory.output_data.get("code", ""),
                }
            )
        elif isinstance(memory, DebuggerMemory):
            base_info.update(
                {
                    "type": "debugging",
                    "compilation_status": memory.compilation_status,
                    "runtime_status": memory.runtime_status,
                    "errors": memory.errors[:3],  # Top 3 errors
                    "warnings": memory.warnings[:3],  # Top 3 warnings
                }
            )
        elif isinstance(memory, EvaluatorMemory):
            base_info.update(
                {
                    "type": "evaluation",
                    "optimization_suggestions": memory.optimization_suggestions,
                    "performance_analysis": memory.performance_analysis,
                    "bottlenecks": memory.bottlenecks,
                }
            )

        # Add input/output summaries
        if memory.input_data:
            base_info["input_summary"] = (
                str(memory.input_data)[:100] + "..."
                if len(str(memory.input_data)) > 100
                else str(memory.input_data)
            )
        if memory.output_data:
            base_info["output_summary"] = (
                str(memory.output_data)[:100] + "..."
                if len(str(memory.output_data)) > 100
                else str(memory.output_data)
            )

        return base_info

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a comprehensive summary of session memories."""
        logger.info(f"Getting session summary for {session_id}")

        memories = self.get_agent_memories(session_id)
        code_memory = self.get_code_memory(session_id)

        summary = {
            "session_id": session_id,
            "total_memories": len(memories),
            "agent_breakdown": {},
            "timeline": [],
            "code_versions": len(code_memory.versions),
            "current_code_version": code_memory.current_version_id,
        }

        # Count memories by agent type
        for memory in memories:
            agent_type = memory.agent_type
            if agent_type not in summary["agent_breakdown"]:
                summary["agent_breakdown"][agent_type] = 0
            summary["agent_breakdown"][agent_type] += 1

        # Create timeline of key events
        for memory in sorted(memories, key=lambda m: m.timestamp):
            event = {
                "timestamp": memory.timestamp.isoformat(),
                "agent_type": memory.agent_type,
                "status": memory.status,
                "memory_id": memory.id,
            }
            summary["timeline"].append(event)

        return summary

"""
Memory manager for Pinocchio multi-agent system.

Manages session-isolated code versions, agent memories, performance metrics, and optimization history.
"""
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .models.agent_memories import (
    BaseAgentMemory,
    DebuggerMemory,
    EvaluatorMemory,
    GeneratorMemory,
)
from .models.code import CodeMemory, CodeVersion
from .models.optimization import OptimizationHistory
from .models.performance import PerformanceHistory, PerformanceMetrics


class MemoryManager:
    """Memory manager for session-isolated agent memories, code versions, performance, and optimization."""

    def __init__(self, store_dir: str = "./memory_store"):
        """Initialize memory manager with storage directory."""
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._session_cache: Dict[
            str, Dict[str, Any]
        ] = {}  # session_id -> {code, perf, opt, memories}

    def _session_path(self, session_id: str) -> Path:
        p = self.store_dir / session_id
        p.mkdir(exist_ok=True)
        (p / "memories").mkdir(exist_ok=True)
        return p

    def store_agent_memory(self, memory: BaseAgentMemory) -> str:
        """Store an agent memory record."""
        session_path = self._session_path(memory.session_id)
        fname = f"{memory.agent_type}_{memory.id}.json"
        fpath = session_path / "memories" / fname
        with open(fpath, "w") as f:
            f.write(memory.model_dump_json())
        # Cache
        self._session_cache.setdefault(memory.session_id, {}).setdefault(
            "memories", []
        ).append(memory)
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
        """Log a generator agent interaction and code version."""
        # Create code version
        code = output_data.get("code", "")
        language = output_data.get("language", "")
        kernel_type = output_data.get("kernel_type", "")
        code_version = CodeVersion.create_new_version(
            code=code,
            source_agent="generator",
            description="Generator output",
            optimization_techniques=optimization_techniques,
            hyperparameters=hyperparameters,
        )
        self.add_code_version(session_id, code_version)
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
        return code_version.version_id, memory_id

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
            code_version = CodeVersion.create_new_version(
                code=modified_code,
                source_agent="debugger",
                optimization_techniques=output_data.get("optimization_techniques", []),
                hyperparameters=output_data.get("hyperparameters", {}),
                description="Debugger modified code",
            )
            self.add_code_version(session_id, code_version)
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

    def add_code_version(self, session_id: str, code_version: CodeVersion) -> str:
        """Add a new code version to the session."""
        session_path = self._session_path(session_id)
        fpath = session_path / "code_memory.json"
        code_memory = self.get_code_memory(session_id)
        code_memory.add_version(code_version)
        with open(fpath, "w") as f:
            f.write(code_memory.model_dump_json())
        self._session_cache.setdefault(session_id, {})["code_memory"] = code_memory
        return code_version.version_id

    def get_code_memory(self, session_id: str) -> CodeMemory:
        """Get the code memory for a session."""
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
            code_memory = CodeMemory.model_validate(data)
        else:
            code_memory = CodeMemory(session_id=session_id)
        self._session_cache.setdefault(session_id, {})["code_memory"] = code_memory
        return code_memory

    def get_current_code(self, session_id: str) -> Optional[str]:
        """Get the current code for a session."""
        code_memory = self.get_code_memory(session_id)
        current = code_memory.get_current_version()
        return current.code if current else None

    def get_code_version(
        self, session_id: str, version_id: Optional[str] = None
    ) -> Optional[CodeVersion]:
        """Get a specific code version or current version if not specified."""
        code_memory = self.get_code_memory(session_id)
        if version_id is None:
            return code_memory.get_current_version()
        return code_memory.versions.get(version_id)

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
        with open(fpath, "w") as f:
            f.write(perf_history.model_dump_json())
        self._session_cache.setdefault(session_id, {})[
            "performance_history"
        ] = perf_history
        return metrics.timestamp.isoformat()

    def get_performance_history(self, session_id: str) -> PerformanceHistory:
        """Get performance history for a session."""
        if (
            session_id in self._session_cache
            and "performance_history" in self._session_cache[session_id]
        ):
            return self._session_cache[session_id]["performance_history"]  # type: ignore
        session_path = self._session_path(session_id)
        fpath = session_path / "performance_history.json"
        if fpath.exists():
            with open(fpath, "r") as f:
                data = json.load(f)
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
        session_path = self._session_path(session_id)
        fpath = session_path / "optimization_history.json"
        opt_history = self.get_optimization_history(session_id)
        opt_history.add_iteration(techniques, hyperparameters, performance_impact)
        with open(fpath, "w") as f:
            f.write(opt_history.model_dump_json())
        self._session_cache.setdefault(session_id, {})[
            "optimization_history"
        ] = opt_history

    def get_optimization_history(self, session_id: str) -> OptimizationHistory:
        """Get optimization history for a session."""
        if (
            session_id in self._session_cache
            and "optimization_history" in self._session_cache[session_id]
        ):
            return self._session_cache[session_id]["optimization_history"]  # type: ignore[no-any-return]
        session_path = self._session_path(session_id)
        fpath = session_path / "optimization_history.json"
        if fpath.exists():
            with open(fpath, "r") as f:
                data = json.load(f)
            opt_history = OptimizationHistory.model_validate(data)
        else:
            opt_history = OptimizationHistory(session_id=session_id)
        self._session_cache.setdefault(session_id, {})[
            "optimization_history"
        ] = opt_history
        return opt_history

    def get_optimization_summary(self, session_id: str) -> Dict[str, Any]:
        """Get optimization summary for a session."""
        return self.get_optimization_history(session_id).get_optimization_summary()

    def query_agent_memories(
        self,
        session_id: str,
        agent_type: Optional[str] = None,
        filter_func: Optional[Callable[[BaseAgentMemory], bool]] = None,
        limit: int = 10,
    ) -> List[BaseAgentMemory]:
        """Query agent memories with optional filtering."""
        session_path = self._session_path(session_id)
        memories_dir = session_path / "memories"
        results = []
        for f in sorted(memories_dir.glob("*.json"), reverse=True):
            with open(f, "r") as file:
                data = json.load(file)
            memory = BaseAgentMemory.model_validate(data)
            if agent_type and memory.agent_type != agent_type:
                continue
            if filter_func and not filter_func(memory):
                continue
            results.append(memory)
            if len(results) >= limit:
                break
        return results

    def get_agent_memories(self, session_id: str) -> List[BaseAgentMemory]:
        """Get all agent memories for a session."""
        return self.query_agent_memories(session_id, limit=100)

    def export_logs(self, session_id: str, output_file: Optional[str] = None) -> str:
        """Export session logs to JSON file."""
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
        if code_memory_file.exists():
            with open(code_memory_file, "r") as f:
                export_data["code_memory"] = json.load(f)

        # Load performance history
        perf_history_file = session_path / "performance_history.json"
        if perf_history_file.exists():
            with open(perf_history_file, "r") as f:
                export_data["performance_history"] = json.load(f)

        # Load optimization history
        opt_history_file = session_path / "optimization_history.json"
        if opt_history_file.exists():
            with open(opt_history_file, "r") as f:
                export_data["optimization_history"] = json.load(f)

        # Load memories
        memories_dir = session_path / "memories"
        if memories_dir.exists():
            for memory_path in sorted(memories_dir.glob("*.json")):
                with open(memory_path, "r") as memory_file:
                    export_data["memories"].append(json.load(memory_file))

        with open(export_path, "w") as f:
            json.dump(export_data, f, indent=2)
        return export_path

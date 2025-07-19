#!/usr/bin/env python3
"""
Enhanced EAEPT Validation Logger
Comprehensive logging system for tracking AI work effects and validation during EAEPT workflow execution
"""

import json
import os
import sys
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import sqlite3
from contextlib import contextmanager
import hashlib
import psutil


class ValidationLevel(Enum):
    """Validation logging levels"""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    METRICS = "metrics"
    EFFECTIVENESS = "effectiveness"


class ValidationEvent(Enum):
    """Types of validation events to track"""

    PHASE_START = "phase_start"
    PHASE_END = "phase_end"
    PHASE_TRANSITION = "phase_transition"
    CONTEXT_OPTIMIZATION = "context_optimization"
    RAG_QUERY = "rag_query"
    USER_INTERACTION = "user_interaction"
    CODE_GENERATION = "code_generation"
    ERROR_OCCURRENCE = "error_occurrence"
    QUALITY_ASSESSMENT = "quality_assessment"
    PERFORMANCE_METRIC = "performance_metric"
    WORKFLOW_DECISION = "workflow_decision"
    ORCHESTRATION_ACTION = "orchestration_action"


@dataclass
class ValidationMetric:
    """Individual validation metric data structure"""

    timestamp: datetime
    phase: str
    event_type: ValidationEvent
    level: ValidationLevel
    metric_name: str
    metric_value: Union[float, int, str, bool]
    context: Dict[str, Any] = field(default_factory=dict)
    effectiveness_score: Optional[float] = None
    confidence_score: Optional[float] = None
    quality_score: Optional[float] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class PhaseValidationSummary:
    """Summary of validation metrics for a phase"""

    phase_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_events: int = 0
    error_count: int = 0
    warning_count: int = 0
    average_effectiveness: float = 0.0
    average_confidence: float = 0.0
    average_quality: float = 0.0
    context_optimizations: int = 0
    rag_queries: int = 0
    user_interactions: int = 0
    metrics: List[ValidationMetric] = field(default_factory=list)


class EAEPTValidationLogger:
    """Enhanced EAEPT Validation Logger for tracking AI work effects"""

    def __init__(
        self, project_root: Optional[str] = None, session_id: Optional[str] = None
    ):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.session_id = session_id or self._generate_session_id()
        self.user_id = os.getenv("USER", "unknown")

        # Setup logging directories
        self.log_dir = self.project_root / "logs" / "eaept_validation"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Database for structured logging
        self.db_path = self.log_dir / "validation_metrics.db"
        self._init_database()

        # File logging
        self.log_file = (
            self.log_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        self._setup_file_logging()

        # In-memory metrics for real-time analysis
        self.current_phase: Optional[str] = None
        self.phase_summaries: Dict[str, PhaseValidationSummary] = {}
        self.session_metrics: List[ValidationMetric] = []

        # Performance monitoring
        self.performance_baseline = self._capture_system_baseline()

        # Thread-safe logging
        self._lock = threading.Lock()

        self.logger.info(
            f"Enhanced EAEPT Validation Logger initialized for session {self.session_id}"
        )

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pid = os.getpid()
        return f"eaept_{timestamp}_{pid}"

    def _setup_file_logging(self):
        """Setup file-based logging"""
        self.logger = logging.getLogger(f"eaept_validation_{self.session_id}")
        self.logger.setLevel(logging.DEBUG)

        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)

        # Console handler for immediate feedback
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _init_database(self):
        """Initialize SQLite database for structured metrics"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS validation_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    user_id TEXT,
                    phase TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    level TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value TEXT NOT NULL,
                    context_json TEXT,
                    effectiveness_score REAL,
                    confidence_score REAL,
                    quality_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS phase_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    phase_name TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    total_events INTEGER DEFAULT 0,
                    error_count INTEGER DEFAULT 0,
                    warning_count INTEGER DEFAULT 0,
                    average_effectiveness REAL DEFAULT 0.0,
                    average_confidence REAL DEFAULT 0.0,
                    average_quality REAL DEFAULT 0.0,
                    context_optimizations INTEGER DEFAULT 0,
                    rag_queries INTEGER DEFAULT 0,
                    user_interactions INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_id ON validation_metrics(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_phase ON validation_metrics(phase)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON validation_metrics(timestamp)"
            )

            conn.commit()

    def _capture_system_baseline(self) -> Dict[str, Any]:
        """Capture system performance baseline"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage("/").percent,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.warning(f"Could not capture system baseline: {e}")
            return {"timestamp": datetime.now().isoformat()}

    @contextmanager
    def phase_context(self, phase_name: str):
        """Context manager for phase-specific logging"""
        self.start_phase(phase_name)
        try:
            yield self
        except Exception as e:
            self.log_error(f"Phase {phase_name} error", str(e))
            raise
        finally:
            self.end_phase(phase_name)

    def start_phase(self, phase_name: str):
        """Start logging for a new phase"""
        with self._lock:
            self.current_phase = phase_name

            if phase_name not in self.phase_summaries:
                self.phase_summaries[phase_name] = PhaseValidationSummary(
                    phase_name=phase_name, start_time=datetime.now()
                )

            self.log_metric(
                ValidationEvent.PHASE_START,
                ValidationLevel.INFO,
                "phase_start",
                phase_name,
                context={"system_state": self._capture_system_state()},
            )

            self.logger.info(f"🚀 Started phase: {phase_name}")

    def end_phase(self, phase_name: str):
        """End logging for current phase"""
        with self._lock:
            if phase_name in self.phase_summaries:
                summary = self.phase_summaries[phase_name]
                summary.end_time = datetime.now()

                # Calculate summary metrics
                phase_metrics = [
                    m for m in self.session_metrics if m.phase == phase_name
                ]
                if phase_metrics:
                    summary.total_events = len(phase_metrics)
                    summary.error_count = len(
                        [m for m in phase_metrics if m.level == ValidationLevel.ERROR]
                    )
                    summary.warning_count = len(
                        [m for m in phase_metrics if m.level == ValidationLevel.WARNING]
                    )

                    effectiveness_scores = [
                        m.effectiveness_score
                        for m in phase_metrics
                        if m.effectiveness_score is not None
                    ]
                    confidence_scores = [
                        m.confidence_score
                        for m in phase_metrics
                        if m.confidence_score is not None
                    ]
                    quality_scores = [
                        m.quality_score
                        for m in phase_metrics
                        if m.quality_score is not None
                    ]

                    summary.average_effectiveness = (
                        sum(effectiveness_scores) / len(effectiveness_scores)
                        if effectiveness_scores
                        else 0.0
                    )
                    summary.average_confidence = (
                        sum(confidence_scores) / len(confidence_scores)
                        if confidence_scores
                        else 0.0
                    )
                    summary.average_quality = (
                        sum(quality_scores) / len(quality_scores)
                        if quality_scores
                        else 0.0
                    )

                    summary.context_optimizations = len(
                        [
                            m
                            for m in phase_metrics
                            if m.event_type == ValidationEvent.CONTEXT_OPTIMIZATION
                        ]
                    )
                    summary.rag_queries = len(
                        [
                            m
                            for m in phase_metrics
                            if m.event_type == ValidationEvent.RAG_QUERY
                        ]
                    )
                    summary.user_interactions = len(
                        [
                            m
                            for m in phase_metrics
                            if m.event_type == ValidationEvent.USER_INTERACTION
                        ]
                    )

                # Save to database
                self._save_phase_summary(summary)

                self.log_metric(
                    ValidationEvent.PHASE_END,
                    ValidationLevel.INFO,
                    "phase_end",
                    phase_name,
                    context={
                        "duration_minutes": (
                            summary.end_time - summary.start_time
                        ).total_seconds()
                        / 60,
                        "total_events": summary.total_events,
                        "error_count": summary.error_count,
                        "average_effectiveness": summary.average_effectiveness,
                    },
                )

                self.logger.info(
                    f"✅ Completed phase: {phase_name} ({summary.total_events} events, {summary.error_count} errors)"
                )

    def log_metric(
        self,
        event_type: ValidationEvent,
        level: ValidationLevel,
        metric_name: str,
        metric_value: Union[float, int, str, bool],
        context: Optional[Dict[str, Any]] = None,
        effectiveness_score: Optional[float] = None,
        confidence_score: Optional[float] = None,
        quality_score: Optional[float] = None,
    ):
        """Log a validation metric"""
        with self._lock:
            metric = ValidationMetric(
                timestamp=datetime.now(),
                phase=self.current_phase or "unknown",
                event_type=event_type,
                level=level,
                metric_name=metric_name,
                metric_value=metric_value,
                context=context or {},
                effectiveness_score=effectiveness_score,
                confidence_score=confidence_score,
                quality_score=quality_score,
                user_id=self.user_id,
                session_id=self.session_id,
            )

            self.session_metrics.append(metric)
            self._save_metric_to_db(metric)

            # Log to file
            log_message = f"{event_type.value} | {metric_name}: {metric_value}"
            if effectiveness_score is not None:
                log_message += f" | effectiveness: {effectiveness_score:.2f}"
            if confidence_score is not None:
                log_message += f" | confidence: {confidence_score:.2f}"
            if quality_score is not None:
                log_message += f" | quality: {quality_score:.2f}"
            if context:
                log_message += f" | context: {json.dumps(context, default=str)}"

            getattr(self.logger, level.value.lower())(log_message)

    def log_phase_transition(
        self,
        from_phase: str,
        to_phase: str,
        confidence: float,
        auto_transition: bool = True,
    ):
        """Log phase transition with confidence metrics"""
        self.log_metric(
            ValidationEvent.PHASE_TRANSITION,
            ValidationLevel.INFO,
            "phase_transition",
            f"{from_phase} -> {to_phase}",
            context={
                "from_phase": from_phase,
                "to_phase": to_phase,
                "auto_transition": auto_transition,
                "transition_confidence": confidence,
            },
            confidence_score=confidence,
        )

    def log_context_optimization(
        self, strategy: str, tokens_before: int, tokens_after: int, effectiveness: float
    ):
        """Log context optimization events"""
        self.log_metric(
            ValidationEvent.CONTEXT_OPTIMIZATION,
            ValidationLevel.INFO,
            "context_optimization",
            strategy,
            context={
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "token_reduction": tokens_before - tokens_after,
                "reduction_percentage": (
                    ((tokens_before - tokens_after) / tokens_before) * 100
                    if tokens_before > 0
                    else 0
                ),
            },
            effectiveness_score=effectiveness,
        )

    def log_rag_query(
        self,
        query: str,
        results_count: int,
        relevance_score: float,
        response_time_ms: float,
    ):
        """Log RAG query performance"""
        self.log_metric(
            ValidationEvent.RAG_QUERY,
            ValidationLevel.INFO,
            "rag_query",
            query[:100] + "..." if len(query) > 100 else query,
            context={
                "results_count": results_count,
                "response_time_ms": response_time_ms,
                "query_length": len(query),
            },
            effectiveness_score=relevance_score,
            quality_score=(
                min(1.0, 1000 / response_time_ms) if response_time_ms > 0 else 1.0
            ),  # Performance score
        )

    def log_user_interaction(self, interaction_type: str, response_quality: float):
        """Log user interaction quality"""
        self.log_metric(
            ValidationEvent.USER_INTERACTION,
            ValidationLevel.INFO,
            "user_interaction",
            interaction_type,
            context={"interaction_type": interaction_type},
            quality_score=response_quality,
        )

    def log_code_generation(
        self,
        language: str,
        lines_generated: int,
        quality_score: float,
        test_coverage: Optional[float] = None,
    ):
        """Log code generation metrics"""
        context = {
            "language": language,
            "lines_generated": lines_generated,
            "estimated_complexity": (
                "low"
                if lines_generated < 50
                else "medium" if lines_generated < 200 else "high"
            ),
        }
        if test_coverage is not None:
            context["test_coverage"] = test_coverage

        self.log_metric(
            ValidationEvent.CODE_GENERATION,
            ValidationLevel.INFO,
            "code_generation",
            f"{lines_generated} lines of {language}",
            context=context,
            quality_score=quality_score,
            effectiveness_score=(quality_score + (test_coverage or 0.5)) / 2,
        )

    def log_error(
        self, error_type: str, error_message: str, recovery_attempted: bool = False
    ):
        """Log error occurrences"""
        self.log_metric(
            ValidationEvent.ERROR_OCCURRENCE,
            ValidationLevel.ERROR,
            "error",
            error_type,
            context={
                "error_message": error_message,
                "recovery_attempted": recovery_attempted,
                "error_hash": hashlib.md5(error_message.encode()).hexdigest()[:8],
            },
        )

    def log_performance_metric(
        self,
        metric_name: str,
        metric_value: float,
        target_value: Optional[float] = None,
    ):
        """Log performance metrics"""
        effectiveness = None
        if target_value is not None:
            effectiveness = (
                min(1.0, target_value / metric_value) if metric_value > 0 else 0.0
            )

        self.log_metric(
            ValidationEvent.PERFORMANCE_METRIC,
            ValidationLevel.METRICS,
            metric_name,
            metric_value,
            context={
                "target_value": target_value,
                "meets_target": metric_value <= target_value if target_value else None,
            },
            effectiveness_score=effectiveness,
        )

    def log_workflow_decision(self, decision: str, confidence: float, reasoning: str):
        """Log AI workflow decisions"""
        self.log_metric(
            ValidationEvent.WORKFLOW_DECISION,
            ValidationLevel.INFO,
            "workflow_decision",
            decision,
            context={"reasoning": reasoning, "decision_confidence": confidence},
            confidence_score=confidence,
        )

    def log_orchestration_action(self, action: str, result: str, effectiveness: float):
        """Log orchestration engine actions"""
        self.log_metric(
            ValidationEvent.ORCHESTRATION_ACTION,
            ValidationLevel.INFO,
            "orchestration_action",
            action,
            context={"action_result": result, "action_effectiveness": effectiveness},
            effectiveness_score=effectiveness,
        )

    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for context"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception:
            return {"timestamp": datetime.now().isoformat()}

    def _save_metric_to_db(self, metric: ValidationMetric):
        """Save metric to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO validation_metrics 
                    (timestamp, session_id, user_id, phase, event_type, level, metric_name, 
                     metric_value, context_json, effectiveness_score, confidence_score, quality_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        metric.timestamp.isoformat(),
                        metric.session_id,
                        metric.user_id,
                        metric.phase,
                        metric.event_type.value,
                        metric.level.value,
                        metric.metric_name,
                        json.dumps(metric.metric_value, default=str),
                        json.dumps(metric.context, default=str),
                        metric.effectiveness_score,
                        metric.confidence_score,
                        metric.quality_score,
                    ),
                )
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save metric to database: {e}")

    def _save_phase_summary(self, summary: PhaseValidationSummary):
        """Save phase summary to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO phase_summaries
                    (session_id, phase_name, start_time, end_time, total_events, error_count, 
                     warning_count, average_effectiveness, average_confidence, average_quality,
                     context_optimizations, rag_queries, user_interactions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        self.session_id,
                        summary.phase_name,
                        summary.start_time.isoformat(),
                        summary.end_time.isoformat() if summary.end_time else None,
                        summary.total_events,
                        summary.error_count,
                        summary.warning_count,
                        summary.average_effectiveness,
                        summary.average_confidence,
                        summary.average_quality,
                        summary.context_optimizations,
                        summary.rag_queries,
                        summary.user_interactions,
                    ),
                )
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save phase summary to database: {e}")

    def generate_effectiveness_report(
        self, output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive effectiveness report"""
        report = {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "generated_at": datetime.now().isoformat(),
            "total_duration_minutes": 0,
            "total_events": len(self.session_metrics),
            "phases": {},
            "overall_metrics": {},
            "recommendations": [],
        }

        # Calculate overall metrics
        if self.session_metrics:
            effectiveness_scores = [
                m.effectiveness_score
                for m in self.session_metrics
                if m.effectiveness_score is not None
            ]
            confidence_scores = [
                m.confidence_score
                for m in self.session_metrics
                if m.confidence_score is not None
            ]
            quality_scores = [
                m.quality_score
                for m in self.session_metrics
                if m.quality_score is not None
            ]

            report["overall_metrics"] = {
                "average_effectiveness": (
                    sum(effectiveness_scores) / len(effectiveness_scores)
                    if effectiveness_scores
                    else 0.0
                ),
                "average_confidence": (
                    sum(confidence_scores) / len(confidence_scores)
                    if confidence_scores
                    else 0.0
                ),
                "average_quality": (
                    sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
                ),
                "total_errors": len(
                    [
                        m
                        for m in self.session_metrics
                        if m.level == ValidationLevel.ERROR
                    ]
                ),
                "total_context_optimizations": len(
                    [
                        m
                        for m in self.session_metrics
                        if m.event_type == ValidationEvent.CONTEXT_OPTIMIZATION
                    ]
                ),
                "total_rag_queries": len(
                    [
                        m
                        for m in self.session_metrics
                        if m.event_type == ValidationEvent.RAG_QUERY
                    ]
                ),
                "total_user_interactions": len(
                    [
                        m
                        for m in self.session_metrics
                        if m.event_type == ValidationEvent.USER_INTERACTION
                    ]
                ),
            }

        # Generate phase reports
        for phase_name, summary in self.phase_summaries.items():
            duration = 0
            if summary.end_time:
                duration = (summary.end_time - summary.start_time).total_seconds() / 60
                report["total_duration_minutes"] += duration

            report["phases"][phase_name] = {
                "duration_minutes": duration,
                "total_events": summary.total_events,
                "error_count": summary.error_count,
                "warning_count": summary.warning_count,
                "average_effectiveness": summary.average_effectiveness,
                "average_confidence": summary.average_confidence,
                "average_quality": summary.average_quality,
                "context_optimizations": summary.context_optimizations,
                "rag_queries": summary.rag_queries,
                "user_interactions": summary.user_interactions,
                "status": "completed" if summary.end_time else "in_progress",
            }

        # Generate recommendations
        overall_effectiveness = report["overall_metrics"].get(
            "average_effectiveness", 0.0
        )
        if overall_effectiveness < 0.7:
            report["recommendations"].append(
                "Consider more frequent RAG queries for better context"
            )
        if report["overall_metrics"].get("total_errors", 0) > 5:
            report["recommendations"].append(
                "High error count detected - review error patterns"
            )
        if report["overall_metrics"].get("total_context_optimizations", 0) < 2:
            report["recommendations"].append(
                "More context optimizations may improve performance"
            )

        # Save report if output file specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Effectiveness report saved to {output_path}")

        return report

    def get_session_summary(self) -> Dict[str, Any]:
        """Get current session summary"""
        return {
            "session_id": self.session_id,
            "current_phase": self.current_phase,
            "total_events": len(self.session_metrics),
            "phases_completed": len(
                [s for s in self.phase_summaries.values() if s.end_time]
            ),
            "total_duration_minutes": sum(
                (s.end_time - s.start_time).total_seconds() / 60
                for s in self.phase_summaries.values()
                if s.end_time
            ),
            "phase_summaries": {
                name: {
                    "events": summary.total_events,
                    "errors": summary.error_count,
                    "effectiveness": summary.average_effectiveness,
                }
                for name, summary in self.phase_summaries.items()
            },
        }

    def cleanup(self):
        """Cleanup resources and generate final report"""
        if self.current_phase:
            self.end_phase(self.current_phase)

        # Generate final effectiveness report
        report_path = self.log_dir / f"effectiveness_report_{self.session_id}.json"
        self.generate_effectiveness_report(str(report_path))

        self.logger.info(
            f"🎯 Validation logging session completed. Report: {report_path}"
        )


# Convenience functions for easy integration
_logger_instance: Optional[EAEPTValidationLogger] = None


def init_validation_logger(
    project_root: Optional[str] = None, session_id: Optional[str] = None
) -> EAEPTValidationLogger:
    """Initialize global validation logger instance"""
    global _logger_instance
    _logger_instance = EAEPTValidationLogger(project_root, session_id)
    return _logger_instance


def get_validation_logger() -> Optional[EAEPTValidationLogger]:
    """Get current validation logger instance"""
    return _logger_instance


def log_phase_start(phase: str):
    """Convenience function to log phase start"""
    if _logger_instance:
        _logger_instance.start_phase(phase)


def log_phase_end(phase: str):
    """Convenience function to log phase end"""
    if _logger_instance:
        _logger_instance.end_phase(phase)


def log_effectiveness(
    event_type: ValidationEvent,
    metric_name: str,
    value: Any,
    effectiveness: float,
    **kwargs,
):
    """Convenience function to log effectiveness metrics"""
    if _logger_instance:
        _logger_instance.log_metric(
            event_type,
            ValidationLevel.EFFECTIVENESS,
            metric_name,
            value,
            effectiveness_score=effectiveness,
            **kwargs,
        )


if __name__ == "__main__":
    # Demo usage
    logger = EAEPTValidationLogger()

    with logger.phase_context("express"):
        logger.log_workflow_decision(
            "analyze_task", 0.85, "Task complexity assessment complete"
        )
        logger.log_rag_query("casino game mechanics", 5, 0.9, 150.0)
        logger.log_performance_metric("thinking_time_seconds", 45.0, 60.0)

    with logger.phase_context("code"):
        logger.log_code_generation("python", 150, 0.9, 0.85)
        logger.log_context_optimization("preserve_code", 8000, 6000, 0.8)

    # Generate effectiveness report
    report = logger.generate_effectiveness_report()
    print(json.dumps(report, indent=2, default=str))

    logger.cleanup()

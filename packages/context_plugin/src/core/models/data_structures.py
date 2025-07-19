"""
Enhanced Data Structures for Claude Context Management
Comprehensive dataclass models with validation and serialization
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum, auto
import json
import uuid
from abc import ABC, abstractmethod


class Priority(Enum):
    """Task priority levels"""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class SessionState(Enum):
    """Session state tracking"""

    ACTIVE = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    ARCHIVED = auto()


class CommandStatus(Enum):
    """Command execution status"""

    PENDING = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class BaseModel(ABC):
    """Base model with common functionality"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, list):
                result[key] = [self._serialize_item(item) for item in value]
            elif isinstance(value, dict):
                result[key] = {k: self._serialize_item(v) for k, v in value.items()}
            else:
                result[key] = value
        return result

    def _serialize_item(self, item: Any) -> Any:
        """Serialize individual items"""
        if isinstance(item, datetime):
            return item.isoformat()
        elif isinstance(item, Enum):
            return item.value
        elif hasattr(item, "to_dict"):
            return item.to_dict()
        else:
            return item

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create instance from dictionary"""
        # This is a simplified implementation
        # In practice, you'd need proper type conversion
        return cls(**data)


@dataclass
class TaskDefinition(BaseModel):
    """Enhanced task definition with metadata"""

    title: str
    description: str = ""
    priority: Priority = Priority.MEDIUM
    estimated_duration: Optional[timedelta] = None
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    completion_criteria: List[str] = field(default_factory=list)

    def add_dependency(self, task_id: str):
        """Add task dependency"""
        if task_id not in self.dependencies:
            self.dependencies.append(task_id)
            self.updated_at = datetime.now()

    def add_tag(self, tag: str):
        """Add tag to task"""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.now()

    def is_ready_to_start(self, completed_tasks: List[str]) -> bool:
        """Check if all dependencies are completed"""
        return all(dep in completed_tasks for dep in self.dependencies)


@dataclass
class ExecutionResult(BaseModel):
    """Command execution result with detailed information"""

    command: str
    status: CommandStatus
    duration: timedelta
    output: str = ""
    error: str = ""
    exit_code: int = 0
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if execution was successful"""
        return self.status == CommandStatus.SUCCESS and self.exit_code == 0

    def add_metric(self, name: str, value: float):
        """Add performance metric"""
        self.performance_metrics[name] = value
        self.updated_at = datetime.now()


@dataclass
class SessionSnapshot(BaseModel):
    """Complete session snapshot for persistence"""

    session_id: str
    state: SessionState
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration: Optional[timedelta] = None

    # Context information
    total_tokens: int = 0
    message_count: int = 0
    command_count: int = 0
    file_changes: int = 0

    # Task tracking
    tasks_completed: List[str] = field(default_factory=list)
    tasks_failed: List[str] = field(default_factory=list)

    # Performance metrics
    avg_response_time: float = 0.0
    error_rate: float = 0.0
    complexity_score: float = 0.0

    # Metadata
    session_type: str = "general"
    user_goals: List[str] = field(default_factory=list)
    achievements: List[str] = field(default_factory=list)

    def calculate_duration(self):
        """Calculate total session duration"""
        if self.end_time:
            self.total_duration = self.end_time - self.start_time
        else:
            self.total_duration = datetime.now() - self.start_time
        self.updated_at = datetime.now()

    def add_achievement(self, achievement: str):
        """Add achievement to session"""
        if achievement not in self.achievements:
            self.achievements.append(achievement)
            self.updated_at = datetime.now()


@dataclass
class ContextWindow(BaseModel):
    """Context window with sliding window analytics"""

    window_size: int  # in minutes
    max_entries: int = 1000
    entries: List[Dict[str, Any]] = field(default_factory=list)

    def add_entry(self, entry: Dict[str, Any]):
        """Add entry to context window"""
        entry["timestamp"] = datetime.now().isoformat()
        self.entries.append(entry)

        # Trim if necessary
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)

        # Remove old entries
        cutoff_time = datetime.now() - timedelta(minutes=self.window_size)
        self.entries = [
            e
            for e in self.entries
            if datetime.fromisoformat(e["timestamp"]) >= cutoff_time
        ]

        self.updated_at = datetime.now()

    def get_recent_entries(self, minutes: int = None) -> List[Dict[str, Any]]:
        """Get entries from recent time window"""
        if minutes is None:
            minutes = self.window_size

        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            e
            for e in self.entries
            if datetime.fromisoformat(e["timestamp"]) >= cutoff_time
        ]

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate metrics for current window"""
        recent_entries = self.get_recent_entries()

        if not recent_entries:
            return {}

        # Calculate various metrics
        entry_count = len(recent_entries)
        avg_interval = 0.0

        if entry_count > 1:
            timestamps = [
                datetime.fromisoformat(e["timestamp"]) for e in recent_entries
            ]
            intervals = [
                (timestamps[i] - timestamps[i - 1]).total_seconds()
                for i in range(1, len(timestamps))
            ]
            avg_interval = sum(intervals) / len(intervals)

        return {
            "entry_count": float(entry_count),
            "avg_interval_seconds": avg_interval,
            "entries_per_minute": entry_count / max(self.window_size, 1),
            "window_utilization": entry_count / self.max_entries,
        }


@dataclass
class PerformanceProfile(BaseModel):
    """Performance profile for operation tracking"""

    operation_type: str
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    current_metrics: Dict[str, float] = field(default_factory=dict)
    threshold_metrics: Dict[str, float] = field(default_factory=dict)

    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0

    last_execution: Optional[datetime] = None
    best_performance: Dict[str, float] = field(default_factory=dict)
    worst_performance: Dict[str, float] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate"""
        if self.execution_count == 0:
            return 0.0
        return self.failure_count / self.execution_count

    def record_execution(self, result: ExecutionResult):
        """Record execution result"""
        self.execution_count += 1
        self.last_execution = datetime.now()

        if result.success:
            self.success_count += 1
        else:
            self.failure_count += 1

        # Update current metrics
        for metric_name, value in result.performance_metrics.items():
            self.current_metrics[metric_name] = value

            # Update best/worst performance
            if (
                metric_name not in self.best_performance
                or value < self.best_performance[metric_name]
            ):
                self.best_performance[metric_name] = value

            if (
                metric_name not in self.worst_performance
                or value > self.worst_performance[metric_name]
            ):
                self.worst_performance[metric_name] = value

        self.updated_at = datetime.now()

    def is_performance_degraded(self) -> bool:
        """Check if performance has degraded"""
        if not self.baseline_metrics or not self.current_metrics:
            return False

        degradation_threshold = 1.5  # 50% degradation threshold

        for metric_name, baseline in self.baseline_metrics.items():
            current = self.current_metrics.get(metric_name, 0)
            if current > baseline * degradation_threshold:
                return True

        return False


@dataclass
class AnalysisReport(BaseModel):
    """Comprehensive analysis report"""

    report_type: str
    period_start: datetime
    period_end: datetime

    # Summary metrics
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0

    # Performance data
    avg_response_time: float = 0.0
    avg_token_usage: float = 0.0
    peak_memory_usage: float = 0.0

    # Trends
    performance_trend: str = "stable"  # improving, stable, degrading
    usage_trend: str = "stable"
    error_trend: str = "stable"

    # Insights
    key_insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)

    # Raw data references
    data_sources: List[str] = field(default_factory=list)

    def add_insight(self, insight: str):
        """Add key insight"""
        if insight not in self.key_insights:
            self.key_insights.append(insight)
            self.updated_at = datetime.now()

    def add_recommendation(self, recommendation: str):
        """Add recommendation"""
        if recommendation not in self.recommendations:
            self.recommendations.append(recommendation)
            self.updated_at = datetime.now()

    def calculate_summary_metrics(self):
        """Calculate summary metrics"""
        if self.total_operations > 0:
            self.success_rate = self.successful_operations / self.total_operations
            self.failure_rate = self.failed_operations / self.total_operations
        else:
            self.success_rate = 0.0
            self.failure_rate = 0.0

        self.updated_at = datetime.now()


@dataclass
class ConfigurationProfile(BaseModel):
    """Configuration profile for different scenarios"""

    profile_name: str
    description: str = ""

    # Performance settings
    max_token_limit: int = 200000
    burn_rate_threshold: float = 100.0
    response_time_threshold: float = 30.0

    # Analysis settings
    context_window_size: int = 30  # minutes
    sliding_window_size: int = 5  # minutes
    confidence_threshold: float = 0.75

    # Automation settings
    auto_compact_enabled: bool = True
    auto_digest_enabled: bool = True
    auto_optimization_enabled: bool = True

    # Notification settings
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    notification_channels: List[str] = field(default_factory=list)

    def apply_settings(self, engine_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply profile settings to engine configuration"""
        updated_config = engine_config.copy()

        updated_config.update(
            {
                "max_token_limit": self.max_token_limit,
                "burn_rate_threshold": self.burn_rate_threshold,
                "response_time_threshold": self.response_time_threshold,
                "context_window_size": self.context_window_size,
                "sliding_window_size": self.sliding_window_size,
                "confidence_threshold": self.confidence_threshold,
                "auto_compact_enabled": self.auto_compact_enabled,
                "auto_digest_enabled": self.auto_digest_enabled,
                "auto_optimization_enabled": self.auto_optimization_enabled,
            }
        )

        return updated_config

    def validate_settings(self) -> List[str]:
        """Validate configuration settings"""
        errors = []

        if self.max_token_limit <= 0:
            errors.append("max_token_limit must be positive")

        if self.burn_rate_threshold < 0:
            errors.append("burn_rate_threshold must be non-negative")

        if self.response_time_threshold <= 0:
            errors.append("response_time_threshold must be positive")

        if not (0 < self.confidence_threshold <= 1):
            errors.append("confidence_threshold must be between 0 and 1")

        return errors


# Factory functions for common configurations
def create_development_profile() -> ConfigurationProfile:
    """Create development-optimized configuration"""
    return ConfigurationProfile(
        profile_name="development",
        description="Optimized for development workflows",
        max_token_limit=150000,
        burn_rate_threshold=80.0,
        response_time_threshold=20.0,
        auto_compact_enabled=True,
        auto_digest_enabled=True,
        auto_optimization_enabled=True,
    )


def create_research_profile() -> ConfigurationProfile:
    """Create research-optimized configuration"""
    return ConfigurationProfile(
        profile_name="research",
        description="Optimized for research and analysis",
        max_token_limit=200000,
        burn_rate_threshold=120.0,
        response_time_threshold=45.0,
        context_window_size=60,
        auto_compact_enabled=False,
        auto_digest_enabled=True,
        auto_optimization_enabled=False,
    )


def create_production_profile() -> ConfigurationProfile:
    """Create production-optimized configuration"""
    return ConfigurationProfile(
        profile_name="production",
        description="Optimized for production reliability",
        max_token_limit=180000,
        burn_rate_threshold=60.0,
        response_time_threshold=15.0,
        confidence_threshold=0.9,
        auto_compact_enabled=True,
        auto_digest_enabled=True,
        auto_optimization_enabled=True,
    )

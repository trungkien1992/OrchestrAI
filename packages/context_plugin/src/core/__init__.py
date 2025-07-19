"""Core modules for Claude Context Management System"""

# Orchestration components
from .orchestration.orchestrator_engine import OrchestrationEngine
from .orchestration.decision_engine import DecisionEngine

# Analysis components
from .analysis.context_analysis_engine import (
    ContextAnalysisEngine,
    SessionType,
    ComplexityLevel,
    ContextAnalysis,
)
from .analysis.performance_metrics import (
    PerformanceMetricsEngine,
    MetricType,
    metrics_engine,
)
from .analysis.confidence_engine import (
    ConfidenceEngine,
    ConfidenceLevel,
    confidence_engine,
)

# Models
from .models.data_structures import (
    SessionSnapshot,
    TaskDefinition,
    ExecutionResult,
    ConfigurationProfile,
    create_development_profile,
    create_production_profile,
)

# Monitoring
from .monitoring.dashboard import RealTimeMonitor, DashboardMetrics, monitor

# Optimization
from .optimization.recommendations_engine import (
    RecommendationsEngine,
    OptimizationRecommendation,
    recommendations_engine,
)

# Integration
from .integration.rag_connector import (
    RAGConnector,
    ContextRAGManager,
    rag_connector,
    context_rag_manager,
)

# Persistence
from .persistence.session_manager import (
    SessionPersistenceManager,
    SessionState,
    session_manager,
)

# Configuration
from .rag.config import Config

__all__ = [
    # Orchestration
    "OrchestrationEngine",
    "DecisionEngine",
    # Analysis
    "ContextAnalysisEngine",
    "SessionType",
    "ComplexityLevel",
    "ContextAnalysis",
    "PerformanceMetricsEngine",
    "MetricType",
    "metrics_engine",
    "ConfidenceEngine",
    "ConfidenceLevel",
    "confidence_engine",
    # Models
    "SessionSnapshot",
    "TaskDefinition",
    "ExecutionResult",
    "ConfigurationProfile",
    "create_development_profile",
    "create_production_profile",
    # Monitoring
    "RealTimeMonitor",
    "DashboardMetrics",
    "monitor",
    # Optimization
    "RecommendationsEngine",
    "OptimizationRecommendation",
    "recommendations_engine",
    # Integration
    "RAGConnector",
    "ContextRAGManager",
    "rag_connector",
    "context_rag_manager",
    # Persistence
    "SessionPersistenceManager",
    "SessionState",
    "session_manager",
    # Configuration
    "Config",
]

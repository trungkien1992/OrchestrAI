"""
Decision Engine for Claude Orchestration Workflow

This module provides intelligent decision-making capabilities for automatic
orchestration of digest and compact commands based on context analysis,
timing optimization, and priority balancing, with enhanced adaptability
and performance.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum, auto
from collections import deque

from ..analysis.context_analysis_engine import (
    ContextAnalysisEngine,
    ContextAnalysis,
    SessionType,
    ComplexityLevel,
    TokenUsage,
)
from .simple_rag import ClaudeRAG
from ..config import ClaudeRAGConfig

logger = logging.getLogger(__name__)


class CommandType(Enum):
    """Types of orchestration commands with clearer priorities"""

    DIGEST_CREATE = auto()  # High priority for archival/complexity needs
    DIGEST_QUERY = auto()  # /archive-memory-related
    DIGEST_UPDATE = auto()  # complexity context preservation
    DIGEST_SHOW = auto()  # High priority for transient needs
    DIGEST_SNAPSHOT = auto()  # Forget, process without external changes
    COMPACT_DEEP = auto()  # Critical token management
    COMPACT_PRESERVE_CODE = auto()  # Moderate token reduction
    COMPACT_SUMMARIZE = auto()  # Lightweight cleanup
    COMPACT_FOCUS = auto()  # Error suppression
    HYBRID_TRANSITION = auto()  # Session state changes
    HYBRID_IMPLEMENTATION = auto()  # Complex workflows
    HYBRID_RESEARCH = auto()  # Research intensification
    NO_ACTION = auto()  # Fallback/low-confidence


class DecisionConfidence(Enum):
    """Confidence levels for decision making"""

    LOW = 0.25
    MEDIUM = 0.5
    HIGH = 0.75
    CRITICAL = 1.0


@dataclass
class OrchestrationDecision:
    """Decision result from the orchestration engine"""

    primary_command: CommandType
    secondary_commands: List[CommandType]
    confidence: DecisionConfidence
    reasoning: str
    estimated_execution_time: float  # minutes
    expected_benefits: List[str]
    risks: List[str]
    should_execute_immediately: bool
    optimal_delay: Optional[float] = None  # minutes to wait
    prerequisites: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_context: Optional[Dict[str, Any]] = None


@dataclass
class DecisionRule:
    """Rule for decision making with enhanced metadata"""

    name: str
    condition: Callable[["DecisionContext"], float]
    command: CommandType
    priority: int
    confidence: DecisionConfidence
    description: str
    depends_on: List[str] = field(default_factory=list)
    score_threshold: float = 0.5  # Minimum score needed to trigger
    is_critical: bool = False  # Flag for high-priority rules

    def __post_init__(self):
        # Validate threshold consistency
        if self.confidence.value < self.score_threshold:
            logger.warning(
                f"Rule {self.name} confidence {self.confidence.value} lower than score threshold {self.score_threshold}"
            )
            self.score_threshold = min(self.score_threshold, self.confidence.value)


@dataclass
class DecisionContext:
    """Enhanced context for decision making with additional metadata"""

    context_analysis: ContextAnalysis
    previous_decisions: List[OrchestrationDecision]
    system_state: Dict[str, Any]
    user_preferences: Dict[str, Any]
    session_history: List[Dict]
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_token_pressure(self) -> bool:
        """Quick access to critical token state"""
        return self.context_analysis.token_usage.threshold_reached

    @property
    def recent_decision_count(self) -> int:
        """Count recent decisions to detect rapid changes"""
        return len(
            [
                d
                for d in self.previous_decisions
                if (datetime.now() - d.metadata.get("timestamp", datetime.min)).seconds
                < 300
            ]
        )

    def get_session_phase(self) -> SessionType:
        """Get inferred phase from analysis"""
        return self.context_analysis.session_type


class DecisionEngine:
    """
    Intelligent decision engine for Claude orchestration workflow.

    Features:
    * Context-aware decision making
    * Configurable rule priorities and thresholds
    * Performance-based optimization
    * Transparent decision reasoning
    * Resilience to uncertain outcomes
    """

    def __init__(
        self, context_analyzer: ContextAnalysisEngine, config: ClaudeRAGConfig
    ):
        self.context_analyzer = context_analyzer
        self.config = config
        self.decision_history: List[OrchestrationDecision] = []
        self.last_decision_time: Optional[datetime] = None
        self.last_token_check: Optional[datetime] = None

        # Load all configuration parameters
        self._load_configuration()

        # Initialize core components
        self.decision_rules = self._initialize_decision_rules()
        self.performance_model = self._initialize_performance_model()
        # self.fallback_handler = FallbackHandler()  # Removed undefined handler

        logger.info(
            "Decision Engine initialized with %d rules", len(self.decision_rules)
        )

    def _load_configuration(self):
        """Load all configuration values from config object"""
        # Decision parameters
        decision_cfg = getattr(self.config, "decision", {})
        self.min_decision_interval = decision_cfg.get(
            "min_decision_interval", 5
        )  # minutes
        self.max_token_threshold = decision_cfg.get("max_token_threshold", 0.85)
        self.session_duration_trigger = decision_cfg.get("session_duration_trigger", 30)
        self.complexity_threshold = decision_cfg.get("complexity_threshold", 0.6)

        # Engine parameters
        engine_cfg = getattr(self.config, "engine", {})
        self.confidence_threshold = engine_cfg.get("confidence_threshold", 0.7)
        self.max_retry_attempts = engine_cfg.get("max_retry_attempts", 3)
        self.retry_delay = engine_cfg.get("retry_delay", 1)  # minutes

        # Performance tuning
        perf_tuning = engine_cfg.get("performance_tuning", {})
        self.aggressive_compaction = perf_tuning.get("aggressive_compaction", False)
        self.prioritize_memory = perf_tuning.get("prioritize_memory", False)

    def _initialize_performance_model(self) -> Dict[str, Any]:
        """Initialize a simple performance model for adaptability"""
        return {
            "success_rate": 0.8,  # Default success rate
            "latency_profile": {
                "fast": 0.5,  # Fast commands (Summarize, Show)
                "medium": 0.3,  # Medium commands (Compact)
                "slow": 0.2,  # Slow commands (Digest Create)
            },
            "error_rates": {
                CommandType.COMPACT_DEEP: 0.05,
                CommandType.DIGEST_CREATE: 0.03,
            },
            "last_update": datetime.now(),
        }

    async def make_orchestration_decision(
        self, session_data: Dict
    ) -> OrchestrationDecision:
        """
        Make an intelligent orchestration decision based on current context.

        Args:
            session_data: Current session data to analyze

        Returns:
            OrchestrationDecision with recommended actions and metadata
        """
        try:
            # Get fresh context analysis
            context = await self.context_analyzer.analyze_session_context(session_data)

            # Create enhanced decision context
            decision_context = DecisionContext(
                context_analysis=context,
                previous_decisions=self.decision_history[-10:],
                system_state=await self._get_system_state(session_data),
                user_preferences=await self._get_user_preferences(),
                session_history=session_data.get("conversation_history", []),
                performance_metrics=self.performance_model,
            )

            # Evaluate all decision rules
            rule_results = await self._evaluate_decision_rules(decision_context)

            if not rule_results:
                return self._create_fallback_decision("No applicable rules found")

            # Select the best decision
            decision = await self._select_best_decision(rule_results, decision_context)

            # Apply performance optimization
            decision = await self._optimize_for_performance(decision, decision_context)

            # Validate prerequisites
            if not self._check_prerequisites(decision, decision_context):
                return self._handle_violated_prerequisites(decision, decision_context)

            # Record and return the decision
            self._record_decision(decision, decision_context)
            return decision

        except Exception as e:
            logger.exception(f"Decision making failed: {e}")
            return self._handle_exception(e, session_data)

    async def _evaluate_decision_rules(
        self, context: DecisionContext
    ) -> List[Tuple[DecisionRule, float]]:
        """Evaluate all decision rules and return scored results with metadata"""
        rule_results = []

        # First evaluate boolean rules (those with threshold tests)
        for rule in self._get_boolean_rules():
            score = await self._evaluate_rule_condition(rule, context)
            if score > rule.score_threshold:
                rule_results.append((rule, score))

        # Next evaluate probabilistic rules
        for rule in self._get_probabilistic_rules():
            score = await self._evaluate_rule_condition(rule, context)
            if score > 0 and self._are_dependencies_met(rule, context):
                adjusted_score = score * self._get_adjusted_rule_weight(rule, context)
                rule_results.append((rule, adjusted_score))

        # Sort by priority first, then score
        rule_results.sort(
            key=lambda x: (
                x[0].priority if x[0].is_critical else x[0].priority - 10,
                x[1],
            ),
            reverse=True,
        )

        logger.debug(
            f"Evaluated {len(rule_results)} active rules out of {len(self.decision_rules)} total"
        )
        return rule_results

    def _get_boolean_rules(self) -> List[DecisionRule]:
        """Return all rules that implement threshold tests"""
        return [
            rule for rule in self.decision_rules if "threshold" in rule.name.lower()
        ]

    def _get_probabilistic_rules(self) -> List[DecisionRule]:
        """Return all rules that implement graded evaluations"""
        return [
            rule
            for rule in self.decision_rules
            if rule.name not in [r.name for r in self._get_boolean_rules()]
        ]

    def _are_dependencies_met(
        self, rule: DecisionRule, context: DecisionContext
    ) -> bool:
        """Check if all required dependencies are satisfied"""
        for dep in rule.depends_on:
            if dep not in self.performance_model:
                return False
            if not self.performance_model[dep]:
                return False
        return True

    async def _evaluate_rule_condition(
        self, rule: DecisionRule, context: DecisionContext
    ) -> float:
        """Evaluate a specific rule and return confidence score"""
        try:
            # Call the rule's condition function with enhanced context
            raw_score = rule.condition(context)

            # Apply dynamic weighting based on current context
            if context.has_token_pressure and rule.command == CommandType.COMPACT_DEEP:
                raw_score += 0.2  # Boost compactions when under token pressure

            return min(1.0, max(0.0, raw_score))

        except Exception as e:
            logger.warning(f"Rule evaluation failed for {rule.name}: {e}")
            return 0.0

    def _get_adjusted_rule_weight(
        self, rule: DecisionRule, context: DecisionContext
    ) -> float:
        """Apply dynamic weighting to rule based on context"""
        # Boost critical rules
        if rule.is_critical:
            return 1.5

        # Reduce priority for rules that keep failing
        if (
            rule.name in self.performance_model
            and self.performance_model[rule.name].get("failure_count", 0) > 3
        ):
            return 0.7

        # Default to 1.0 for normal operation
        return 1.0

    async def _select_best_decision(
        self, rule_results: List[Tuple[DecisionRule, float]], context: DecisionContext
    ) -> OrchestrationDecision:
        """Select the best decision from evaluated rules with enhanced context"""
        if not rule_results:
            return self._create_fallback_decision("No rules triggered")

        # Get the highest scoring rule
        best_rule, best_score = rule_results[0]

        # Calculate comprehensive confidence level
        confidence = self._calculate_comprehensive_confidence(
            best_rule, best_score, context
        )

        # Generate secondary commands with awareness of previous decisions
        secondary_commands = await self._generate_secondary_commands(
            best_rule.command, context
        )

        # Estimate execution time with system performance considerations
        execution_time = await self._estimate_execution_time(
            best_rule.command, secondary_commands, context.system_state
        )

        # Generate benefits and risks with dynamic awareness
        benefits = await self._generate_expected_benefits(
            best_rule.command, context, context.system_state
        )
        risks = await self._generate_risks(
            best_rule.command, context, context.system_state
        )

        # Determine immediate execution based on comprehensive assessment
        should_execute = (
            confidence.value > self.confidence_threshold
            and self._should_execute_immediately(best_rule, best_score, context)
        )

        # Create and return the enhanced decision
        decision = OrchestrationDecision(
            primary_command=best_rule.command,
            secondary_commands=secondary_commands,
            confidence=confidence,
            reasoning=f"{best_rule.description} (score: {best_score:.2f}, contextual adjustment: {self._get_contextual_adjustment(best_rule, context)})",
            estimated_execution_time=execution_time,
            expected_benefits=benefits,
            risks=risks,
            should_execute_immediately=should_execute,
            optimal_delay=self._calculate_optimal_delay(context, best_rule.command),
            prerequisites=self._get_prerequisites(best_rule.command, context),
            metadata={
                "rule_name": best_rule.name,
                "score": best_score,
                "timestamp": datetime.now(),
                "rule_dependencies_met": all(
                    self._are_dependencies_met(dep_rule, context)
                    for dep_rule in self._get_rule_dependencies(best_rule)
                ),
            },
            session_context={
                "session_type": context.get_session_phase().value,
                "complexity_level": context.context_analysis.task_complexity.level.value,
                "session_duration": context.context_analysis.session_metrics.duration_minutes,
            },
        )

        return decision

    def _calculate_comprehensive_confidence(
        self, rule: DecisionRule, raw_score: float, context: DecisionContext
    ) -> DecisionConfidence:
        """Calculate confidence level with enhanced contextual factors"""
        # Base confidence from rule
        base_confidence = rule.confidence.value

        # Apply score multiplier (more focused on Sunday trends)
        confidence = base_confidence * raw_score

        # Reduce if token operation is not recommended
        if context.has_token_pressure and rule.command not in [
            CommandType.COMPACT_DEEP,
            CommandType.COMPACT_PRESERVE_CODE,
        ]:
            confidence *= 0.8

        # Reduce if too many recent decisions
        if context.recent_decision_count > 3:
            confidence *= 0.7

        # Cap confidence based on rule and context
        confidence = min(1.0, max(DecisionConfidence.LOW.value, confidence))

        # Convert to appropriate enum value
        if confidence < 0.4:
            return DecisionConfidence.LOW
        elif confidence < 0.65:
            return DecisionConfidence.MEDIUM
        elif confidence < 0.9:
            return DecisionConfidence.HIGH
        else:
            return DecisionConfidence.CRITICAL

    async def _optimize_for_performance(
        self, decision: OrchestrationDecision, context: DecisionContext
    ) -> OrchestrationDecision:
        """Optimize decision based on system performance and historical data"""
        # Check if we should delay execution
        if self.last_decision_time and not self._is_critical_decision(decision):
            time_since_last = (
                datetime.now() - self.last_decision_time
            ).total_seconds() / 60

            if time_since_last < self.min_decision_interval:
                delay = self.min_decision_interval - time_since_last
                decision.optimal_delay = delay
                decision.should_execute_immediately = False
                decision.reasoning += f" [delayed {delay:.1f}m for decision spacing]"

                # Reduce delay if system indicates urgency
                if context.system_state.get(
                    "token_pressure", False
                ) and decision.primary_command in [CommandType.COMPACT_DEEP]:
                    decision.optimal_delay = max(0, decision.optimal_delay - 2)
                    decision.reasoning += " [adjusting for token pressure]"

        # Adjust for system load
        if context.system_state.get("resource_pressure", False):
            if decision.primary_command in [CommandType.DIGEST_CREATE]:
                decision.optimal_delay = max(0, decision.optimal_delay or 3)
                decision.reasoning += " [system pressure adjustment]"

        # Apply performance-based adjustments
        perf_data = self._get_performance_data_for_command(decision.primary_command)
        if perf_data and perf_data.get("success_rate", 0) < 0.7:
            decision.confidence = max(
                DecisionConfidence.LOW,
                DecisionConfidence(decision.confidence.value * 0.8),
            )

        return decision

    async def _generate_secondary_commands(
        self, primary_command: CommandType, context: DecisionContext
    ) -> List[CommandType]:
        """Generate complementary secondary commands with context awareness"""
        # Define command combinations with contextual awareness
        combinations = {
            # Core combinations
            CommandType.DIGEST_CREATE: [CommandType.DIGEST_UPDATE],
            CommandType.COMPACT_DEEP: [CommandType.DIGEST_SHOW],
            # Situation-aware combinations
            CommandType.HYBRID_TRANSITION: [
                CommandType.COMPACT_SUMMARIZE,
                CommandType.DIGEST_SHOW,
            ],
            CommandType.HYBRID_IMPLEMENTATION: [
                CommandType.DIGEST_CREATE,
                CommandType.COMPACT_PRESERVE_CODE,
            ],
            CommandType.HYBRID_RESEARCH: [
                CommandType.COMPACT_SUMMARIZE,
                CommandType.DIGEST_QUERY,
            ],
            # Fallback behaviors
            CommandType.NO_ACTION: [],
        }

        # Override for specific context conditions
        if (
            context.has_token_pressure
            or context.context_analysis.token_usage.time_to_limit < 10
        ):
            if CommandType.DIGEST_QUERY in combinations.get(primary_command, []):
                combinations[primary_command] = [
                    cmd
                    for cmd in combinations[primary_command]
                    if cmd != CommandType.DIGEST_QUERY
                ]
                combinations[primary_command].append(CommandType.DIGEST_SNAPSHOT)

        return combinations.get(primary_command, [])

    async def _estimate_execution_time(
        self,
        primary_command: CommandType,
        secondary_commands: List[CommandType],
        system_state: Dict,
    ) -> float:
        """Estimate execution time with system awareness"""
        # Base time estimates for known commands
        time_estimates = {
            CommandType.DIGEST_CREATE: 2.0,
            CommandType.DIGEST_QUERY: 1.5,
            CommandType.DIGEST_UPDATE: 0.8,
            CommandType.DIGEST_SHOW: 0.5,
            CommandType.DIGEST_SNAPSHOT: 0.2,
            CommandType.COMPACT_DEEP: 1.0,
            CommandType.COMPACT_PRESERVE_CODE: 0.8,
            CommandType.COMPACT_SUMMARIZE: 0.5,
            CommandType.COMPACT_FOCUS: 0.3,
            CommandType.HYBRID_TRANSITION: 1.5,
            CommandType.HYBRID_IMPLEMENTATION: 2.0,
            CommandType.HYBRID_RESEARCH: 2.5,
            CommandType.NO_ACTION: 0.0,
        }

        # Get base time for primary command
        base_time = time_estimates.get(primary_command, 1.0)

        # Add time for secondary commands
        secondary_time = sum(time_estimates.get(cmd, 0.5) for cmd in secondary_commands)

        # Apply system state modifiers
        cpu_load = system_state.get("cpu_percent", 50.0) / 100.0
        memory_pressure = system_state.get("memory_percent", 50.0) / 100.0

        # Increase time estimates under system pressure
        pressure_multiplier = 1.0 + (cpu_load * 0.5) + (memory_pressure * 0.3)

        total_time = (base_time + secondary_time) * pressure_multiplier
        return total_time

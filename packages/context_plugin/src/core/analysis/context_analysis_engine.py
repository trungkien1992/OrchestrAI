"""
Optimized Context Analysis Engine for Claude Orchestration
This module provides real-time analysis of conversation context with:
- Dynamic token usage monitoring
- Session type classification
- Task complexity assessment
- Memory-efficient storage
"""

import asyncio
import logging
import time
import ticktoken  # Anthropic's official tokenizer
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto
from collections import deque
import math

logger = logging.getLogger(__name__)


class SessionType(Enum):
    """Types of development sessions"""

    CODING = auto()
    DEBUGGING = auto()
    ARCHITECTURE = auto()
    TESTING = auto()
    DOCUMENTATION = auto()
    RESEARCH = auto()
    MIXED = auto()


class ComplexityLevel(Enum):
    """Task complexity levels"""

    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


@dataclass
class TokenUsage:
    """Token usage statistics and predictions"""

    current_tokens: int
    estimated_total: int
    burn_rate: float  # tokens per minute
    time_to_limit: Optional[float] = None
    threshold_reached: bool = False
    prediction_confidence: float = 0.0
    token_history: List[Tuple[datetime, int]] = None
    last_update_duration: float = 0.0


@dataclass
class SessionMetrics:
    """Session-level metrics and patterns"""

    duration_minutes: float
    tool_usage_count: Dict[str, int]
    file_modifications: int
    command_execution_count: int
    error_rate: float
    context_switches: int
    relevance_score: float = 0.0


@dataclass
class TaskComplexity:
    """Task complexity assessment"""

    level: ComplexityLevel
    score: float  # 0-1 scalar
    factors: Dict[str, float]
    file_count: int
    language_variety: int
    dependency_depth: int
    estimated_time_hours: float


@dataclass
class ContextAnalysis:
    """Complete context analysis result"""

    session_type: SessionType
    token_usage: TokenUsage
    session_metrics: SessionMetrics
    task_complexity: TaskComplexity
    recommendations: List[str]
    confidence_score: float
    timestamp: datetime
    summary: Optional[str] = None


class ContextAnalysisEngine:
    """
    Optimized Plug-and-play context analysis engine with:
    - Dynamic burn rate calculation
    - Memory-efficient history management
    - Accurate token counting
    - Configuration-driven analysis
    - Parallel processing capabilities
    """

    # Configuration constants
    TOKEN_LIMIT = 200_000
    THRESHOLD_RATIO = 0.75
    CHAT_MAX_MESSAGES = 128  # Internal memory bound
    KEEP_LAST_MINUTES = 30
    BURN_RATE_WINDOW = 5  # Minutes for dynamic calculation
    MAX_SUMMARY_AGE = 60  # Minutes before regenerating summary

    # Weight map for complexity estimator
    COMPLEXITY_WEIGHTS = {
        "file_count": 0.35,
        "language_variety": 0.15,
        "dependency_depth": 0.25,
        "session_duration": 0.15,
        "tool_complexity": 0.10,
    }

    def __init__(self, rag_system: Optional[Any] = None, config: Optional[Dict] = None):
        # Initialize with configuration
        self.rag_system = rag_system
        self.config = config or {}

        # Load configuration parameters
        self.token_limit = config.get("token_limit", self.TOKEN_LIMIT)
        self.threshold_ratio = config.get("threshold_ratio", self.THRESHOLD_RATIO)
        self.max_messages = config.get("max_messages", self.CHAT_MAX_MESSAGES)
        self.keep_last_minutes = config.get("keep_last_minutes", self.KEEP_LAST_MINUTES)
        self.burn_rate_window = config.get("burn_rate_window", self.BURN_RATE_WINDOW)
        self.summarize_old_messages = config.get("summarize_old_messages", False)

        self.weights = config.get("complexity_weights", self.COMPLEXITY_WEIGHTS)

        # State tracking
        self.session_start_time = datetime.now()
        self.conversation_history: deque = deque(maxlen=self.max_messages)
        self.token_history = deque(maxlen=500)  # Keep more history for token tracking
        self.last_token_count = 0
        self.last_summary_time = datetime.min

        # Initialize tokenizer
        try:
            self.token_encoder = ticktoken.get_encoding("cl100k_base")
            logger.info("Using precise ticktoken tokenizer")
        except ImportError:
            self.token_encoder = None
            logger.warning(
                "Using approximate token counting (install ticktoken for accuracy)"
            )

        logger.info("Context Analysis Engine initialized")

    async def analyze_session_context(self, session_data: Dict) -> ContextAnalysis:
        """
        Perform comprehensive context analysis

        Args:
            session_data: Dictionary containing:
                conversation_history: List of message dictionaries
                tool_usage: List of tool usage instances
                file_changes: List of file modification records
                error_count: Integer count of errors

        Returns:
            ContextAnalysis object with complete analysis results
        """
        try:
            # Update internal state
            self._update_session_state(session_data)

            # Perform parallel analysis of different aspects
            token_usage, session_metrics, complexity_analysis, session_type = (
                await asyncio.gather(
                    self._analyze_token_usage(
                        session_data.get("conversation_history", [])
                    ),
                    self._analyze_session_metrics(session_data),
                    self._analyze_task_complexity(session_data),
                    self._classify_session_type(session_data.get("tool_usage", [])),
                )
            )

            # Generate recommendations
            recommendations = self._generate_recommendations(
                token_usage, session_metrics, complexity_analysis, session_type
            )

            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                token_usage, session_metrics, complexity_analysis
            )

            # Generate summary if needed
            summary = None
            if self.summarize_old_messages and self._should_generate_summary():
                summary = await self._generate_conversation_summary(
                    self.conversation_history
                )

            # Create and return analysis
            analysis = ContextAnalysis(
                session_type=session_type,
                token_usage=token_usage,
                session_metrics=session_metrics,
                task_complexity=complexity_analysis,
                recommendations=recommendations,
                confidence_score=confidence_score,
                timestamp=datetime.now(),
                summary=summary,
            )

            logger.debug(
                f"Context analysis completed - Type: {session_type.value} "
                f"Complexity: {complexity_analysis.level.value} "
                f"Tokens: {token_usage.current_tokens}/{self.token_limit}"
            )

            return analysis

        except Exception as e:
            logger.error(f"Context analysis failed: {e}")
            fallback = await self._create_fallback_analysis(session_data)
            logger.warning(f"Returning fallback analysis due to error")
            return fallback

    def _update_session_state(self, session_data: Dict):
        """Update internal session state with new data"""
        new_history = session_data.get("conversation_history", [])

        # Add new messages with timestamps
        for msg in new_history:
            if not msg.get("content"):
                continue

            # Store message with timestamp
            timestamped_msg = msg.copy()
            timestamped_msg["analysis_timestamp"] = datetime.now()

            self.conversation_history.append(timestamped_msg)
            self.token_history.append((datetime.now(), self._count_tokens(msg)))

        # Trim history if needed
        self._trim_conversation_history()

        # Log current states
        logger.debug(
            f"Message count: {len(self.conversation_history)} "
            f"Tokens: {self._count_messages_tokens()}"
        )

    async def _analyze_token_usage(
        self, conversation_history: List[Dict]
    ) -> TokenUsage:
        """Analyze token usage with dynamic burn rate calculation"""
        full_tokens = self._count_messages_tokens(conversation_history)
        history_tokens = self._count_messages_tokens(self.conversation_history)

        # Calculate dynamic burn rate using sliding window
        burn_rate = self._calculate_dynamic_burn_rate()

        # Estimate time to limit
        remaining_tokens = self.token_limit - max(full_tokens, history_tokens)
        time_to_limit = (
            remaining_tokens / max(burn_rate, 0.0001) if burn_rate > 0 else None
        )

        # Determine threshold status
        threshold_reached = max(full_tokens, history_tokens) >= (
            self.token_limit * self.threshold_ratio
        )

        # Calculate prediction confidence
        prediction_confidence = min(
            0.95, math.sqrt(len(self.token_history) / 100)
        )  # grows with more data

        return TokenUsage(
            current_tokens=full_tokens,
            estimated_total=max(full_tokens, history_tokens)
            + (burn_rate * 60),  # Project for next hour
            burn_rate=burn_rate,
            time_to_limit=time_to_limit,
            threshold_reached=threshold_reached,
            prediction_confidence=prediction_confidence,
            token_history=list(self.token_history),
            last_update_duration=(
                datetime.now() - self.session_start_time
            ).total_seconds()
            / 60,
        )

    def _calculate_dynamic_burn_rate(self) -> float:
        """Calculate burn rate using time-based sliding window"""
        now = datetime.now()
        window_start = now - timedelta(minutes=self.burn_rate_window)

        # Filter token history within window
        window_data = [
            (ts, tokens) for ts, tokens in self.token_history if ts >= window_start
        ]

        if len(window_data) < 2:
            # Not enough data, use simple burn rate
            session_duration = max(
                0.01, (now - self.session_start_time).total_seconds() / 60
            )
            return (
                max(0, self._count_messages_tokens(self.conversation_history))
                / session_duration
            )

        # Calculate burn rate from window data
        tokens_start, tokens_end = window_data[0][1], window_data[-1][1]
        time_diff = (window_data[-1][0] - window_data[0][0]).total_seconds() / 60
        burn_rate = (tokens_end - tokens_start) / time_diff if time_diff > 0 else 0

        return max(0, burn_rate)

    async def _analyze_session_metrics(self, session_data: Dict) -> SessionMetrics:
        """Analyze session-level metrics and patterns"""
        duration = (datetime.now() - self.session_start_time).total_seconds() / 60

        # Count tool usage
        tool_usage = session_data.get("tool_usage", [])
        tool_counts = {}
        for tool in tool_usage:
            name = tool.get("name", "unknown")
            tool_counts[name] = tool_counts.get(name, 0) + 1

        # Calculate file changes
        file_changes = session_data.get("file_changes", [])
        file_modifications = len(
            [change for change in file_changes if change.get("type") in ("M", "A")]
        )
        commands = tool_counts.get("Bash", 0)

        # Calculate error rate
        total_operations = sum(tool_counts.values())
        errors = session_data.get("error_count", 0)
        error_rate = errors / max(total_operations, 1)

        # Estimate context switches
        context_switches = self._estimate_context_switches(tool_usage)

        # Calculate relevance score based on message patterns
        relevance_score = self._calculate_relevance_score(session_data)

        return SessionMetrics(
            duration_minutes=duration,
            tool_usage_count=tool_counts,
            file_modifications=file_modifications,
            command_execution_count=commands,
            error_rate=error_rate,
            context_switches=context_switches,
            relevance_score=relevance_score,
        )

    async def _analyze_task_complexity(self, session_data: Dict) -> TaskComplexity:
        """Analyze task complexity with weighted scoring"""
        file_changes = session_data.get("file_changes", [])
        tool_usage = session_data.get("tool_usage", [])

        # Calculate complexity factors
        file_count = len(set(change.get("path", "") for change in file_changes))
        language_variety = self._count_language_variety(file_changes)
        dependency_depth = self._estimate_dependency_depth(file_changes)
        session_duration = (
            datetime.now() - self.session_start_time
        ).total_seconds() / 3600
        tool_complexity = self._calculate_tool_complexity(tool_usage)

        # Calculate weighted complexity score
        factors = {
            "file_count": min(1.0, file_count / 10),
            "language_variety": min(1.0, language_variety / 5),
            "dependency_depth": min(1.0, dependency_depth / 3),
            "session_duration": min(1.0, session_duration / 2),
            "tool_complexity": tool_complexity,
        }

        score = sum(factors[key] * self.weights[key] for key in factors)

        # Determine complexity level
        if score < 0.3:
            level = ComplexityLevel.LOW
        elif score < 0.6:
            level = ComplexityLevel.MEDIUM
        elif score < 0.85:
            level = ComplexityLevel.HIGH
        else:
            level = ComplexityLevel.CRITICAL

        return TaskComplexity(
            level=level,
            score=score,
            factors=factors,
            file_count=file_count,
            language_variety=language_variety,
            dependency_depth=dependency_depth,
            estimated_time_hours=session_duration + (score * 2),
        )

    async def _classify_session_type(self, tool_usage: List[Dict]) -> SessionType:
        """Classify session type based on tool usage patterns"""
        if not tool_usage:
            return SessionType.RESEARCH

        tool_counts = {}
        for tool in tool_usage:
            name = tool.get("name", "unknown")
            tool_counts[name] = tool_counts.get(name, 0) + 1

        total_tools = sum(tool_counts.values())
        if total_tools == 0:
            return SessionType.RESEARCH

        # Calculate tool usage ratios
        edit_ratio = (
            tool_counts.get("Edit", 0) + tool_counts.get("MultiEdit", 0)
        ) / total_tools
        read_ratio = tool_counts.get("Read", 0) / total_tools
        bash_ratio = tool_counts.get("Bash", 0) / total_tools
        search_ratio = (
            tool_counts.get("Grep", 0) + tool_counts.get("Glob", 0)
        ) / total_tools

        # Classify based on dominant patterns
        if edit_ratio > 0.4:
            return SessionType.CODING
        elif bash_ratio > 0.3:
            return SessionType.DEBUGGING
        elif search_ratio > 0.3:
            return SessionType.RESEARCH
        elif read_ratio > 0.5:
            return SessionType.ARCHITECTURE
        else:
            return SessionType.MIXED

    def _generate_recommendations(
        self,
        token_usage: TokenUsage,
        session_metrics: SessionMetrics,
        task_complexity: TaskComplexity,
        session_type: SessionType,
    ) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []

        # Token-based recommendations
        if token_usage.threshold_reached:
            recommendations.append("Consider using /compact to reduce token usage")
        if token_usage.time_to_limit and token_usage.time_to_limit < 15:
            recommendations.append(
                "Critical: Token limit approaching, immediate action needed"
            )

        # Session-based recommendations
        if session_metrics.duration_minutes > 60:
            recommendations.append(
                "Long session detected - consider /digest --snapshot"
            )
        if session_metrics.error_rate > 0.2:
            recommendations.append("High error rate detected - review recent changes")

        # Complexity-based recommendations
        if task_complexity.level == ComplexityLevel.CRITICAL:
            recommendations.append(
                "Critical complexity - consider breaking into smaller tasks"
            )
        elif task_complexity.level == ComplexityLevel.HIGH:
            recommendations.append("High complexity - document progress frequently")

        # Type-specific recommendations
        if (
            session_type == SessionType.CODING
            and session_metrics.file_modifications > 10
        ):
            recommendations.append("Many files modified - consider commit checkpoint")
        elif (
            session_type == SessionType.DEBUGGING
            and session_metrics.duration_minutes > 30
        ):
            recommendations.append("Extended debugging - document findings")

        return recommendations

    def _calculate_confidence_score(
        self,
        token_usage: TokenUsage,
        session_metrics: SessionMetrics,
        task_complexity: TaskComplexity,
    ) -> float:
        """Calculate overall confidence in analysis"""
        factors = []

        # Token prediction confidence
        factors.append(token_usage.prediction_confidence)

        # Session data completeness
        data_completeness = min(1.0, session_metrics.duration_minutes / 10)
        factors.append(data_completeness)

        # History depth
        history_depth = min(1.0, len(self.conversation_history) / 50)
        factors.append(history_depth)

        # Tool usage diversity
        tool_diversity = min(1.0, len(session_metrics.tool_usage_count) / 5)
        factors.append(tool_diversity)

        return sum(factors) / len(factors)

    def _count_tokens(self, message: Dict) -> int:
        """Count tokens in a message"""
        content = message.get("content", "")
        if self.token_encoder:
            return len(self.token_encoder.encode(str(content)))
        else:
            # Approximate token count
            return len(str(content).split()) * 1.3

    def _count_messages_tokens(self, messages: List[Dict] = None) -> int:
        """Count total tokens in message list"""
        if messages is None:
            messages = list(self.conversation_history)
        return sum(self._count_tokens(msg) for msg in messages)

    def _trim_conversation_history(self):
        """Trim conversation history based on time and size limits"""
        if not self.conversation_history:
            return

        now = datetime.now()
        cutoff_time = now - timedelta(minutes=self.keep_last_minutes)

        # Remove old messages
        while (
            self.conversation_history
            and self.conversation_history[0].get("analysis_timestamp", now)
            < cutoff_time
        ):
            self.conversation_history.popleft()

    def _should_generate_summary(self) -> bool:
        """Check if we should generate a new summary"""
        time_since_last = datetime.now() - self.last_summary_time
        return time_since_last.total_seconds() > (self.MAX_SUMMARY_AGE * 60)

    async def _generate_conversation_summary(self, messages: deque) -> str:
        """Generate summary of conversation history"""
        if not messages:
            return "No conversation history available"

        # Simple extractive summary
        key_messages = [msg for msg in messages if len(msg.get("content", "")) > 100]
        if not key_messages:
            return "Brief conversation with limited content"

        # Update last summary time
        self.last_summary_time = datetime.now()

        return f"Session summary: {len(key_messages)} key messages analyzed"

    def _estimate_context_switches(self, tool_usage: List[Dict]) -> int:
        """Estimate number of context switches"""
        if len(tool_usage) < 2:
            return 0

        switches = 0
        prev_tool = None

        for tool in tool_usage:
            current_tool = tool.get("name", "unknown")
            if prev_tool and prev_tool != current_tool:
                switches += 1
            prev_tool = current_tool

        return switches

    def _calculate_relevance_score(self, session_data: Dict) -> float:
        """Calculate session relevance score"""
        # Simple relevance based on activity level
        tool_usage = len(session_data.get("tool_usage", []))
        file_changes = len(session_data.get("file_changes", []))
        duration = (datetime.now() - self.session_start_time).total_seconds() / 60

        activity_score = (tool_usage + file_changes) / max(duration, 1)
        return min(1.0, activity_score / 5)

    def _count_language_variety(self, file_changes: List[Dict]) -> int:
        """Count variety of programming languages"""
        extensions = set()
        for change in file_changes:
            path = change.get("path", "")
            if "." in path:
                ext = path.split(".")[-1].lower()
                extensions.add(ext)
        return len(extensions)

    def _estimate_dependency_depth(self, file_changes: List[Dict]) -> int:
        """Estimate dependency depth from file paths"""
        max_depth = 0
        for change in file_changes:
            path = change.get("path", "")
            depth = path.count("/")
            max_depth = max(max_depth, depth)
        return max_depth

    def _calculate_tool_complexity(self, tool_usage: List[Dict]) -> float:
        """Calculate tool complexity score"""
        if not tool_usage:
            return 0.0

        complex_tools = {"Bash", "MultiEdit", "Task", "WebFetch"}
        simple_tools = {"Read", "LS", "Glob"}

        complex_count = sum(
            1 for tool in tool_usage if tool.get("name") in complex_tools
        )
        simple_count = sum(1 for tool in tool_usage if tool.get("name") in simple_tools)
        total = len(tool_usage)

        if total == 0:
            return 0.0

        return (complex_count * 1.0 + simple_count * 0.3) / total

    async def _create_fallback_analysis(self, session_data: Dict) -> ContextAnalysis:
        """Create basic fallback analysis when main analysis fails"""
        duration = (datetime.now() - self.session_start_time).total_seconds() / 60

        return ContextAnalysis(
            session_type=SessionType.MIXED,
            token_usage=TokenUsage(
                current_tokens=0,
                estimated_total=0,
                burn_rate=0.0,
                threshold_reached=False,
                prediction_confidence=0.0,
            ),
            session_metrics=SessionMetrics(
                duration_minutes=duration,
                tool_usage_count={},
                file_modifications=0,
                command_execution_count=0,
                error_rate=0.0,
                context_switches=0,
                relevance_score=0.0,
            ),
            task_complexity=TaskComplexity(
                level=ComplexityLevel.LOW,
                score=0.0,
                factors={},
                file_count=0,
                language_variety=0,
                dependency_depth=0,
                estimated_time_hours=0.0,
            ),
            recommendations=["Analysis failed - using fallback mode"],
            confidence_score=0.0,
            timestamp=datetime.now(),
            summary="Fallback analysis due to error",
        )

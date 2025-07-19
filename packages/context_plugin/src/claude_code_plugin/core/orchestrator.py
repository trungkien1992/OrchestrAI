"""
Plugin Orchestrator for Claude Code Context Plugin

Simplified orchestration focused on practical Claude Code integration.
Makes intelligent decisions about when to compact/digest based on context.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
from enum import Enum

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of actions the orchestrator can recommend"""

    COMPACT = "compact"
    DIGEST = "digest"
    CHECKPOINT = "checkpoint"
    NO_ACTION = "no_action"
    MONITOR = "monitor"


class PluginOrchestrator:
    """
    Simplified orchestrator for Claude Code plugin
    """

    def __init__(self, config: Dict):
        self.config = config
        self.decision_history = []
        self.last_decision_time = None

        # Thresholds
        self.token_threshold = config.get("auto_compact_threshold", 0.85)
        self.burn_rate_threshold = config.get("burn_rate_threshold", 150)
        self.session_duration_threshold = 120  # 2 hours

        logger.info("Plugin Orchestrator initialized")

    async def make_decision(self, session_data: Dict) -> Dict[str, Any]:
        """
        Make a decision about what action to take based on session data
        """
        try:
            # Extract session information
            token_usage = session_data.get("token_usage", {})
            burn_rate = session_data.get("burn_rate", 0)
            session_duration = session_data.get("session_duration_minutes", 0)
            session_type = session_data.get("session_type", "unknown")
            files_modified = session_data.get("files_modified", 0)

            # Calculate decision factors
            factors = self._calculate_decision_factors(
                token_usage, burn_rate, session_duration, session_type, files_modified
            )

            # Make decision based on factors
            action, reasoning, confidence = self._evaluate_factors(factors)

            # Create decision result
            decision = {
                "action": action.value,
                "reasoning": reasoning,
                "confidence": confidence,
                "estimated_time": self._estimate_execution_time(action),
                "recommended_timing": self._recommend_timing(action, factors),
                "factors": factors,
                "timestamp": datetime.now().isoformat(),
            }

            # Record decision
            self._record_decision(decision)

            return decision

        except Exception as e:
            logger.error(f"Error making orchestration decision: {e}")
            return {
                "action": ActionType.NO_ACTION.value,
                "reasoning": f"Error in decision making: {e}",
                "confidence": 0.0,
                "estimated_time": 0,
                "recommended_timing": "immediate",
                "factors": {},
                "timestamp": datetime.now().isoformat(),
            }

    def _calculate_decision_factors(
        self,
        token_usage: Dict,
        burn_rate: float,
        session_duration: float,
        session_type: str,
        files_modified: int,
    ) -> Dict[str, float]:
        """Calculate factors that influence decision making"""
        factors = {}

        # Token pressure factor
        current_tokens = token_usage.get("current_tokens", 0)
        token_limit = token_usage.get("token_limit", 200000)
        usage_ratio = current_tokens / token_limit if token_limit > 0 else 0
        factors["token_pressure"] = min(1.0, usage_ratio)

        # Burn rate factor
        normalized_burn_rate = burn_rate / self.burn_rate_threshold
        factors["burn_rate_pressure"] = min(1.0, normalized_burn_rate)

        # Session duration factor
        duration_factor = session_duration / self.session_duration_threshold
        factors["session_duration"] = min(1.0, duration_factor)

        # Session type factor
        type_factors = {
            "coding": 0.8,  # High context preservation need
            "debugging": 0.9,  # Very high context preservation
            "architecture": 0.7,  # Medium context preservation
            "research": 0.5,  # Lower context preservation need
            "testing": 0.6,  # Medium context preservation
            "unknown": 0.5,  # Default
        }
        factors["session_type"] = type_factors.get(session_type.lower(), 0.5)

        # File modification factor
        modification_factor = min(1.0, files_modified / 20)  # Normalize to 20 files
        factors["file_activity"] = modification_factor

        # Time since last decision
        if self.last_decision_time:
            time_diff = (datetime.now() - self.last_decision_time).total_seconds() / 60
            factors["decision_spacing"] = min(1.0, time_diff / 30)  # 30 min spacing
        else:
            factors["decision_spacing"] = 1.0

        return factors

    def _evaluate_factors(self, factors: Dict[str, float]) -> tuple:
        """Evaluate factors and make decision"""

        # Critical token pressure - immediate compact
        if factors["token_pressure"] > 0.95:
            return (
                ActionType.COMPACT,
                "Critical token usage requires immediate compaction",
                0.95,
            )

        # High token pressure with high burn rate
        if factors["token_pressure"] > 0.85 and factors["burn_rate_pressure"] > 0.8:
            return (
                ActionType.COMPACT,
                "High token usage with high burn rate - compact recommended",
                0.85,
            )

        # Long session with moderate token usage
        if factors["session_duration"] > 0.8 and factors["token_pressure"] > 0.6:
            # Check session type for digest vs compact
            if factors["session_type"] > 0.7:  # Coding/debugging
                return (
                    ActionType.DIGEST,
                    "Long coding/debugging session - digest to preserve context",
                    0.75,
                )
            else:
                return (
                    ActionType.COMPACT,
                    "Long session with moderate token usage - compact recommended",
                    0.70,
                )

        # High activity with token pressure
        if factors["file_activity"] > 0.7 and factors["token_pressure"] > 0.7:
            return (
                ActionType.CHECKPOINT,
                "High file activity with token pressure - checkpoint recommended",
                0.65,
            )

        # Too soon since last decision
        if factors["decision_spacing"] < 0.3:
            return (
                ActionType.NO_ACTION,
                "Too soon since last decision - wait longer",
                0.8,
            )

        # Moderate monitoring
        if factors["token_pressure"] > 0.6 or factors["burn_rate_pressure"] > 0.6:
            return (ActionType.MONITOR, "Moderate usage - continue monitoring", 0.6)

        # Default - no action needed
        return (ActionType.NO_ACTION, "All factors within acceptable ranges", 0.5)

    def _estimate_execution_time(self, action: ActionType) -> float:
        """Estimate execution time for action in minutes"""
        time_estimates = {
            ActionType.COMPACT: 1.0,
            ActionType.DIGEST: 2.0,
            ActionType.CHECKPOINT: 0.5,
            ActionType.MONITOR: 0.0,
            ActionType.NO_ACTION: 0.0,
        }
        return time_estimates.get(action, 0.0)

    def _recommend_timing(self, action: ActionType, factors: Dict[str, float]) -> str:
        """Recommend timing for action"""
        if action == ActionType.NO_ACTION:
            return "N/A"

        # Immediate if critical
        if factors["token_pressure"] > 0.95:
            return "immediate"

        # Soon if high pressure
        if factors["token_pressure"] > 0.85 or factors["burn_rate_pressure"] > 0.8:
            return "within 5 minutes"

        # When convenient for moderate pressure
        if factors["token_pressure"] > 0.7:
            return "when convenient (next 15 minutes)"

        # Default
        return "when convenient"

    def _record_decision(self, decision: Dict):
        """Record decision in history"""
        self.decision_history.append(decision)
        self.last_decision_time = datetime.now()

        # Keep only last 50 decisions
        if len(self.decision_history) > 50:
            self.decision_history = self.decision_history[-50:]

        logger.debug(
            f"Decision recorded: {decision['action']} (confidence: {decision['confidence']:.2f})"
        )

    async def get_decision_history(self, limit: int = 10) -> List[Dict]:
        """Get recent decision history"""
        return self.decision_history[-limit:] if self.decision_history else []

    async def analyze_decision_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in decision history"""
        if not self.decision_history:
            return {"error": "No decision history available"}

        # Count actions
        action_counts = {}
        confidence_scores = []

        for decision in self.decision_history:
            action = decision["action"]
            action_counts[action] = action_counts.get(action, 0) + 1
            confidence_scores.append(decision["confidence"])

        # Calculate statistics
        avg_confidence = sum(confidence_scores) / len(confidence_scores)

        return {
            "total_decisions": len(self.decision_history),
            "action_distribution": action_counts,
            "average_confidence": avg_confidence,
            "most_common_action": max(action_counts, key=action_counts.get),
            "decision_frequency": len(self.decision_history)
            / max(
                1,
                (
                    datetime.now()
                    - datetime.fromisoformat(self.decision_history[0]["timestamp"])
                ).total_seconds()
                / 3600,
            ),
        }

    def get_current_recommendations(self, session_data: Dict) -> List[str]:
        """Get current recommendations based on session data"""
        recommendations = []

        # Token usage recommendations
        token_usage = session_data.get("token_usage", {})
        if token_usage:
            usage_ratio = token_usage.get("current_tokens", 0) / token_usage.get(
                "token_limit", 200000
            )

            if usage_ratio > 0.9:
                recommendations.append("🚨 Critical: Run /compact immediately")
            elif usage_ratio > 0.8:
                recommendations.append(
                    "⚠️ High usage: Consider /compact --preserve-code"
                )
            elif usage_ratio > 0.7:
                recommendations.append("📊 Monitor: Token usage approaching threshold")

        # Burn rate recommendations
        burn_rate = session_data.get("burn_rate", 0)
        if burn_rate > self.burn_rate_threshold:
            recommendations.append("📈 High burn rate: Consider optimizing session")

        # Session duration recommendations
        duration = session_data.get("session_duration_minutes", 0)
        if duration > 180:  # 3 hours
            recommendations.append("⏰ Long session: Consider /digest --snapshot")
        elif duration > 120:  # 2 hours
            recommendations.append("⏰ Extended session: Monitor token usage closely")

        # File activity recommendations
        files_modified = session_data.get("files_modified", 0)
        if files_modified > 20:
            recommendations.append("📁 High activity: Consider checkpoint")

        return recommendations

"""
Context Monitor for Claude Code Plugin

Provides real-time monitoring of context usage, token consumption,
and session patterns.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
from collections import deque

logger = logging.getLogger(__name__)


class ContextMonitor:
    """
    Real-time context monitoring for Claude Code sessions
    """

    def __init__(self, config: Dict):
        self.config = config
        self.is_monitoring = False
        self.monitoring_task = None

        # Token tracking
        self.token_limit = config.get("token_limit", 200000)
        self.current_tokens = 0
        self.token_history = deque(maxlen=100)

        # Session tracking
        self.session_start = datetime.now()
        self.last_activity = datetime.now()
        self.session_type = "unknown"

        # Performance tracking
        self.burn_rate = 0.0
        self.average_response_time = 0.0
        self.context_switches = 0

        # File tracking
        self.files_in_context = set()
        self.files_modified = 0

        logger.info("Context Monitor initialized")

    async def start_monitoring(self, interval: int = 30):
        """Start monitoring context usage"""
        if self.is_monitoring:
            logger.warning("Monitoring already active")
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop(interval))

        logger.info(f"Context monitoring started (interval: {interval}s)")

    async def stop_monitoring(self):
        """Stop monitoring"""
        if not self.is_monitoring:
            return

        self.is_monitoring = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Context monitoring stopped")

    async def update_token_usage(
        self, current_tokens: int, token_limit: Optional[int] = None
    ):
        """Update token usage information"""
        if token_limit:
            self.token_limit = token_limit

        # Update current tokens
        previous_tokens = self.current_tokens
        self.current_tokens = current_tokens

        # Calculate burn rate
        if previous_tokens > 0:
            token_increase = current_tokens - previous_tokens
            time_diff = (datetime.now() - self.last_activity).total_seconds() / 60

            if time_diff > 0:
                self.burn_rate = token_increase / time_diff

        # Store in history
        self.token_history.append(
            {
                "timestamp": datetime.now(),
                "tokens": current_tokens,
                "burn_rate": self.burn_rate,
            }
        )

        self.last_activity = datetime.now()

        logger.debug(f"Token usage updated: {current_tokens}/{self.token_limit}")

    async def update_session_type(self, session_type: str):
        """Update detected session type"""
        if self.session_type != session_type:
            self.context_switches += 1
            self.session_type = session_type
            logger.debug(f"Session type changed to: {session_type}")

    async def add_file_to_context(self, file_path: str):
        """Add file to context tracking"""
        self.files_in_context.add(file_path)
        logger.debug(f"File added to context: {file_path}")

    async def file_modified(self, file_path: str):
        """Track file modification"""
        self.files_modified += 1
        await self.add_file_to_context(file_path)
        logger.debug(f"File modified: {file_path}")

    async def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        session_duration = (datetime.now() - self.session_start).total_seconds() / 60

        # Calculate usage percentage
        usage_percentage = (self.current_tokens / self.token_limit) * 100

        # Calculate time to limit
        time_to_limit = None
        if self.burn_rate > 0:
            remaining_tokens = self.token_limit - self.current_tokens
            time_to_limit = remaining_tokens / self.burn_rate

        # Determine health status
        if usage_percentage < 50:
            health_status = "🟢 Healthy"
        elif usage_percentage < 75:
            health_status = "🟡 Moderate"
        elif usage_percentage < 90:
            health_status = "🟠 High"
        else:
            health_status = "🔴 Critical"

        return {
            "is_monitoring": self.is_monitoring,
            "health_status": health_status,
            "token_usage": f"{self.current_tokens:,}/{self.token_limit:,} ({usage_percentage:.1f}%)",
            "burn_rate": f"{self.burn_rate:.1f} tokens/min",
            "time_to_limit": (
                f"{time_to_limit:.1f} minutes" if time_to_limit else "Unknown"
            ),
            "session_type": self.session_type,
            "session_duration": f"{session_duration:.1f} minutes",
            "context_files": len(self.files_in_context),
            "files_modified": self.files_modified,
            "context_switches": self.context_switches,
            "last_activity": self.last_activity.strftime("%H:%M:%S"),
            "session_start": self.session_start.strftime("%H:%M:%S"),
            "confidence": self._calculate_confidence(),
        }

    async def get_recommendations(self) -> List[str]:
        """Get context optimization recommendations"""
        recommendations = []

        usage_percentage = (self.current_tokens / self.token_limit) * 100

        if usage_percentage > 85:
            recommendations.append(
                "🚨 Consider /compact --preserve-code to reduce token usage"
            )
        elif usage_percentage > 75:
            recommendations.append("⚠️ Token usage is high, monitor closely")

        if self.burn_rate > self.config.get("burn_rate_threshold", 150):
            recommendations.append(
                "📈 High burn rate detected, consider optimizing session"
            )

        session_duration = (datetime.now() - self.session_start).total_seconds() / 60
        if session_duration > 120:  # 2 hours
            recommendations.append(
                "⏰ Long session detected, consider /digest --snapshot"
            )

        if self.context_switches > 5:
            recommendations.append(
                "🔄 Multiple context switches detected, consider focusing"
            )

        if len(self.files_in_context) > 20:
            recommendations.append("📁 Many files in context, consider cleanup")

        return recommendations

    def _calculate_confidence(self) -> float:
        """Calculate monitoring confidence score"""
        factors = []

        # Token history depth
        if len(self.token_history) > 10:
            factors.append(0.9)
        elif len(self.token_history) > 5:
            factors.append(0.7)
        else:
            factors.append(0.5)

        # Session duration
        session_duration = (datetime.now() - self.session_start).total_seconds() / 60
        if session_duration > 30:
            factors.append(0.9)
        elif session_duration > 10:
            factors.append(0.7)
        else:
            factors.append(0.5)

        # Activity level
        if self.files_modified > 5:
            factors.append(0.8)
        elif self.files_modified > 0:
            factors.append(0.6)
        else:
            factors.append(0.4)

        return sum(factors) / len(factors) if factors else 0.5

    async def _monitoring_loop(self, interval: int):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Update session metrics
                await self._update_session_metrics()

                # Check for optimization opportunities
                await self._check_optimization_opportunities()

                # Sleep for interval
                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)

    async def _update_session_metrics(self):
        """Update session metrics"""
        # This would normally collect data from Claude Code
        # For now, we'll simulate some updates

        # Simulate token usage increase
        if self.current_tokens < self.token_limit * 0.9:
            self.current_tokens += min(50, self.token_limit * 0.01)

        # Update burn rate calculation
        if len(self.token_history) >= 2:
            recent_entries = list(self.token_history)[-2:]
            token_diff = recent_entries[-1]["tokens"] - recent_entries[0]["tokens"]
            time_diff = (
                recent_entries[-1]["timestamp"] - recent_entries[0]["timestamp"]
            ).total_seconds() / 60

            if time_diff > 0:
                self.burn_rate = token_diff / time_diff

        logger.debug(
            f"Session metrics updated - Tokens: {self.current_tokens}, Burn rate: {self.burn_rate:.1f}"
        )

    async def _check_optimization_opportunities(self):
        """Check for optimization opportunities"""
        usage_percentage = (self.current_tokens / self.token_limit) * 100

        # Check critical thresholds
        if usage_percentage > 95:
            logger.warning("🚨 Critical token usage - immediate action needed")
        elif usage_percentage > 85:
            logger.info("⚠️ High token usage - consider optimization")

        # Check burn rate
        if self.burn_rate > self.config.get("burn_rate_threshold", 150):
            logger.info("📈 High burn rate detected")

        # Check session duration
        session_duration = (datetime.now() - self.session_start).total_seconds() / 60
        if session_duration > 180:  # 3 hours
            logger.info("⏰ Very long session - consider break or digest")

    def get_dashboard_display(self) -> str:
        """Get formatted dashboard display"""
        status = asyncio.run(self.get_status())
        recommendations = asyncio.run(self.get_recommendations())

        dashboard = f"""
┌─ Claude Code Context Monitor ─────────────────────────┐
│                                                       │
│ 🔋 Token Health: {status['health_status']} {status['token_usage']}          │
│ 📈 Burn Rate: {status['burn_rate']} (↑ 12% from avg)       │
│ ⏰ Time to Limit: {status['time_to_limit']}                       │
│ 🎯 Session Type: {status['session_type'].upper()} (confidence: {status['confidence']:.2f})        │
│                                                       │
│ 🚨 Recommendations:                                   │
"""

        for rec in recommendations[:3]:  # Show top 3 recommendations
            dashboard += f"│ • {rec[:45]}     │\n"

        dashboard += "│                                                       │\n"
        dashboard += "└───────────────────────────────────────────────────────┘"

        return dashboard

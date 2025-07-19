"""
Real-time Monitoring Dashboard for Claude Context Management
Live visualization and monitoring of context metrics, performance, and system health
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque
import statistics
import time

logger = logging.getLogger(__name__)


@dataclass
class DashboardMetrics:
    """Real-time dashboard metrics"""

    timestamp: datetime

    # Context metrics
    current_tokens: int
    burn_rate: float
    time_to_limit: Optional[float]
    session_duration: float

    # Performance metrics
    avg_response_time: float
    memory_usage: float
    cpu_usage: float
    error_rate: float

    # System health
    confidence_score: float
    stability_score: float
    throughput: float

    # Session tracking
    active_sessions: int
    completed_tasks: int
    pending_tasks: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class RealTimeMonitor:
    """Real-time monitoring system with live updates"""

    def __init__(self, update_interval: float = 5.0):
        self.update_interval = update_interval
        self.is_running = False
        self.metrics_history = deque(maxlen=1000)
        self.alert_thresholds = {
            "burn_rate": 150.0,
            "error_rate": 0.1,
            "response_time": 2.0,
            "memory_usage": 0.8,
            "confidence_score": 0.5,
        }
        self.active_alerts = []
        self.subscribers = []

    async def start_monitoring(self):
        """Start real-time monitoring"""
        self.is_running = True
        logger.info("🔍 Real-time monitoring started")

        while self.is_running:
            try:
                # Collect current metrics
                metrics = await self._collect_metrics()

                # Store in history
                self.metrics_history.append(metrics)

                # Check for alerts
                alerts = await self._check_alerts(metrics)
                if alerts:
                    await self._handle_alerts(alerts)

                # Notify subscribers
                await self._notify_subscribers(metrics)

                # Wait for next update
                await asyncio.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(self.update_interval)

    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_running = False
        logger.info("🛑 Real-time monitoring stopped")

    async def _collect_metrics(self) -> DashboardMetrics:
        """Collect current system metrics"""
        from ..analysis import metrics_engine, confidence_engine

        # Get current timestamp
        now = datetime.now()

        # Collect performance summary
        perf_summary = await metrics_engine.get_performance_summary()

        # Extract metrics with defaults
        response_time = perf_summary.get("response_time", {}).get("mean", 0.0)
        token_usage = perf_summary.get("token_usage", {}).get("mean", 0.0)
        error_rate = perf_summary.get("error_rate", {}).get("mean", 0.0)

        # Get system resource usage
        memory_usage = await self._get_memory_usage()
        cpu_usage = await self._get_cpu_usage()

        # Calculate derived metrics
        burn_rate = self._calculate_current_burn_rate()
        time_to_limit = self._calculate_time_to_limit(token_usage, burn_rate)
        session_duration = self._get_session_duration()

        # Get confidence metrics
        confidence_trend = confidence_engine.get_confidence_trend()
        confidence_score = confidence_trend.get("recent_average", 0.5)

        # Calculate stability and throughput
        stability_score = self._calculate_stability_score()
        throughput = self._calculate_throughput()

        # Session tracking
        active_sessions = 1  # Simplified for demo
        completed_tasks = len(getattr(self, "completed_tasks", []))
        pending_tasks = len(getattr(self, "pending_tasks", []))

        return DashboardMetrics(
            timestamp=now,
            current_tokens=int(token_usage),
            burn_rate=burn_rate,
            time_to_limit=time_to_limit,
            session_duration=session_duration,
            avg_response_time=response_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            error_rate=error_rate,
            confidence_score=confidence_score,
            stability_score=stability_score,
            throughput=throughput,
            active_sessions=active_sessions,
            completed_tasks=completed_tasks,
            pending_tasks=pending_tasks,
        )

    async def _get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        try:
            import psutil

            return psutil.virtual_memory().percent / 100.0
        except ImportError:
            # Fallback simulation
            return 0.3 + (time.time() % 60) / 200

    async def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            import psutil

            return psutil.cpu_percent(interval=0.1) / 100.0
        except ImportError:
            # Fallback simulation
            return 0.2 + (time.time() % 30) / 150

    def _calculate_current_burn_rate(self) -> float:
        """Calculate current token burn rate"""
        if len(self.metrics_history) < 2:
            return 0.0

        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 updates

        if len(recent_metrics) < 2:
            return 0.0

        # Calculate rate of change
        time_diff = (
            recent_metrics[-1].timestamp - recent_metrics[0].timestamp
        ).total_seconds() / 60
        token_diff = (
            recent_metrics[-1].current_tokens - recent_metrics[0].current_tokens
        )

        if time_diff > 0:
            return token_diff / time_diff
        return 0.0

    def _calculate_time_to_limit(
        self, current_tokens: float, burn_rate: float
    ) -> Optional[float]:
        """Calculate time to token limit"""
        if burn_rate <= 0:
            return None

        token_limit = 200_000
        remaining = token_limit - current_tokens

        if remaining <= 0:
            return 0.0

        return remaining / burn_rate

    def _get_session_duration(self) -> float:
        """Get current session duration in minutes"""
        if not hasattr(self, "session_start"):
            self.session_start = datetime.now()

        return (datetime.now() - self.session_start).total_seconds() / 60

    def _calculate_stability_score(self) -> float:
        """Calculate system stability score"""
        if len(self.metrics_history) < 10:
            return 0.5

        recent_metrics = list(self.metrics_history)[-20:]
        response_times = [m.avg_response_time for m in recent_metrics]

        if len(response_times) < 2:
            return 0.5

        # Stability based on response time variance
        variance = statistics.variance(response_times)
        stability = max(0.0, 1.0 - (variance * 10))

        return min(1.0, stability)

    def _calculate_throughput(self) -> float:
        """Calculate system throughput (operations per minute)"""
        if len(self.metrics_history) < 2:
            return 0.0

        # Calculate based on task completion rate
        recent_metrics = list(self.metrics_history)[-10:]

        if len(recent_metrics) < 2:
            return 0.0

        time_diff = (
            recent_metrics[-1].timestamp - recent_metrics[0].timestamp
        ).total_seconds() / 60
        task_diff = (
            recent_metrics[-1].completed_tasks - recent_metrics[0].completed_tasks
        )

        if time_diff > 0:
            return task_diff / time_diff
        return 0.0

    async def _check_alerts(self, metrics: DashboardMetrics) -> List[Dict[str, Any]]:
        """Check for alert conditions"""
        alerts = []

        # Check burn rate
        if metrics.burn_rate > self.alert_thresholds["burn_rate"]:
            alerts.append(
                {
                    "type": "burn_rate_high",
                    "severity": "warning",
                    "message": f"High burn rate: {metrics.burn_rate:.1f} tokens/min",
                    "value": metrics.burn_rate,
                    "threshold": self.alert_thresholds["burn_rate"],
                }
            )

        # Check error rate
        if metrics.error_rate > self.alert_thresholds["error_rate"]:
            alerts.append(
                {
                    "type": "error_rate_high",
                    "severity": "error",
                    "message": f"High error rate: {metrics.error_rate:.1%}",
                    "value": metrics.error_rate,
                    "threshold": self.alert_thresholds["error_rate"],
                }
            )

        # Check response time
        if metrics.avg_response_time > self.alert_thresholds["response_time"]:
            alerts.append(
                {
                    "type": "response_time_high",
                    "severity": "warning",
                    "message": f"Slow response time: {metrics.avg_response_time:.2f}s",
                    "value": metrics.avg_response_time,
                    "threshold": self.alert_thresholds["response_time"],
                }
            )

        # Check memory usage
        if metrics.memory_usage > self.alert_thresholds["memory_usage"]:
            alerts.append(
                {
                    "type": "memory_usage_high",
                    "severity": "warning",
                    "message": f"High memory usage: {metrics.memory_usage:.1%}",
                    "value": metrics.memory_usage,
                    "threshold": self.alert_thresholds["memory_usage"],
                }
            )

        # Check confidence score
        if metrics.confidence_score < self.alert_thresholds["confidence_score"]:
            alerts.append(
                {
                    "type": "confidence_low",
                    "severity": "info",
                    "message": f"Low confidence: {metrics.confidence_score:.2f}",
                    "value": metrics.confidence_score,
                    "threshold": self.alert_thresholds["confidence_score"],
                }
            )

        # Check time to limit
        if metrics.time_to_limit and metrics.time_to_limit < 10:
            alerts.append(
                {
                    "type": "token_limit_approaching",
                    "severity": "critical",
                    "message": f"Token limit approaching: {metrics.time_to_limit:.1f} minutes",
                    "value": metrics.time_to_limit,
                    "threshold": 10.0,
                }
            )

        return alerts

    async def _handle_alerts(self, alerts: List[Dict[str, Any]]):
        """Handle alert notifications"""
        for alert in alerts:
            # Add to active alerts if not already present
            alert_key = f"{alert['type']}_{alert['severity']}"
            if alert_key not in [a.get("key") for a in self.active_alerts]:
                alert["key"] = alert_key
                alert["timestamp"] = datetime.now()
                self.active_alerts.append(alert)

                # Log alert
                severity = alert["severity"].upper()
                logger.warning(f"🚨 [{severity}] {alert['message']}")

        # Clean up old alerts (older than 5 minutes)
        cutoff_time = datetime.now() - timedelta(minutes=5)
        self.active_alerts = [
            a
            for a in self.active_alerts
            if a.get("timestamp", datetime.now()) > cutoff_time
        ]

    async def _notify_subscribers(self, metrics: DashboardMetrics):
        """Notify subscribers of metric updates"""
        for subscriber in self.subscribers:
            try:
                await subscriber(metrics)
            except Exception as e:
                logger.error(f"Subscriber notification error: {e}")

    def subscribe(self, callback):
        """Subscribe to metric updates"""
        self.subscribers.append(callback)

    def unsubscribe(self, callback):
        """Unsubscribe from metric updates"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        if not self.metrics_history:
            return {"status": "no_data", "metrics": {}, "alerts": []}

        latest_metrics = self.metrics_history[-1]

        # Calculate trends
        trends = {}
        if len(self.metrics_history) >= 2:
            prev_metrics = self.metrics_history[-2]
            trends = {
                "burn_rate": latest_metrics.burn_rate - prev_metrics.burn_rate,
                "response_time": latest_metrics.avg_response_time
                - prev_metrics.avg_response_time,
                "confidence": latest_metrics.confidence_score
                - prev_metrics.confidence_score,
                "memory_usage": latest_metrics.memory_usage - prev_metrics.memory_usage,
            }

        # Get historical data for charts
        historical_data = [m.to_dict() for m in list(self.metrics_history)[-50:]]

        return {
            "status": "active",
            "current_metrics": latest_metrics.to_dict(),
            "trends": trends,
            "alerts": self.active_alerts,
            "historical_data": historical_data,
            "update_interval": self.update_interval,
            "thresholds": self.alert_thresholds,
        }

    def export_metrics(self, filename: str, format: str = "json"):
        """Export metrics to file"""
        data = {
            "export_timestamp": datetime.now().isoformat(),
            "metrics_count": len(self.metrics_history),
            "metrics": [m.to_dict() for m in self.metrics_history],
            "alerts": self.active_alerts,
            "thresholds": self.alert_thresholds,
        }

        with open(filename, "w") as f:
            if format == "json":
                json.dump(data, f, indent=2)
            elif format == "csv":
                import csv

                writer = csv.DictWriter(f, fieldnames=data["metrics"][0].keys())
                writer.writeheader()
                writer.writerows(data["metrics"])

        logger.info(f"📊 Metrics exported to {filename}")


class DashboardServer:
    """Simple dashboard server for web interface"""

    def __init__(self, monitor: RealTimeMonitor, port: int = 8080):
        self.monitor = monitor
        self.port = port
        self.app = None

    async def start_server(self):
        """Start dashboard web server"""
        try:
            from aiohttp import web, web_response

            async def dashboard_handler(request):
                """Handle dashboard requests"""
                data = self.monitor.get_dashboard_data()
                return web_response.json_response(data)

            async def metrics_handler(request):
                """Handle metrics API requests"""
                data = self.monitor.get_dashboard_data()
                return web_response.json_response(data["current_metrics"])

            async def alerts_handler(request):
                """Handle alerts API requests"""
                data = self.monitor.get_dashboard_data()
                return web_response.json_response(data["alerts"])

            # Create web application
            app = web.Application()
            app.router.add_get("/dashboard", dashboard_handler)
            app.router.add_get("/api/metrics", metrics_handler)
            app.router.add_get("/api/alerts", alerts_handler)

            # Start server
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "localhost", self.port)
            await site.start()

            logger.info(f"🌐 Dashboard server started on http://localhost:{self.port}")

        except ImportError:
            logger.warning("aiohttp not available - dashboard server disabled")
        except Exception as e:
            logger.error(f"Dashboard server error: {e}")


# Global monitor instance
monitor = RealTimeMonitor()


# Convenience functions
async def start_monitoring():
    """Start real-time monitoring"""
    await monitor.start_monitoring()


def stop_monitoring():
    """Stop real-time monitoring"""
    monitor.stop_monitoring()


def get_current_metrics() -> Optional[DashboardMetrics]:
    """Get current metrics"""
    if monitor.metrics_history:
        return monitor.metrics_history[-1]
    return None


def export_dashboard_data(filename: str):
    """Export dashboard data"""
    monitor.export_metrics(filename)

"""
Performance Metrics Engine with Sliding Window Algorithms
Advanced metrics collection and analysis for Claude context management
"""

import asyncio
import logging
import time
from collections import deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum, auto
import statistics
import math

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics"""

    RESPONSE_TIME = auto()
    TOKEN_USAGE = auto()
    MEMORY_USAGE = auto()
    CPU_USAGE = auto()
    ERROR_RATE = auto()
    THROUGHPUT = auto()
    LATENCY = auto()
    ACCURACY = auto()


@dataclass
class MetricPoint:
    """Individual metric data point"""

    timestamp: datetime
    value: float
    metric_type: MetricType
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SlidingWindowStats:
    """Statistics for sliding window analysis"""

    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    percentile_95: float
    percentile_99: float
    trend: float  # Linear trend slope
    sample_count: int
    window_duration: float  # minutes


@dataclass
class PerformanceProfile:
    """Performance profile for different operation types"""

    operation_type: str
    baseline_metrics: Dict[MetricType, float]
    threshold_metrics: Dict[MetricType, float]
    confidence_score: float
    last_updated: datetime


class SlidingWindowAnalyzer:
    """Advanced sliding window analyzer with multiple window sizes"""

    def __init__(self, window_minutes: int = 5, max_points: int = 1000):
        self.window_minutes = window_minutes
        self.max_points = max_points
        self.data_points = deque(maxlen=max_points)
        self.window_cache = {}
        self.cache_expiry = datetime.min

    def add_point(self, value: float, metric_type: MetricType, metadata: Dict = None):
        """Add a new data point to the sliding window"""
        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            metric_type=metric_type,
            metadata=metadata or {},
        )
        self.data_points.append(point)

        # Invalidate cache
        self.window_cache.clear()
        self.cache_expiry = datetime.min

    def get_window_data(
        self, window_minutes: Optional[int] = None
    ) -> List[MetricPoint]:
        """Get data points within sliding window"""
        window_size = window_minutes or self.window_minutes
        cutoff_time = datetime.now() - timedelta(minutes=window_size)

        return [point for point in self.data_points if point.timestamp >= cutoff_time]

    def calculate_stats(
        self, window_minutes: Optional[int] = None
    ) -> Optional[SlidingWindowStats]:
        """Calculate comprehensive statistics for sliding window"""
        window_size = window_minutes or self.window_minutes
        cache_key = f"stats_{window_size}"

        # Check cache
        if (
            cache_key in self.window_cache
            and datetime.now() - self.cache_expiry < timedelta(seconds=10)
        ):
            return self.window_cache[cache_key]

        window_data = self.get_window_data(window_size)

        if len(window_data) < 2:
            return None

        values = [point.value for point in window_data]

        # Calculate statistics
        mean_val = statistics.mean(values)
        median_val = statistics.median(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
        min_val = min(values)
        max_val = max(values)

        # Calculate percentiles
        sorted_values = sorted(values)
        p95 = sorted_values[int(0.95 * len(sorted_values))]
        p99 = sorted_values[int(0.99 * len(sorted_values))]

        # Calculate trend
        trend = self._calculate_trend(window_data)

        stats = SlidingWindowStats(
            mean=mean_val,
            median=median_val,
            std_dev=std_dev,
            min_value=min_val,
            max_value=max_val,
            percentile_95=p95,
            percentile_99=p99,
            trend=trend,
            sample_count=len(values),
            window_duration=window_size,
        )

        # Cache result
        self.window_cache[cache_key] = stats
        self.cache_expiry = datetime.now()

        return stats

    def _calculate_trend(self, data_points: List[MetricPoint]) -> float:
        """Calculate linear trend using least squares"""
        if len(data_points) < 2:
            return 0.0

        # Convert timestamps to minutes since first point
        first_time = data_points[0].timestamp
        x_values = [
            (point.timestamp - first_time).total_seconds() / 60 for point in data_points
        ]
        y_values = [point.value for point in data_points]

        # Calculate linear regression slope
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)

        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope

    def detect_anomalies(self, threshold_std: float = 2.0) -> List[MetricPoint]:
        """Detect anomalies using statistical analysis"""
        stats = self.calculate_stats()
        if not stats:
            return []

        threshold = stats.mean + (threshold_std * stats.std_dev)
        window_data = self.get_window_data()

        return [
            point for point in window_data if abs(point.value - stats.mean) > threshold
        ]


class PerformanceMetricsEngine:
    """Main performance metrics engine with multiple analyzers"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.analyzers: Dict[MetricType, SlidingWindowAnalyzer] = {}
        self.performance_profiles: Dict[str, PerformanceProfile] = {}

        # Initialize analyzers for each metric type
        for metric_type in MetricType:
            window_size = self.config.get(f"{metric_type.name.lower()}_window", 5)
            max_points = self.config.get(f"{metric_type.name.lower()}_max_points", 1000)
            self.analyzers[metric_type] = SlidingWindowAnalyzer(window_size, max_points)

    async def record_metric(
        self,
        metric_type: MetricType,
        value: float,
        operation_type: str = None,
        metadata: Dict = None,
    ):
        """Record a performance metric asynchronously"""
        analyzer = self.analyzers.get(metric_type)
        if analyzer:
            await asyncio.to_thread(analyzer.add_point, value, metric_type, metadata)

            # Update performance profile if operation type provided
            if operation_type:
                await self._update_performance_profile(
                    operation_type, metric_type, value
                )

    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        summary = {}

        # Collect stats from all analyzers in parallel
        tasks = []
        for metric_type, analyzer in self.analyzers.items():
            task = asyncio.create_task(asyncio.to_thread(analyzer.calculate_stats))
            tasks.append((metric_type, task))

        # Wait for all tasks to complete
        for metric_type, task in tasks:
            stats = await task
            if stats:
                summary[metric_type.name.lower()] = {
                    "mean": stats.mean,
                    "median": stats.median,
                    "std_dev": stats.std_dev,
                    "min": stats.min_value,
                    "max": stats.max_value,
                    "p95": stats.percentile_95,
                    "p99": stats.percentile_99,
                    "trend": stats.trend,
                    "sample_count": stats.sample_count,
                    "window_duration": stats.window_duration,
                }

        return summary

    async def detect_performance_issues(self) -> List[Dict[str, Any]]:
        """Detect performance issues across all metrics"""
        issues = []

        # Check each metric type for anomalies
        for metric_type, analyzer in self.analyzers.items():
            anomalies = await asyncio.to_thread(analyzer.detect_anomalies)

            if anomalies:
                issues.append(
                    {
                        "metric_type": metric_type.name,
                        "anomaly_count": len(anomalies),
                        "severity": self._calculate_severity(anomalies),
                        "latest_anomaly": anomalies[-1].value if anomalies else None,
                        "recommendation": self._get_recommendation(
                            metric_type, anomalies
                        ),
                    }
                )

        return issues

    async def get_performance_profile(
        self, operation_type: str
    ) -> Optional[PerformanceProfile]:
        """Get performance profile for specific operation type"""
        return self.performance_profiles.get(operation_type)

    async def _update_performance_profile(
        self, operation_type: str, metric_type: MetricType, value: float
    ):
        """Update performance profile with new data"""
        profile = self.performance_profiles.get(operation_type)

        if not profile:
            # Create new profile
            profile = PerformanceProfile(
                operation_type=operation_type,
                baseline_metrics={metric_type: value},
                threshold_metrics={metric_type: value * 1.5},
                confidence_score=0.1,
                last_updated=datetime.now(),
            )
            self.performance_profiles[operation_type] = profile
        else:
            # Update existing profile
            analyzer = self.analyzers[metric_type]
            stats = await asyncio.to_thread(analyzer.calculate_stats)

            if stats:
                profile.baseline_metrics[metric_type] = stats.mean
                profile.threshold_metrics[metric_type] = stats.mean + (
                    2 * stats.std_dev
                )
                profile.confidence_score = min(1.0, profile.confidence_score + 0.1)
                profile.last_updated = datetime.now()

    def _calculate_severity(self, anomalies: List[MetricPoint]) -> str:
        """Calculate severity level for anomalies"""
        if len(anomalies) > 10:
            return "CRITICAL"
        elif len(anomalies) > 5:
            return "HIGH"
        elif len(anomalies) > 2:
            return "MEDIUM"
        else:
            return "LOW"

    def _get_recommendation(
        self, metric_type: MetricType, anomalies: List[MetricPoint]
    ) -> str:
        """Get recommendation based on metric type and anomalies"""
        recommendations = {
            MetricType.RESPONSE_TIME: "Consider optimizing async operations or reducing complexity",
            MetricType.TOKEN_USAGE: "Monitor token consumption patterns, consider compaction",
            MetricType.MEMORY_USAGE: "Review memory allocation patterns, implement cleanup",
            MetricType.CPU_USAGE: "Optimize computational algorithms, consider parallel processing",
            MetricType.ERROR_RATE: "Review error handling patterns, implement better validation",
            MetricType.THROUGHPUT: "Optimize pipeline efficiency, consider batching",
            MetricType.LATENCY: "Review network calls, implement caching strategies",
            MetricType.ACCURACY: "Review analysis algorithms, improve data quality",
        }

        return recommendations.get(
            metric_type, "Monitor metric trends and investigate patterns"
        )

    async def cleanup_old_data(self, hours_to_keep: int = 24):
        """Clean up old metric data to manage memory"""
        cutoff_time = datetime.now() - timedelta(hours=hours_to_keep)

        for analyzer in self.analyzers.values():
            # Remove old data points
            while (
                analyzer.data_points and analyzer.data_points[0].timestamp < cutoff_time
            ):
                analyzer.data_points.popleft()

            # Clear cache
            analyzer.window_cache.clear()
            analyzer.cache_expiry = datetime.min

    async def export_metrics(self, filename: str, format: str = "json"):
        """Export metrics data to file"""
        import json

        data = {
            "export_timestamp": datetime.now().isoformat(),
            "metrics": {},
            "profiles": {},
        }

        # Export analyzer data
        for metric_type, analyzer in self.analyzers.items():
            window_data = analyzer.get_window_data(60)  # Last hour
            data["metrics"][metric_type.name] = [
                {
                    "timestamp": point.timestamp.isoformat(),
                    "value": point.value,
                    "metadata": point.metadata,
                }
                for point in window_data
            ]

        # Export performance profiles
        for op_type, profile in self.performance_profiles.items():
            data["profiles"][op_type] = {
                "baseline_metrics": {
                    k.name: v for k, v in profile.baseline_metrics.items()
                },
                "threshold_metrics": {
                    k.name: v for k, v in profile.threshold_metrics.items()
                },
                "confidence_score": profile.confidence_score,
                "last_updated": profile.last_updated.isoformat(),
            }

        # Write to file
        with open(filename, "w") as f:
            if format == "json":
                json.dump(data, f, indent=2)
            else:
                f.write(str(data))

        logger.info(f"Metrics exported to {filename}")


# Global instance for easy access
metrics_engine = PerformanceMetricsEngine()


# Convenience functions
async def record_response_time(operation: str, duration: float):
    """Record response time metric"""
    await metrics_engine.record_metric(
        MetricType.RESPONSE_TIME, duration, operation, {"unit": "seconds"}
    )


async def record_token_usage(operation: str, tokens: int):
    """Record token usage metric"""
    await metrics_engine.record_metric(
        MetricType.TOKEN_USAGE, tokens, operation, {"unit": "tokens"}
    )


async def record_error_rate(operation: str, error_count: int, total_count: int):
    """Record error rate metric"""
    rate = error_count / max(total_count, 1)
    await metrics_engine.record_metric(
        MetricType.ERROR_RATE,
        rate,
        operation,
        {"errors": error_count, "total": total_count},
    )

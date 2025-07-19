"""
Performance Benchmarking Suite for Claude Context Management
Comprehensive performance testing and benchmarking tools
"""

import asyncio
import time
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import sys
import traceback

logger = logging.getLogger(__name__)


class BenchmarkType(Enum):
    """Types of benchmarks"""

    LATENCY = auto()
    THROUGHPUT = auto()
    MEMORY = auto()
    CONCURRENCY = auto()
    ACCURACY = auto()
    STABILITY = auto()


class BenchmarkStatus(Enum):
    """Benchmark execution status"""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class BenchmarkResult:
    """Individual benchmark result"""

    name: str
    benchmark_type: BenchmarkType
    status: BenchmarkStatus

    # Timing metrics
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: float = 0.0

    # Performance metrics
    operations_per_second: float = 0.0
    latency_mean: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    memory_usage: float = 0.0

    # Detailed results
    individual_results: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Error information
    error_message: str = ""
    error_count: int = 0

    def calculate_stats(self):
        """Calculate statistical metrics from individual results"""
        if not self.individual_results:
            return

        self.latency_mean = statistics.mean(self.individual_results)

        if len(self.individual_results) > 1:
            sorted_results = sorted(self.individual_results)
            self.latency_p95 = sorted_results[int(0.95 * len(sorted_results))]
            self.latency_p99 = sorted_results[int(0.99 * len(sorted_results))]

        if self.duration > 0:
            self.operations_per_second = len(self.individual_results) / self.duration

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "benchmark_type": self.benchmark_type.name,
            "status": self.status.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "operations_per_second": self.operations_per_second,
            "latency_mean": self.latency_mean,
            "latency_p95": self.latency_p95,
            "latency_p99": self.latency_p99,
            "memory_usage": self.memory_usage,
            "error_count": self.error_count,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results"""

    name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    results: List[BenchmarkResult] = field(default_factory=list)

    def add_result(self, result: BenchmarkResult):
        """Add benchmark result"""
        self.results.append(result)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all results"""
        if not self.results:
            return {"total": 0, "passed": 0, "failed": 0}

        total = len(self.results)
        passed = len([r for r in self.results if r.status == BenchmarkStatus.COMPLETED])
        failed = len([r for r in self.results if r.status == BenchmarkStatus.FAILED])

        # Calculate overall performance metrics
        latencies = [r.latency_mean for r in self.results if r.latency_mean > 0]
        throughputs = [
            r.operations_per_second for r in self.results if r.operations_per_second > 0
        ]

        summary = {
            "total": total,
            "passed": passed,
            "failed": failed,
            "success_rate": passed / total if total > 0 else 0,
            "total_duration": (
                (self.end_time - self.start_time).total_seconds()
                if self.end_time
                else 0
            ),
        }

        if latencies:
            summary["avg_latency"] = statistics.mean(latencies)

        if throughputs:
            summary["avg_throughput"] = statistics.mean(throughputs)

        return summary


class PerformanceBenchmarker:
    """Main performance benchmarking engine"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.default_iterations = self.config.get("default_iterations", 100)
        self.default_warmup = self.config.get("default_warmup", 10)
        self.default_timeout = self.config.get("default_timeout", 30)

    async def run_latency_benchmark(
        self, name: str, operation: Callable, iterations: int = None, warmup: int = None
    ) -> BenchmarkResult:
        """Run latency benchmark"""

        iterations = iterations or self.default_iterations
        warmup = warmup or self.default_warmup

        result = BenchmarkResult(
            name=name,
            benchmark_type=BenchmarkType.LATENCY,
            status=BenchmarkStatus.RUNNING,
            start_time=datetime.now(),
        )

        try:
            logger.info(f"🏃 Starting latency benchmark: {name}")

            # Warmup phase
            for _ in range(warmup):
                try:
                    if asyncio.iscoroutinefunction(operation):
                        await operation()
                    else:
                        operation()
                except Exception:
                    pass  # Ignore warmup errors

            # Main benchmark
            for i in range(iterations):
                start_time = time.perf_counter()

                try:
                    if asyncio.iscoroutinefunction(operation):
                        await operation()
                    else:
                        operation()

                    duration = time.perf_counter() - start_time
                    result.individual_results.append(duration)

                except Exception as e:
                    result.error_count += 1
                    if not result.error_message:
                        result.error_message = str(e)

            # Finalize results
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
            result.calculate_stats()
            result.status = BenchmarkStatus.COMPLETED

            logger.info(
                f"✅ Latency benchmark completed: {name} - {result.latency_mean:.4f}s avg"
            )

        except Exception as e:
            result.status = BenchmarkStatus.FAILED
            result.error_message = str(e)
            logger.error(f"❌ Latency benchmark failed: {name} - {e}")

        return result

    async def run_throughput_benchmark(
        self, name: str, operation: Callable, duration_seconds: float = 10.0
    ) -> BenchmarkResult:
        """Run throughput benchmark"""

        result = BenchmarkResult(
            name=name,
            benchmark_type=BenchmarkType.THROUGHPUT,
            status=BenchmarkStatus.RUNNING,
            start_time=datetime.now(),
        )

        try:
            logger.info(f"🏃 Starting throughput benchmark: {name}")

            operation_count = 0
            start_time = time.perf_counter()
            end_time = start_time + duration_seconds

            while time.perf_counter() < end_time:
                try:
                    if asyncio.iscoroutinefunction(operation):
                        await operation()
                    else:
                        operation()

                    operation_count += 1

                except Exception as e:
                    result.error_count += 1
                    if not result.error_message:
                        result.error_message = str(e)

            actual_duration = time.perf_counter() - start_time

            # Finalize results
            result.end_time = datetime.now()
            result.duration = actual_duration
            result.operations_per_second = operation_count / actual_duration
            result.status = BenchmarkStatus.COMPLETED
            result.metadata["operation_count"] = operation_count

            logger.info(
                f"✅ Throughput benchmark completed: {name} - {result.operations_per_second:.2f} ops/sec"
            )

        except Exception as e:
            result.status = BenchmarkStatus.FAILED
            result.error_message = str(e)
            logger.error(f"❌ Throughput benchmark failed: {name} - {e}")

        return result

    async def run_memory_benchmark(
        self, name: str, operation: Callable, iterations: int = None
    ) -> BenchmarkResult:
        """Run memory usage benchmark"""

        iterations = iterations or self.default_iterations

        result = BenchmarkResult(
            name=name,
            benchmark_type=BenchmarkType.MEMORY,
            status=BenchmarkStatus.RUNNING,
            start_time=datetime.now(),
        )

        try:
            import psutil
            import os

            logger.info(f"🏃 Starting memory benchmark: {name}")

            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Run operations
            for i in range(iterations):
                try:
                    if asyncio.iscoroutinefunction(operation):
                        await operation()
                    else:
                        operation()

                    # Sample memory usage
                    current_memory = process.memory_info().rss / 1024 / 1024  # MB
                    result.individual_results.append(current_memory - initial_memory)

                except Exception as e:
                    result.error_count += 1
                    if not result.error_message:
                        result.error_message = str(e)

            # Finalize results
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()

            if result.individual_results:
                result.memory_usage = max(result.individual_results)
                result.latency_mean = statistics.mean(result.individual_results)

            result.status = BenchmarkStatus.COMPLETED

            logger.info(
                f"✅ Memory benchmark completed: {name} - {result.memory_usage:.2f}MB peak"
            )

        except ImportError:
            result.status = BenchmarkStatus.FAILED
            result.error_message = "psutil not available for memory benchmarking"
            logger.error(f"❌ Memory benchmark failed: {name} - psutil not available")
        except Exception as e:
            result.status = BenchmarkStatus.FAILED
            result.error_message = str(e)
            logger.error(f"❌ Memory benchmark failed: {name} - {e}")

        return result

    async def run_concurrency_benchmark(
        self, name: str, operation: Callable, concurrency_levels: List[int] = None
    ) -> BenchmarkResult:
        """Run concurrency benchmark"""

        concurrency_levels = concurrency_levels or [1, 2, 4, 8, 16]

        result = BenchmarkResult(
            name=name,
            benchmark_type=BenchmarkType.CONCURRENCY,
            status=BenchmarkStatus.RUNNING,
            start_time=datetime.now(),
        )

        try:
            logger.info(f"🏃 Starting concurrency benchmark: {name}")

            concurrency_results = {}

            for concurrency in concurrency_levels:
                start_time = time.perf_counter()

                # Create concurrent tasks
                tasks = []
                for _ in range(concurrency):
                    if asyncio.iscoroutinefunction(operation):
                        tasks.append(asyncio.create_task(operation()))
                    else:
                        tasks.append(asyncio.create_task(asyncio.to_thread(operation)))

                # Wait for all tasks to complete
                try:
                    await asyncio.gather(*tasks, return_exceptions=True)
                    duration = time.perf_counter() - start_time
                    throughput = concurrency / duration

                    concurrency_results[concurrency] = {
                        "duration": duration,
                        "throughput": throughput,
                    }

                    result.individual_results.append(throughput)

                except Exception as e:
                    result.error_count += 1
                    if not result.error_message:
                        result.error_message = str(e)

            # Finalize results
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
            result.calculate_stats()
            result.metadata["concurrency_results"] = concurrency_results
            result.status = BenchmarkStatus.COMPLETED

            logger.info(f"✅ Concurrency benchmark completed: {name}")

        except Exception as e:
            result.status = BenchmarkStatus.FAILED
            result.error_message = str(e)
            logger.error(f"❌ Concurrency benchmark failed: {name} - {e}")

        return result

    async def run_benchmark_suite(
        self, benchmarks: List[Tuple[str, Callable, Dict[str, Any]]]
    ) -> BenchmarkSuite:
        """Run a complete benchmark suite"""

        suite = BenchmarkSuite(
            name=f"benchmark_suite_{datetime.now().timestamp()}",
            start_time=datetime.now(),
        )

        logger.info(f"🏁 Starting benchmark suite with {len(benchmarks)} benchmarks")

        for name, operation, config in benchmarks:
            benchmark_type = config.get("type", "latency")

            try:
                if benchmark_type == "latency":
                    result = await self.run_latency_benchmark(
                        name, operation, config.get("iterations"), config.get("warmup")
                    )
                elif benchmark_type == "throughput":
                    result = await self.run_throughput_benchmark(
                        name, operation, config.get("duration", 10.0)
                    )
                elif benchmark_type == "memory":
                    result = await self.run_memory_benchmark(
                        name, operation, config.get("iterations")
                    )
                elif benchmark_type == "concurrency":
                    result = await self.run_concurrency_benchmark(
                        name, operation, config.get("concurrency_levels")
                    )
                else:
                    result = await self.run_latency_benchmark(name, operation)

                suite.add_result(result)

            except Exception as e:
                logger.error(f"Benchmark suite error for {name}: {e}")

                # Create failed result
                failed_result = BenchmarkResult(
                    name=name,
                    benchmark_type=BenchmarkType.LATENCY,
                    status=BenchmarkStatus.FAILED,
                    start_time=datetime.now(),
                    error_message=str(e),
                )
                suite.add_result(failed_result)

        suite.end_time = datetime.now()

        # Generate summary
        summary = suite.get_summary()
        logger.info(
            f"🏁 Benchmark suite completed: {summary['passed']}/{summary['total']} passed"
        )

        return suite

    def export_results(
        self, suite: BenchmarkSuite, filename: str, format: str = "json"
    ):
        """Export benchmark results to file"""

        data = {
            "suite_name": suite.name,
            "start_time": suite.start_time.isoformat(),
            "end_time": suite.end_time.isoformat() if suite.end_time else None,
            "summary": suite.get_summary(),
            "results": [result.to_dict() for result in suite.results],
        }

        with open(filename, "w") as f:
            if format == "json":
                json.dump(data, f, indent=2)
            elif format == "csv":
                import csv

                writer = csv.DictWriter(f, fieldnames=data["results"][0].keys())
                writer.writeheader()
                writer.writerows(data["results"])

        logger.info(f"📊 Benchmark results exported to {filename}")


# Global benchmarker instance
benchmarker = PerformanceBenchmarker()


# Pre-configured benchmark operations
async def benchmark_context_analysis():
    """Benchmark context analysis operation"""
    from core.analysis import ContextAnalysisEngine

    engine = ContextAnalysisEngine()
    session_data = {
        "conversation_history": [
            {"content": "Test message 1", "timestamp": datetime.now()},
            {"content": "Test message 2", "timestamp": datetime.now()},
        ],
        "tool_usage": [
            {"name": "Read", "timestamp": datetime.now()},
            {"name": "Edit", "timestamp": datetime.now()},
        ],
        "file_changes": [{"path": "test.py", "type": "M"}],
        "error_count": 0,
    }

    await engine.analyze_session_context(session_data)


async def benchmark_confidence_scoring():
    """Benchmark confidence scoring operation"""
    from core.analysis import calculate_analysis_confidence

    await calculate_analysis_confidence(
        data_quality=0.8, sample_size=100, historical_accuracy=0.75
    )


async def benchmark_performance_metrics():
    """Benchmark performance metrics operation"""
    from core.analysis import metrics_engine, MetricType

    await metrics_engine.record_metric(MetricType.RESPONSE_TIME, 0.15, "test_operation")


# Convenience functions
async def run_full_benchmark_suite() -> BenchmarkSuite:
    """Run full benchmark suite for Claude Context Management"""

    benchmarks = [
        (
            "context_analysis",
            benchmark_context_analysis,
            {"type": "latency", "iterations": 50},
        ),
        (
            "confidence_scoring",
            benchmark_confidence_scoring,
            {"type": "latency", "iterations": 100},
        ),
        (
            "performance_metrics",
            benchmark_performance_metrics,
            {"type": "throughput", "duration": 5.0},
        ),
        (
            "context_analysis_memory",
            benchmark_context_analysis,
            {"type": "memory", "iterations": 20},
        ),
        (
            "confidence_scoring_concurrency",
            benchmark_confidence_scoring,
            {"type": "concurrency", "concurrency_levels": [1, 2, 4, 8]},
        ),
    ]

    return await benchmarker.run_benchmark_suite(benchmarks)


async def quick_performance_check() -> Dict[str, float]:
    """Quick performance check of key operations"""

    results = {}

    # Test context analysis
    start_time = time.perf_counter()
    await benchmark_context_analysis()
    results["context_analysis"] = time.perf_counter() - start_time

    # Test confidence scoring
    start_time = time.perf_counter()
    await benchmark_confidence_scoring()
    results["confidence_scoring"] = time.perf_counter() - start_time

    # Test performance metrics
    start_time = time.perf_counter()
    await benchmark_performance_metrics()
    results["performance_metrics"] = time.perf_counter() - start_time

    return results

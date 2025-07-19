"""
Confidence Scoring Engine for Claude Context Management
Advanced confidence calculation and reliability assessment
"""

import asyncio
import logging
import math
from collections import deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum, auto
import statistics

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence level categories"""

    VERY_LOW = auto()  # 0.0 - 0.2
    LOW = auto()  # 0.2 - 0.4
    MEDIUM = auto()  # 0.4 - 0.6
    HIGH = auto()  # 0.6 - 0.8
    VERY_HIGH = auto()  # 0.8 - 1.0


class ConfidenceSource(Enum):
    """Sources of confidence data"""

    DATA_QUALITY = auto()
    HISTORICAL_ACCURACY = auto()
    SAMPLE_SIZE = auto()
    MODEL_STABILITY = auto()
    EXTERNAL_VALIDATION = auto()
    TEMPORAL_CONSISTENCY = auto()
    CROSS_VALIDATION = auto()


@dataclass
class ConfidenceMetric:
    """Individual confidence metric"""

    source: ConfidenceSource
    value: float  # 0.0 to 1.0
    weight: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if metric is valid"""
        return 0.0 <= self.value <= 1.0 and self.weight > 0


@dataclass
class ConfidenceAssessment:
    """Complete confidence assessment"""

    overall_confidence: float
    confidence_level: ConfidenceLevel
    component_scores: Dict[ConfidenceSource, float]
    reliability_score: float
    stability_score: float

    # Detailed breakdown
    data_quality_score: float = 0.0
    sample_size_score: float = 0.0
    historical_accuracy_score: float = 0.0
    model_stability_score: float = 0.0

    # Recommendations
    improvement_suggestions: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)

    timestamp: datetime = field(default_factory=datetime.now)


class ConfidenceEngine:
    """Main confidence scoring engine"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # Confidence weights for different sources
        self.source_weights = {
            ConfidenceSource.DATA_QUALITY: 0.25,
            ConfidenceSource.HISTORICAL_ACCURACY: 0.20,
            ConfidenceSource.SAMPLE_SIZE: 0.15,
            ConfidenceSource.MODEL_STABILITY: 0.15,
            ConfidenceSource.EXTERNAL_VALIDATION: 0.10,
            ConfidenceSource.TEMPORAL_CONSISTENCY: 0.10,
            ConfidenceSource.CROSS_VALIDATION: 0.05,
        }

        # Update weights from config
        self.source_weights.update(self.config.get("confidence_weights", {}))

        # Historical data for tracking
        self.confidence_history = deque(maxlen=1000)
        self.accuracy_history = deque(maxlen=500)
        self.stability_metrics = deque(maxlen=100)

        # Thresholds
        self.min_sample_size = self.config.get("min_sample_size", 10)
        self.stability_window = self.config.get("stability_window", 50)
        self.decay_factor = self.config.get("decay_factor", 0.95)

    async def calculate_confidence(
        self, metrics: List[ConfidenceMetric], context: Dict[str, Any] = None
    ) -> ConfidenceAssessment:
        """Calculate comprehensive confidence assessment"""

        # Validate metrics
        valid_metrics = [m for m in metrics if m.is_valid()]

        if not valid_metrics:
            return await self._create_low_confidence_assessment("No valid metrics")

        # Calculate component scores in parallel
        tasks = [
            self._calculate_data_quality_score(valid_metrics),
            self._calculate_sample_size_score(valid_metrics, context),
            self._calculate_historical_accuracy_score(),
            self._calculate_model_stability_score(),
            self._calculate_temporal_consistency_score(valid_metrics),
            self._calculate_cross_validation_score(valid_metrics),
        ]

        scores = await asyncio.gather(*tasks)

        # Organize scores by source
        component_scores = {
            ConfidenceSource.DATA_QUALITY: scores[0],
            ConfidenceSource.SAMPLE_SIZE: scores[1],
            ConfidenceSource.HISTORICAL_ACCURACY: scores[2],
            ConfidenceSource.MODEL_STABILITY: scores[3],
            ConfidenceSource.TEMPORAL_CONSISTENCY: scores[4],
            ConfidenceSource.CROSS_VALIDATION: scores[5],
        }

        # Calculate weighted overall confidence
        overall_confidence = self._calculate_weighted_confidence(component_scores)

        # Determine confidence level
        confidence_level = self._determine_confidence_level(overall_confidence)

        # Calculate reliability and stability
        reliability_score = await self._calculate_reliability_score(component_scores)
        stability_score = await self._calculate_stability_score()

        # Generate recommendations and risk factors
        suggestions = await self._generate_improvement_suggestions(component_scores)
        risk_factors = await self._identify_risk_factors(component_scores)

        # Create assessment
        assessment = ConfidenceAssessment(
            overall_confidence=overall_confidence,
            confidence_level=confidence_level,
            component_scores=component_scores,
            reliability_score=reliability_score,
            stability_score=stability_score,
            data_quality_score=scores[0],
            sample_size_score=scores[1],
            historical_accuracy_score=scores[2],
            model_stability_score=scores[3],
            improvement_suggestions=suggestions,
            risk_factors=risk_factors,
        )

        # Store in history
        self.confidence_history.append(assessment)

        return assessment

    async def _calculate_data_quality_score(
        self, metrics: List[ConfidenceMetric]
    ) -> float:
        """Calculate data quality confidence score"""
        if not metrics:
            return 0.0

        quality_factors = []

        # Check for completeness
        completeness = len([m for m in metrics if m.value > 0]) / len(metrics)
        quality_factors.append(completeness)

        # Check for consistency
        values = [m.value for m in metrics]
        if len(values) > 1:
            consistency = 1.0 - (
                statistics.stdev(values) / max(statistics.mean(values), 0.001)
            )
            quality_factors.append(max(0.0, consistency))

        # Check for freshness
        now = datetime.now()
        freshness_scores = []
        for metric in metrics:
            age_hours = (now - metric.timestamp).total_seconds() / 3600
            freshness = math.exp(-age_hours / 24)  # Decay over 24 hours
            freshness_scores.append(freshness)

        if freshness_scores:
            quality_factors.append(statistics.mean(freshness_scores))

        return statistics.mean(quality_factors) if quality_factors else 0.0

    async def _calculate_sample_size_score(
        self, metrics: List[ConfidenceMetric], context: Dict[str, Any] = None
    ) -> float:
        """Calculate sample size confidence score"""
        sample_size = len(metrics)

        if sample_size < self.min_sample_size:
            return sample_size / self.min_sample_size

        # Logarithmic scaling for diminishing returns
        max_useful_size = 100
        if sample_size >= max_useful_size:
            return 1.0

        return math.log(sample_size) / math.log(max_useful_size)

    async def _calculate_historical_accuracy_score(self) -> float:
        """Calculate historical accuracy confidence score"""
        if not self.accuracy_history:
            return 0.5  # Neutral when no history

        # Calculate weighted average with decay
        total_weight = 0.0
        weighted_sum = 0.0

        for i, accuracy in enumerate(reversed(self.accuracy_history)):
            weight = self.decay_factor**i
            weighted_sum += accuracy * weight
            total_weight += weight

        if total_weight == 0:
            return 0.5

        return weighted_sum / total_weight

    async def _calculate_model_stability_score(self) -> float:
        """Calculate model stability confidence score"""
        if len(self.stability_metrics) < 2:
            return 0.5  # Neutral when insufficient data

        # Calculate variance in recent predictions
        recent_metrics = list(self.stability_metrics)[-self.stability_window :]

        if len(recent_metrics) < 2:
            return 0.5

        variance = statistics.variance(recent_metrics)

        # Convert variance to stability score (lower variance = higher stability)
        stability = math.exp(-variance * 10)  # Exponential decay

        return min(1.0, stability)

    async def _calculate_temporal_consistency_score(
        self, metrics: List[ConfidenceMetric]
    ) -> float:
        """Calculate temporal consistency confidence score"""
        if len(metrics) < 2:
            return 0.5

        # Sort by timestamp
        sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)

        # Calculate consistency over time
        consistency_scores = []
        window_size = min(5, len(sorted_metrics))

        for i in range(len(sorted_metrics) - window_size + 1):
            window = sorted_metrics[i : i + window_size]
            values = [m.value for m in window]

            if len(values) > 1:
                mean_val = statistics.mean(values)
                if mean_val > 0:
                    cv = statistics.stdev(values) / mean_val  # Coefficient of variation
                    consistency = math.exp(-cv * 2)  # Convert to consistency score
                    consistency_scores.append(consistency)

        return statistics.mean(consistency_scores) if consistency_scores else 0.5

    async def _calculate_cross_validation_score(
        self, metrics: List[ConfidenceMetric]
    ) -> float:
        """Calculate cross-validation confidence score"""
        # This is a simplified implementation
        # In practice, you'd implement proper cross-validation

        if len(metrics) < 3:
            return 0.5

        # Calculate agreement between different metrics
        values = [m.value for m in metrics]
        mean_val = statistics.mean(values)

        if mean_val == 0:
            return 0.5

        # Calculate how closely metrics agree
        deviations = [abs(v - mean_val) / mean_val for v in values]
        agreement = 1.0 - statistics.mean(deviations)

        return max(0.0, agreement)

    def _calculate_weighted_confidence(
        self, component_scores: Dict[ConfidenceSource, float]
    ) -> float:
        """Calculate weighted overall confidence"""
        total_weight = 0.0
        weighted_sum = 0.0

        for source, score in component_scores.items():
            weight = self.source_weights.get(source, 0.0)
            weighted_sum += score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def _determine_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Determine confidence level category"""
        if confidence < 0.2:
            return ConfidenceLevel.VERY_LOW
        elif confidence < 0.4:
            return ConfidenceLevel.LOW
        elif confidence < 0.6:
            return ConfidenceLevel.MEDIUM
        elif confidence < 0.8:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH

    async def _calculate_reliability_score(
        self, component_scores: Dict[ConfidenceSource, float]
    ) -> float:
        """Calculate reliability score based on consistency"""
        if not component_scores:
            return 0.0

        values = list(component_scores.values())

        if len(values) < 2:
            return values[0] if values else 0.0

        # Reliability is higher when scores are consistently high
        mean_score = statistics.mean(values)
        variance = statistics.variance(values)

        # Penalize high variance
        reliability = mean_score * math.exp(-variance * 5)

        return min(1.0, reliability)

    async def _calculate_stability_score(self) -> float:
        """Calculate stability score from historical data"""
        if len(self.confidence_history) < 5:
            return 0.5

        # Get recent confidence scores
        recent_scores = [
            a.overall_confidence for a in list(self.confidence_history)[-20:]
        ]

        if len(recent_scores) < 2:
            return 0.5

        # Calculate stability as inverse of variance
        variance = statistics.variance(recent_scores)
        stability = math.exp(-variance * 10)

        return min(1.0, stability)

    async def _generate_improvement_suggestions(
        self, component_scores: Dict[ConfidenceSource, float]
    ) -> List[str]:
        """Generate suggestions for improving confidence"""
        suggestions = []

        # Check each component for improvement opportunities
        for source, score in component_scores.items():
            if score < 0.6:  # Below acceptable threshold
                if source == ConfidenceSource.DATA_QUALITY:
                    suggestions.append(
                        "Improve data quality by increasing sample freshness and consistency"
                    )
                elif source == ConfidenceSource.SAMPLE_SIZE:
                    suggestions.append(
                        "Increase sample size for more reliable estimates"
                    )
                elif source == ConfidenceSource.HISTORICAL_ACCURACY:
                    suggestions.append(
                        "Review and improve prediction accuracy based on historical performance"
                    )
                elif source == ConfidenceSource.MODEL_STABILITY:
                    suggestions.append(
                        "Improve model stability through better parameter tuning"
                    )
                elif source == ConfidenceSource.TEMPORAL_CONSISTENCY:
                    suggestions.append("Ensure consistent measurements over time")
                elif source == ConfidenceSource.CROSS_VALIDATION:
                    suggestions.append("Implement better cross-validation techniques")

        return suggestions

    async def _identify_risk_factors(
        self, component_scores: Dict[ConfidenceSource, float]
    ) -> List[str]:
        """Identify risk factors based on low confidence scores"""
        risk_factors = []

        # Identify critical risk factors
        for source, score in component_scores.items():
            if score < 0.3:  # Critical threshold
                if source == ConfidenceSource.DATA_QUALITY:
                    risk_factors.append(
                        "Poor data quality may lead to unreliable conclusions"
                    )
                elif source == ConfidenceSource.SAMPLE_SIZE:
                    risk_factors.append(
                        "Insufficient sample size increases uncertainty"
                    )
                elif source == ConfidenceSource.HISTORICAL_ACCURACY:
                    risk_factors.append(
                        "Poor historical accuracy indicates potential systematic errors"
                    )
                elif source == ConfidenceSource.MODEL_STABILITY:
                    risk_factors.append(
                        "Model instability may cause unpredictable results"
                    )

        return risk_factors

    async def _create_low_confidence_assessment(
        self, reason: str
    ) -> ConfidenceAssessment:
        """Create low confidence assessment for error cases"""
        return ConfidenceAssessment(
            overall_confidence=0.1,
            confidence_level=ConfidenceLevel.VERY_LOW,
            component_scores={},
            reliability_score=0.0,
            stability_score=0.0,
            improvement_suggestions=[f"Address issue: {reason}"],
            risk_factors=[f"Critical issue: {reason}"],
        )

    def record_accuracy(self, predicted: float, actual: float):
        """Record accuracy for historical tracking"""
        if actual != 0:
            accuracy = 1.0 - abs(predicted - actual) / abs(actual)
            self.accuracy_history.append(max(0.0, accuracy))

    def record_stability_metric(self, value: float):
        """Record stability metric"""
        self.stability_metrics.append(value)

    def get_confidence_trend(self, periods: int = 20) -> Dict[str, Any]:
        """Get confidence trend analysis"""
        if len(self.confidence_history) < periods:
            return {"trend": "insufficient_data", "slope": 0.0}

        recent_assessments = list(self.confidence_history)[-periods:]
        confidences = [a.overall_confidence for a in recent_assessments]

        # Calculate trend slope
        x_values = list(range(len(confidences)))
        if len(x_values) > 1:
            slope = (
                statistics.correlation(x_values, confidences)
                if len(set(confidences)) > 1
                else 0.0
            )
        else:
            slope = 0.0

        # Determine trend direction
        if slope > 0.1:
            trend = "improving"
        elif slope < -0.1:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "slope": slope,
            "recent_average": statistics.mean(confidences),
            "recent_variance": (
                statistics.variance(confidences) if len(confidences) > 1 else 0.0
            ),
        }


# Global confidence engine instance
confidence_engine = ConfidenceEngine()


# Convenience functions
async def calculate_prediction_confidence(
    prediction_metrics: List[Dict[str, Any]],
) -> ConfidenceAssessment:
    """Calculate confidence for predictions"""
    metrics = []

    for metric_data in prediction_metrics:
        metric = ConfidenceMetric(
            source=ConfidenceSource.MODEL_STABILITY,
            value=metric_data.get("confidence", 0.5),
            weight=metric_data.get("weight", 1.0),
            metadata=metric_data.get("metadata", {}),
        )
        metrics.append(metric)

    return await confidence_engine.calculate_confidence(metrics)


async def calculate_analysis_confidence(
    data_quality: float, sample_size: int, historical_accuracy: float = 0.5
) -> ConfidenceAssessment:
    """Calculate confidence for analysis results"""
    metrics = [
        ConfidenceMetric(ConfidenceSource.DATA_QUALITY, data_quality),
        ConfidenceMetric(ConfidenceSource.SAMPLE_SIZE, min(1.0, sample_size / 50)),
        ConfidenceMetric(ConfidenceSource.HISTORICAL_ACCURACY, historical_accuracy),
    ]

    return await confidence_engine.calculate_confidence(metrics)

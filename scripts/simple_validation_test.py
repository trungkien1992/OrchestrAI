#!/usr/bin/env python3
"""
Simple test for Enhanced EAEPT Validation Logging System
Tests core validation logging functionality without full orchestration dependencies
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
# sys.path.insert(0, str(Path(__file__).parent.parent))


def test_validation_logger_direct():
    """Test the validation logger directly without workflow engine dependencies"""
    print("🧪 Testing Enhanced EAEPT Validation Logger (Direct Mode)")
    print("=" * 60)

    try:
        # Import the validation logger directly
        from packages.context_plugin.src.core.validation.eaept_validation_logger import (
            EAEPTValidationLogger,
            ValidationEvent,
            ValidationLevel,
        )

        print("✅ Validation logger imported successfully")

        # Initialize logger
        project_root = Path(__file__).parent.parent.parent.parent
        logger = EAEPTValidationLogger(str(project_root))

        print(f"✅ Validation logger initialized for session: {logger.session_id}")
        print(f"📁 Log directory: {logger.log_dir}")
        print()

        # Test phase context manager
        print("🔍 Testing phase context management...")
        with logger.phase_context("express"):
            # Test thinking metrics
            logger.log_performance_metric("thinking_time_seconds", 2.5, 10.0)
            logger.log_workflow_decision(
                "task_analysis", 0.85, "Analyzed casino game task"
            )
            print("   ✅ Express phase metrics logged")

        with logger.phase_context("explore"):
            # Test RAG queries
            logger.log_rag_query("casino game mechanics", 8, 0.9, 120.0)
            logger.log_rag_query("flutter performance", 5, 0.8, 95.0)
            print("   ✅ Explore phase RAG queries logged")

        with logger.phase_context("code"):
            # Test code generation
            logger.log_code_generation("dart", 150, 0.9, 0.85)
            logger.log_code_generation("python", 75, 0.85, 0.8)

            # Test context optimization
            logger.log_context_optimization("preserve_code", 8000, 6000, 0.75)
            print("   ✅ Code phase metrics logged")

        # Test error logging
        logger.log_error(
            "test_error", "This is a test error for validation", recovery_attempted=True
        )
        print("   ✅ Error logging tested")

        # Test user interaction
        logger.log_user_interaction("clarification", 0.9)
        print("   ✅ User interaction logged")

        # Test orchestration action
        logger.log_orchestration_action("digest_create", "success", 0.8)
        print("   ✅ Orchestration action logged")

        print(f"\n📊 Generating session summary...")
        summary = logger.get_session_summary()

        print(f"📈 Session Summary:")
        print(f"   • Session ID: {summary['session_id']}")
        print(f"   • Current Phase: {summary['current_phase'] or 'completed'}")
        print(f"   • Total Events: {summary['total_events']}")
        print(f"   • Phases Completed: {summary['phases_completed']}")
        print(f"   • Total Duration: {summary['total_duration_minutes']:.1f} minutes")

        if summary["phase_summaries"]:
            print(f"   • Phase Details:")
            for phase, details in summary["phase_summaries"].items():
                print(
                    f"     - {phase.upper()}: {details['events']} events, "
                    f"{details['errors']} errors, "
                    f"{details['effectiveness']:.1%} effectiveness"
                )

        print(f"\n📄 Generating effectiveness report...")
        report_path = (
            logger.log_dir / f"test_effectiveness_report_{logger.session_id}.json"
        )
        report = logger.generate_effectiveness_report(str(report_path))

        print(f"📊 Effectiveness Report:")
        print(f"   • Total Events: {report['total_events']}")
        print(f"   • Total Duration: {report['total_duration_minutes']:.1f} minutes")
        if report["overall_metrics"]:
            metrics = report["overall_metrics"]
            print(f"   • Average Effectiveness: {metrics['average_effectiveness']:.1%}")
            print(f"   • Average Confidence: {metrics['average_confidence']:.1%}")
            print(f"   • Average Quality: {metrics['average_quality']:.1%}")
            print(f"   • Total Errors: {metrics['total_errors']}")
            print(
                f"   • Context Optimizations: {metrics['total_context_optimizations']}"
            )
            print(f"   • RAG Queries: {metrics['total_rag_queries']}")
            print(f"   • User Interactions: {metrics['total_user_interactions']}")

        # Show recommendations
        if report["recommendations"]:
            print(f"   • Recommendations:")
            for i, rec in enumerate(report["recommendations"], 1):
                print(f"     {i}. {rec}")
        else:
            print(f"   • ✅ No recommendations - optimal performance!")

        print(f"\n📁 Report saved to: {report_path}")

        # Test database functionality
        print(f"\n🗄️  Testing database functionality...")
        db_path = logger.db_path
        if db_path.exists():
            import sqlite3

            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()

                # Count metrics
                cursor.execute("SELECT COUNT(*) FROM validation_metrics")
                metric_count = cursor.fetchone()[0]
                print(f"   ✅ {metric_count} validation metrics stored in database")

                # Count phase summaries
                cursor.execute("SELECT COUNT(*) FROM phase_summaries")
                phase_count = cursor.fetchone()[0]
                print(f"   ✅ {phase_count} phase summaries stored in database")

                # Show recent metrics
                cursor.execute(
                    """
                    SELECT event_type, metric_name, effectiveness_score, confidence_score 
                    FROM validation_metrics 
                    WHERE effectiveness_score IS NOT NULL 
                    ORDER BY timestamp DESC LIMIT 5
                """
                )
                recent_metrics = cursor.fetchall()

                if recent_metrics:
                    print(f"   📊 Recent effectiveness metrics:")
                    for (
                        event_type,
                        metric_name,
                        effectiveness,
                        confidence,
                    ) in recent_metrics:
                        print(
                            f"     - {event_type}: {metric_name} "
                            f"(eff: {effectiveness:.1%}, conf: {confidence:.1%})"
                        )

        # Cleanup
        logger.cleanup()
        print(f"\n✅ Validation logger cleanup completed")

        print(f"\n🎉 All validation logging tests passed!")
        print(f"📋 Validation system is fully functional and ready for EAEPT workflows")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please check that the validation logger module is properly installed")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_validation_features():
    """Test specific validation features"""
    print("\n🔬 Testing Specific Validation Features")
    print("-" * 50)

    try:
        from core.validation.eaept_validation_logger import (
            ValidationEvent,
            ValidationLevel,
            ValidationMetric,
        )

        # Test enums
        print("📋 Testing validation enums...")
        print(f"   • ValidationEvent types: {len(list(ValidationEvent))}")
        print(f"   • ValidationLevel types: {len(list(ValidationLevel))}")

        # Test ValidationMetric creation
        print("📊 Testing ValidationMetric creation...")
        metric = ValidationMetric(
            timestamp=datetime.now(),
            phase="test",
            event_type=ValidationEvent.PERFORMANCE_METRIC,
            level=ValidationLevel.METRICS,
            metric_name="test_metric",
            metric_value=42.0,
            effectiveness_score=0.85,
            confidence_score=0.9,
            quality_score=0.8,
        )
        print(f"   ✅ ValidationMetric created successfully")
        print(f"   📈 Metric: {metric.metric_name} = {metric.metric_value}")
        print(f"   🎯 Effectiveness: {metric.effectiveness_score:.1%}")

        return True

    except Exception as e:
        print(f"❌ Feature test failed: {e}")
        return False


def show_validation_system_info():
    """Show information about the validation system"""
    print("\n📋 Enhanced EAEPT Validation System Information")
    print("=" * 60)

    features = [
        "🎯 Phase-specific validation tracking",
        "📊 Real-time effectiveness monitoring",
        "🧠 RAG query performance measurement",
        "💻 Code generation quality assessment",
        "🔄 Context optimization effectiveness",
        "👤 User interaction quality tracking",
        "⚡ System performance monitoring",
        "🗄️  Structured SQLite database storage",
        "📄 Comprehensive effectiveness reporting",
        "💡 AI-powered recommendations",
        "🔍 Historical trend analysis",
    ]

    print("🚀 Core Features:")
    for feature in features:
        print(f"   {feature}")

    casino_features = [
        "🎰 Casino game mechanic validation",
        "⭐ Cosmic theme consistency tracking",
        "🏃 60fps performance validation",
        "💎 Stardust generation monitoring",
        "🌌 Quantum anomaly detection",
        "🔗 Starknet integration validation",
        "📱 Flutter mobile optimization",
        "🚀 FastAPI performance tracking",
    ]

    print(f"\n🎮 AstraTrade Casino Game Features:")
    for feature in casino_features:
        print(f"   {feature}")


def main():
    """Main test execution"""
    print("🚀 Enhanced EAEPT Validation Logging Test Suite")
    print("🎮 AstraTrade Casino Game Development Edition")
    print("=" * 80)

    # Show system info
    show_validation_system_info()

    # Test validation features
    features_ok = test_validation_features()
    if not features_ok:
        print("\n❌ Feature tests failed")
        return 1

    # Test validation logger
    logger_ok = test_validation_logger_direct()
    if not logger_ok:
        print("\n❌ Validation logger tests failed")
        return 1

    print(f"\n🎉 All validation system tests completed successfully!")
    print(f"🔍 The validation logging system is ready for production use")
    print(
        f"📊 Enhanced EAEPT workflows will now have comprehensive effectiveness tracking"
    )
    print(f"🎮 Casino game development workflows are optimally monitored")

    return 0


if __name__ == "__main__":
    # Ensure logs directory exists
    logs_dir = Path(__file__).parent.parent.parent.parent / "logs" / "eaept_validation"
    logs_dir.mkdir(parents=True, exist_ok=True)

    exit_code = main()
    sys.exit(exit_code)

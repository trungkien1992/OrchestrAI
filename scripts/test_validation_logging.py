#!/usr/bin/env python3
"""
Test script for Enhanced EAEPT Validation Logging System
Demonstrates comprehensive validation logging and effectiveness tracking
"""

import asyncio
import json
import sys
import os
from pathlib import Path

try:
    from packages.context_plugin.src.workflow.eaept_engine import EAEPTWorkflowEngine
    from packages.context_plugin.src.core.validation.eaept_validation_logger import (
        EAEPTValidationLogger,
        ValidationEvent,
        ValidationLevel,
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you're running from the correct directory")
    sys.exit(1)


async def test_validation_logging():
    """Test the validation logging system with a sample EAEPT workflow"""
    print("🧪 Testing Enhanced EAEPT Validation Logging System")
    print("=" * 60)

    # Initialize workflow engine with validation enabled
    project_root = Path(__file__).parent.parent.parent.parent
    workflow = EAEPTWorkflowEngine(str(project_root), enable_validation=True)

    if not workflow.validation_logger:
        print("❌ Validation logging not available")
        return

    print(
        f"✅ Validation logger initialized for session: {workflow.validation_logger.session_id}"
    )
    print()

    # Test task: Implement a cosmic trading interface
    test_task = "Implement a cosmic casino trading interface with real-time stardust generation and quantum anomaly detection"

    try:
        print(f"🚀 Starting EAEPT workflow with validation logging")
        print(f"📝 Task: {test_task}")
        print()

        # Execute the full workflow
        result = await workflow.start_workflow(test_task, auto_execute=True)

        print("\n" + "=" * 60)
        print("🎯 Workflow Execution Complete!")
        print("=" * 60)

        # Display basic workflow results
        print(
            f"✅ Status: {result.get('workflow_summary', {}).get('phases_completed', 0)} phases completed"
        )
        print(
            f"⏱️  Duration: {result.get('workflow_summary', {}).get('total_duration_minutes', 0):.1f} minutes"
        )
        print(
            f"🎯 Confidence: {result.get('workflow_summary', {}).get('average_confidence', 0):.1%}"
        )
        print(
            f"💎 Quality: {result.get('workflow_summary', {}).get('average_quality', 0):.1%}"
        )

        # Display validation metrics if available
        if result.get("workflow_summary", {}).get("validation_available"):
            validation = result["workflow_summary"]
            print(
                f"📊 Validation Events: {validation.get('total_validation_events', 0)}"
            )
            print(
                f"🎯 Validation Effectiveness: {validation.get('validation_effectiveness', 0):.1%}"
            )
            print(
                f"💎 Validation Quality: {validation.get('validation_quality', 0):.1%}"
            )

        print()

        # Generate detailed validation report
        if workflow.validation_logger:
            print("📋 Generating Detailed Validation Report...")

            # Get session summary
            session_summary = workflow.validation_logger.get_session_summary()
            print(f"📊 Session Summary:")
            print(f"   • Session ID: {session_summary['session_id']}")
            print(f"   • Total Events: {session_summary['total_events']}")
            print(f"   • Phases Completed: {session_summary['phases_completed']}")
            print(
                f"   • Total Duration: {session_summary['total_duration_minutes']:.1f} minutes"
            )

            # Show phase-specific metrics
            print(f"\n📈 Phase Performance:")
            for phase_name, phase_data in session_summary.get(
                "phase_summaries", {}
            ).items():
                print(
                    f"   • {phase_name.upper()}: {phase_data['events']} events, "
                    f"{phase_data['errors']} errors, "
                    f"{phase_data['effectiveness']:.1%} effectiveness"
                )

            # Generate and save comprehensive report
            report_path = (
                workflow.validation_logger.log_dir
                / f"test_report_{workflow.validation_logger.session_id}.json"
            )
            validation_report = (
                workflow.validation_logger.generate_effectiveness_report(
                    str(report_path)
                )
            )

            print(f"\n📄 Comprehensive validation report saved to:")
            print(f"    {report_path}")

            # Show recommendations
            recommendations = validation_report.get("recommendations", [])
            if recommendations:
                print(f"\n💡 Recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"   {i}. {rec}")
            else:
                print(f"\n✅ No recommendations - workflow performed optimally!")

            # Test specific validation logging features
            print(f"\n🔬 Testing Specific Validation Features...")

            # Test manual effectiveness logging
            workflow.validation_logger.log_metric(
                ValidationEvent.EFFECTIVENESS,
                ValidationLevel.METRICS,
                "manual_test_metric",
                "test_value",
                effectiveness_score=0.95,
                confidence_score=0.9,
                quality_score=0.85,
                context={"test_type": "manual_validation_test"},
            )
            print(f"   ✅ Manual effectiveness metric logged")

            # Test error logging
            workflow.validation_logger.log_error(
                "test_error",
                "This is a test error for validation system testing",
                recovery_attempted=True,
            )
            print(f"   ✅ Test error logged (for testing purposes)")

            # Test performance metric
            workflow.validation_logger.log_performance_metric(
                "test_performance_metric", 2.5, 5.0  # Target value
            )
            print(f"   ✅ Performance metric logged")

            # Cleanup and generate final report
            workflow.validation_logger.cleanup()
            print(f"   ✅ Validation session cleaned up")

        print(f"\n🎉 Test completed successfully!")
        print(f"📁 All validation logs and reports are available in:")
        print(f"    {workflow.validation_logger.log_dir}")

        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_validation_report_analysis():
    """Test validation report analysis capabilities"""
    print("\n🔍 Testing Validation Report Analysis")
    print("-" * 40)

    # Create a standalone validation logger for testing
    project_root = Path(__file__).parent.parent.parent.parent
    logger = EAEPTValidationLogger(str(project_root))

    # Simulate some validation events
    with logger.phase_context("test_analysis"):
        # Simulate different types of events
        logger.log_rag_query("test query for analysis", 10, 0.9, 150.0)
        logger.log_code_generation("python", 200, 0.85, 0.8)
        logger.log_context_optimization("preserve_analysis", 10000, 8000, 0.8)
        logger.log_performance_metric("analysis_speed", 2.3, 3.0)
        logger.log_user_interaction("analysis_feedback", 0.9)

    # Generate analysis report
    report = logger.generate_effectiveness_report()

    print(f"📊 Analysis Report Generated:")
    print(f"   • Total Events: {report['total_events']}")
    print(
        f"   • Overall Effectiveness: {report['overall_metrics']['average_effectiveness']:.1%}"
    )
    print(f"   • Overall Quality: {report['overall_metrics']['average_quality']:.1%}")
    print(
        f"   • Context Optimizations: {report['overall_metrics']['total_context_optimizations']}"
    )
    print(f"   • RAG Queries: {report['overall_metrics']['total_rag_queries']}")

    logger.cleanup()
    print(f"✅ Analysis test completed")


def print_validation_capabilities():
    """Print information about validation logging capabilities"""
    print("\n📋 Enhanced EAEPT Validation Logging Capabilities")
    print("=" * 60)

    capabilities = [
        "🎯 Phase-specific effectiveness tracking",
        "📊 Real-time performance metrics collection",
        "🧠 RAG query performance and relevance tracking",
        "💻 Code generation quality and coverage metrics",
        "🔄 Context optimization effectiveness measurement",
        "👤 User interaction quality assessment",
        "⚡ System performance baseline and monitoring",
        "🗄️  SQLite database for structured metrics storage",
        "📄 Comprehensive effectiveness report generation",
        "💡 Intelligent recommendations based on metrics",
        "🔍 Session-based tracking and analysis",
        "📈 Historical performance trend analysis",
    ]

    for capability in capabilities:
        print(f"   {capability}")

    print(f"\n🎮 AstraTrade Casino Game Specific Features:")
    casino_features = [
        "🎰 Casino game mechanic validation tracking",
        "⭐ Cosmic theme consistency measurement",
        "🏃 60fps performance target validation",
        "💎 Stardust generation efficiency metrics",
        "🌌 Quantum anomaly detection accuracy",
        "🔗 Starknet blockchain integration validation",
        "📱 Flutter mobile performance tracking",
        "🚀 FastAPI backend response time monitoring",
    ]

    for feature in casino_features:
        print(f"   {feature}")


async def main():
    """Main test execution"""
    print("🚀 Enhanced EAEPT Validation Logging Test Suite")
    print("🎮 Optimized for AstraTrade Casino Game Development")
    print("=" * 80)

    # Show capabilities
    print_validation_capabilities()

    # Run main validation test
    success = await test_validation_logging()

    if success:
        # Run additional analysis test
        await test_validation_report_analysis()

        print(f"\n🎉 All validation logging tests completed successfully!")
        print(
            f"🔍 The validation system is ready for Enhanced EAEPT workflow monitoring"
        )
        print(f"📊 Effectiveness tracking will provide insights into AI work quality")
        print(f"🎮 Casino game development workflows will be optimally monitored")

        return 0
    else:
        print(f"\n❌ Validation logging tests failed")
        return 1


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    logs_dir = Path(__file__).parent.parent.parent.parent / "logs" / "eaept_validation"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Run tests
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

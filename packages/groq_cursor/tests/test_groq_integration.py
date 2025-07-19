#!/usr/bin/env python3
"""
Test script for Cursor Agent Groq Integration
Demonstrates various use cases and performance metrics
"""

import asyncio
import time
import json
import os
from pathlib import Path


# Load environment variables from config
def load_groq_env():
    config_dir = Path.home() / ".cursor" / "groq"
    env_file = config_dir / ".env"
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value


# Load environment before importing the tool
load_groq_env()

from groq_reasoning_tool import (
    reason_about_code,
    analyze_problem,
    generate_implementation_plan,
    debug_assistance,
    check_groq_health,
)


async def test_health_check():
    """Test Groq API health and connectivity."""
    print("🔍 Testing Groq API Health...")
    print("-" * 40)

    health = await check_groq_health()

    if health["status"] == "healthy":
        print(f"✅ Groq API is healthy!")
        print(f"   Response time: {health['response_time_seconds']:.2f}s")
        print(f"   Model: {health['model']}")
        return True
    else:
        print(f"❌ Groq API health check failed:")
        print(f"   Error: {health['error']}")
        return False


async def test_code_reasoning():
    """Test code analysis and reasoning capabilities."""
    print("\n🧠 Testing Code Reasoning...")
    print("-" * 40)

    code_context = """
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    
    def fibonacci_memo(n, memo={}):
        if n in memo:
            return memo[n]
        if n <= 1:
            return n
        memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
        return memo[n]
    """

    question = "Compare the performance characteristics of these two Fibonacci implementations. Which one is better and why?"

    start_time = time.time()
    result = await reason_about_code(code_context, question)
    duration = time.time() - start_time

    print(f"⏱️  Total time: {duration:.2f}s")
    print(f"📊 Tokens used: {result.get('tokens_used', 'N/A')}")
    print(f"🤖 Model: {result.get('model_used', 'N/A')}")
    print(f"\n💭 Analysis:\n{result.get('answer', 'No answer received')}")


async def test_problem_analysis():
    """Test complex problem breakdown capabilities."""
    print("\n🔍 Testing Problem Analysis...")
    print("-" * 40)

    problem = "Design a real-time collaborative code editor with WebSocket support"
    context = "Must support multiple users, handle conflicts, and provide real-time syntax highlighting"

    start_time = time.time()
    result = await analyze_problem(problem, context)
    duration = time.time() - start_time

    print(f"⏱️  Total time: {duration:.2f}s")
    print(f"📊 Tokens used: {result.get('tokens_used', 'N/A')}")
    print(f"🤖 Model: {result.get('model_used', 'N/A')}")
    print(f"\n📋 Analysis:\n{result.get('analysis', 'No analysis received')}")


async def test_implementation_planning():
    """Test implementation planning capabilities."""
    print("\n📋 Testing Implementation Planning...")
    print("-" * 40)

    requirements = "Add OAuth2 authentication to an existing Flask REST API"
    constraints = "Must maintain backward compatibility, support Google and GitHub OAuth, and handle token refresh"

    start_time = time.time()
    result = await generate_implementation_plan(requirements, constraints)
    duration = time.time() - start_time

    print(f"⏱️  Total time: {duration:.2f}s")
    print(f"📊 Tokens used: {result.get('tokens_used', 'N/A')}")
    print(f"🤖 Model: {result.get('model_used', 'N/A')}")
    print(f"\n📝 Implementation Plan:\n{result.get('plan', 'No plan received')}")


async def test_debug_assistance():
    """Test debugging assistance capabilities."""
    print("\n🐛 Testing Debug Assistance...")
    print("-" * 40)

    error_message = "ConnectionTimeoutError: Failed to establish connection to database after 30 seconds"
    code_snippet = """
    import psycopg2
    
    def connect_to_db():
        return psycopg2.connect(
            host='localhost',
            database='myapp',
            user='postgres',
            password='password',
            connect_timeout=30
        )
    """
    context = "Python 3.9, PostgreSQL 13, running in Docker container"

    start_time = time.time()
    result = await debug_assistance(error_message, code_snippet, context)
    duration = time.time() - start_time

    print(f"⏱️  Total time: {duration:.2f}s")
    print(f"📊 Tokens used: {result.get('tokens_used', 'N/A')}")
    print(f"🤖 Model: {result.get('model_used', 'N/A')}")
    print(
        f"\n🔧 Debug Analysis:\n{result.get('debug_analysis', 'No analysis received')}"
    )


async def performance_benchmark():
    """Run a performance benchmark comparing different reasoning tasks."""
    print("\n⚡ Performance Benchmark...")
    print("-" * 40)

    tasks = [
        (
            "Code Analysis",
            lambda: reason_about_code("print('Hello')", "What does this do?"),
        ),
        ("Problem Analysis", lambda: analyze_problem("Design a simple cache")),
        (
            "Implementation Plan",
            lambda: generate_implementation_plan("Add logging to API"),
        ),
        (
            "Debug Help",
            lambda: debug_assistance("ImportError", "import nonexistent", "Python 3.9"),
        ),
    ]

    results = []

    for task_name, task_func in tasks:
        print(f"Running {task_name}...")
        start_time = time.time()
        result = await task_func()
        duration = time.time() - start_time

        results.append(
            {
                "task": task_name,
                "duration": duration,
                "tokens": result.get("tokens_used", 0),
                "status": "success" if "error" not in result else "error",
            }
        )

        print(f"   ✅ {duration:.2f}s ({result.get('tokens_used', 0)} tokens)")

    # Summary
    print(f"\n📊 Benchmark Summary:")
    print(f"   Total tasks: {len(results)}")
    print(f"   Average time: {sum(r['duration'] for r in results) / len(results):.2f}s")
    print(f"   Total tokens: {sum(r['tokens'] for r in results)}")
    print(
        f"   Success rate: {sum(1 for r in results if r['status'] == 'success')}/{len(results)}"
    )


async def main():
    """Run all tests."""
    print("🚀 Cursor Agent Groq Integration Test Suite")
    print("=" * 60)

    # Test health first
    if not await test_health_check():
        print(
            "\n❌ Health check failed. Please check your API key and network connection."
        )
        return

    # Run all tests
    await test_code_reasoning()
    await test_problem_analysis()
    await test_implementation_planning()
    await test_debug_assistance()
    await performance_benchmark()

    print("\n🎉 All tests completed!")
    print("\n💡 Integration is ready for use with Cursor Agent!")


if __name__ == "__main__":
    asyncio.run(main())

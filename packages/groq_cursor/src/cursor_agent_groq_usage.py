#!/usr/bin/env python3
"""
Cursor Agent Groq Integration - Direct Import Usage
Practical examples of how to use Groq's fast reasoning in Cursor Agent workflows
"""

import asyncio
import os
from pathlib import Path


# Load Groq environment variables
def load_groq_env():
    config_dir = Path.home() / ".cursor" / "groq"
    env_file = config_dir / ".env"
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value


# Load environment before importing
load_groq_env()

# Import Groq reasoning functions
from groq_reasoning_tool import (
    reason_about_code,
    analyze_problem,
    generate_implementation_plan,
    debug_assistance,
    check_groq_health,
)


class CursorAgentGroqHelper:
    """
    Helper class for Cursor Agent to use Groq's fast reasoning capabilities.
    """

    def __init__(self):
        self.health_status = None

    async def ensure_groq_available(self):
        """Ensure Groq API is available before using."""
        if self.health_status is None:
            self.health_status = await check_groq_health()

        if self.health_status["status"] != "healthy":
            raise Exception(
                f"Groq API not available: {self.health_status.get('error', 'Unknown error')}"
            )

        return True

    async def analyze_code_security(self, code_context: str) -> dict:
        """
        Analyze code for security vulnerabilities using Groq.

        Args:
            code_context: The code to analyze

        Returns:
            Security analysis with vulnerabilities and recommendations
        """
        await self.ensure_groq_available()

        question = """Analyze this code for security vulnerabilities. Consider:
        1. Input validation and sanitization
        2. Authentication and authorization
        3. Data exposure and privacy
        4. Injection attacks (SQL, XSS, etc.)
        5. Cryptographic weaknesses
        6. Resource management issues
        
        Provide specific recommendations for each issue found."""

        result = await reason_about_code(code_context, question)
        return {
            "analysis": result["answer"],
            "response_time": result["response_time_seconds"],
            "tokens_used": result["tokens_used"],
        }

    async def analyze_code_performance(self, code_context: str) -> dict:
        """
        Analyze code for performance issues using Groq.

        Args:
            code_context: The code to analyze

        Returns:
            Performance analysis with bottlenecks and optimizations
        """
        await self.ensure_groq_available()

        question = """Analyze this code for performance issues. Consider:
        1. Time complexity and algorithmic efficiency
        2. Memory usage and potential leaks
        3. I/O operations and blocking calls
        4. Database query optimization
        5. Caching opportunities
        6. Concurrency and threading issues
        
        Provide specific optimization recommendations."""

        result = await reason_about_code(code_context, question)
        return {
            "analysis": result["answer"],
            "response_time": result["response_time_seconds"],
            "tokens_used": result["tokens_used"],
        }

    async def plan_feature_implementation(
        self, feature_description: str, constraints: str = None
    ) -> dict:
        """
        Generate implementation plan for a new feature using Groq.

        Args:
            feature_description: Description of the feature to implement
            constraints: Technical or business constraints

        Returns:
            Detailed implementation plan with timeline and risks
        """
        await self.ensure_groq_available()

        result = await generate_implementation_plan(feature_description, constraints)
        return {
            "plan": result["plan"],
            "response_time": result["response_time_seconds"],
            "tokens_used": result["tokens_used"],
        }

    async def debug_code_issue(
        self, error_message: str, code_snippet: str, context: str = None
    ) -> dict:
        """
        Get debugging assistance for code issues using Groq.

        Args:
            error_message: The error message or stack trace
            code_snippet: The relevant code snippet
            context: Additional context about the environment

        Returns:
            Debug analysis with root cause and solutions
        """
        await self.ensure_groq_available()

        result = await debug_assistance(error_message, code_snippet, context)
        return {
            "debug_analysis": result["debug_analysis"],
            "response_time": result["response_time_seconds"],
            "tokens_used": result["tokens_used"],
        }

    async def analyze_architecture_decision(
        self, problem: str, context: str = None
    ) -> dict:
        """
        Analyze architecture decisions using Groq.

        Args:
            problem: The architectural problem to solve
            context: Additional context and constraints

        Returns:
            Architecture analysis with recommendations
        """
        await self.ensure_groq_available()

        result = await analyze_problem(problem, context)
        return {
            "analysis": result["analysis"],
            "response_time": result["response_time_seconds"],
            "tokens_used": result["tokens_used"],
        }


# Example usage functions for Cursor Agent
async def quick_code_review(code: str) -> str:
    """
    Quick code review using Groq - perfect for Cursor Agent workflows.

    Args:
        code: The code to review

    Returns:
        Review summary
    """
    helper = CursorAgentGroqHelper()

    try:
        # Quick security check
        security_result = await helper.analyze_code_security(code)

        # Quick performance check
        performance_result = await helper.analyze_code_performance(code)

        return f"""
🔍 **Quick Code Review Results**

**Security Analysis:**
{security_result['analysis'][:500]}...

**Performance Analysis:**
{performance_result['analysis'][:500]}...

⏱️ Total time: {security_result['response_time'] + performance_result['response_time']:.2f}s
📊 Total tokens: {security_result['tokens_used'] + performance_result['tokens_used']}
"""
    except Exception as e:
        return f"❌ Code review failed: {str(e)}"


async def plan_new_feature(feature: str, constraints: str = None) -> str:
    """
    Plan implementation of a new feature using Groq.

    Args:
        feature: Feature description
        constraints: Technical constraints

    Returns:
        Implementation plan
    """
    helper = CursorAgentGroqHelper()

    try:
        result = await helper.plan_feature_implementation(feature, constraints)

        return f"""
📋 **Implementation Plan**

{result['plan']}

⏱️ Generated in: {result['response_time']:.2f}s
📊 Tokens used: {result['tokens_used']}
"""
    except Exception as e:
        return f"❌ Planning failed: {str(e)}"


async def debug_issue(error: str, code: str, context: str = None) -> str:
    """
    Debug a code issue using Groq.

    Args:
        error: Error message
        code: Relevant code
        context: Environment context

    Returns:
        Debug analysis
    """
    helper = CursorAgentGroqHelper()

    try:
        result = await helper.debug_code_issue(error, code, context)

        return f"""
🐛 **Debug Analysis**

{result['debug_analysis']}

⏱️ Analyzed in: {result['response_time']:.2f}s
📊 Tokens used: {result['tokens_used']}
"""
    except Exception as e:
        return f"❌ Debug analysis failed: {str(e)}"


# Example: How to use in Cursor Agent workflow
async def example_cursor_agent_workflow():
    """
    Example of how to integrate Groq reasoning into Cursor Agent workflows.
    """
    print("🚀 Cursor Agent Groq Integration - Direct Import Example")
    print("=" * 60)

    # Example 1: Quick code review
    print("\n1️⃣ Quick Code Review Example:")
    print("-" * 40)

    sample_code = """
    def process_user_data(user_input):
        query = f"SELECT * FROM users WHERE id = {user_input}"
        result = db.execute(query)
        return result.fetchall()
    """

    review = await quick_code_review(sample_code)
    print(review)

    # Example 2: Feature planning
    print("\n2️⃣ Feature Planning Example:")
    print("-" * 40)

    feature = "Add real-time notifications to the trading platform"
    constraints = "Must work with existing WebSocket infrastructure and handle 1000+ concurrent users"

    plan = await plan_new_feature(feature, constraints)
    print(plan[:1000] + "...")  # Truncate for display

    # Example 3: Debug assistance
    print("\n3️⃣ Debug Assistance Example:")
    print("-" * 40)

    error = "ConnectionTimeoutError: Failed to establish connection to database"
    code = """
    import psycopg2
    conn = psycopg2.connect(host='localhost', database='mydb')
    """
    context = "Python 3.9, PostgreSQL, Docker environment"

    debug = await debug_issue(error, code, context)
    print(debug[:800] + "...")  # Truncate for display


# Usage in Cursor Agent
async def cursor_agent_groq_reasoning(
    code_context: str, task_type: str, **kwargs
) -> str:
    """
    Main function for Cursor Agent to use Groq reasoning.

    Args:
        code_context: Code or context to analyze
        task_type: Type of reasoning task ('security', 'performance', 'plan', 'debug', 'architecture')
        **kwargs: Additional arguments for specific tasks

    Returns:
        Reasoning result
    """
    helper = CursorAgentGroqHelper()

    try:
        if task_type == "security":
            result = await helper.analyze_code_security(code_context)
            return f"🔒 Security Analysis:\n{result['analysis']}"

        elif task_type == "performance":
            result = await helper.analyze_code_performance(code_context)
            return f"⚡ Performance Analysis:\n{result['analysis']}"

        elif task_type == "plan":
            constraints = kwargs.get("constraints")
            result = await helper.plan_feature_implementation(code_context, constraints)
            return f"📋 Implementation Plan:\n{result['plan']}"

        elif task_type == "debug":
            error = kwargs.get("error", "")
            context = kwargs.get("context")
            result = await helper.debug_code_issue(error, code_context, context)
            return f"🐛 Debug Analysis:\n{result['debug_analysis']}"

        elif task_type == "architecture":
            context = kwargs.get("context")
            result = await helper.analyze_architecture_decision(code_context, context)
            return f"🏗️ Architecture Analysis:\n{result['analysis']}"

        else:
            return f"❌ Unknown task type: {task_type}. Use 'security', 'performance', 'plan', 'debug', or 'architecture'"

    except Exception as e:
        return f"❌ Reasoning failed: {str(e)}"


if __name__ == "__main__":
    # Run the example workflow
    asyncio.run(example_cursor_agent_workflow())

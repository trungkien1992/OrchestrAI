"""
Groq Reasoning Tool for Cursor Agent
Provides fast inference capabilities for reasoning tasks using Groq's API
"""

import json
import os
import time
from typing import Dict, List, Optional, Any
import httpx
import asyncio
from datetime import datetime


class GroqReasoningTool:
    """
    Tool for Cursor Agent to use Groq's fast inference API for reasoning tasks.
    """

    def __init__(self):
        self.api_key = os.getenv(
            "GROQ_API_KEY", "REMOVED_GROQ_KEY"
        )
        self.base_url = "https://api.groq.com"
        self.model = "moonshotai/kimi-k2-instruct"
        self.timeout = 30
        self._session = None

    async def _get_session(self):
        """Get or create HTTP session."""
        if self._session is None:
            self._session = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
        return self._session

    async def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make request to Groq API."""
        session = await self._get_session()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Cursor-Agent-Groq-Tool/1.0.0",
        }

        try:
            response = await session.post(
                f"{self.base_url}/openai/v1/chat/completions",
                headers=headers,
                json=payload,
            )

            if response.status_code != 200:
                raise Exception(
                    f"Groq API error: {response.status_code} - {response.text}"
                )

            return response.json()

        except Exception as e:
            raise Exception(f"Request failed: {str(e)}")

    async def reason_about_code(
        self,
        code_context: str,
        question: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Use Groq to reason about code and answer questions.

        Args:
            code_context: The code or context to reason about
            question: The question to answer
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)

        Returns:
            Reasoning result with answer and metadata
        """
        messages = [
            {
                "role": "system",
                "content": """You are an expert code analyst and reasoning assistant. 
                Analyze the provided code context and answer questions with clear, 
                logical reasoning. Focus on understanding the code structure, 
                identifying potential issues, and providing actionable insights.""",
            },
            {
                "role": "user",
                "content": f"""Code Context:
{code_context}

Question: {question}

Please provide a reasoned analysis and answer to the question.""",
            },
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        start_time = time.time()

        try:
            response = await self._make_request(payload)

            duration = time.time() - start_time

            return {
                "answer": response["choices"][0]["message"]["content"],
                "model_used": response["model"],
                "tokens_used": response["usage"]["total_tokens"],
                "response_time_seconds": duration,
                "timestamp": datetime.utcnow().isoformat(),
                "reasoning_type": "code_analysis",
            }

        except Exception as e:
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}

    async def analyze_problem(
        self,
        problem_description: str,
        context: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Use Groq to analyze and break down complex problems.

        Args:
            problem_description: Description of the problem to analyze
            context: Additional context or constraints
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Problem analysis with breakdown and recommendations
        """
        messages = [
            {
                "role": "system",
                "content": """You are an expert problem solver and systems analyst. 
                Break down complex problems into manageable components, identify 
                key challenges, and provide structured recommendations for solutions.""",
            },
            {
                "role": "user",
                "content": f"""Problem: {problem_description}
                
                {f'Context: {context}' if context else ''}
                
                Please provide a structured analysis including:
                1. Problem breakdown
                2. Key challenges identified
                3. Potential solutions
                4. Recommended approach
                5. Risk considerations""",
            },
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        start_time = time.time()

        try:
            response = await self._make_request(payload)

            duration = time.time() - start_time

            return {
                "analysis": response["choices"][0]["message"]["content"],
                "model_used": response["model"],
                "tokens_used": response["usage"]["total_tokens"],
                "response_time_seconds": duration,
                "timestamp": datetime.utcnow().isoformat(),
                "reasoning_type": "problem_analysis",
            }

        except Exception as e:
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}

    async def generate_implementation_plan(
        self,
        requirements: str,
        constraints: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.4,
    ) -> Dict[str, Any]:
        """
        Use Groq to generate implementation plans for features or systems.

        Args:
            requirements: Feature or system requirements
            constraints: Technical or business constraints
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Implementation plan with steps and considerations
        """
        messages = [
            {
                "role": "system",
                "content": """You are an expert software architect and implementation 
                planner. Create detailed, actionable implementation plans that consider 
                technical feasibility, best practices, and practical constraints.""",
            },
            {
                "role": "user",
                "content": f"""Requirements: {requirements}
                
                {f'Constraints: {constraints}' if constraints else ''}
                
                Please create a detailed implementation plan including:
                1. Architecture overview
                2. Implementation steps
                3. Technology recommendations
                4. Timeline estimates
                5. Risk mitigation strategies
                6. Testing approach""",
            },
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        start_time = time.time()

        try:
            response = await self._make_request(payload)

            duration = time.time() - start_time

            return {
                "plan": response["choices"][0]["message"]["content"],
                "model_used": response["model"],
                "tokens_used": response["usage"]["total_tokens"],
                "response_time_seconds": duration,
                "timestamp": datetime.utcnow().isoformat(),
                "reasoning_type": "implementation_planning",
            }

        except Exception as e:
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}

    async def debug_assistance(
        self,
        error_message: str,
        code_snippet: str,
        context: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Use Groq to assist with debugging by analyzing errors and code.

        Args:
            error_message: The error message or stack trace
            code_snippet: The relevant code snippet
            context: Additional context about the environment
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Debug analysis with potential solutions
        """
        messages = [
            {
                "role": "system",
                "content": """You are an expert debugger and troubleshooting specialist. 
                Analyze error messages and code to identify root causes and provide 
                clear, actionable solutions.""",
            },
            {
                "role": "user",
                "content": f"""Error: {error_message}

Code: {code_snippet}

{f'Context: {context}' if context else ''}

Please provide:
1. Root cause analysis
2. Potential solutions
3. Prevention strategies
4. Code fixes if applicable""",
            },
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        start_time = time.time()

        try:
            response = await self._make_request(payload)

            duration = time.time() - start_time

            return {
                "debug_analysis": response["choices"][0]["message"]["content"],
                "model_used": response["model"],
                "tokens_used": response["usage"]["total_tokens"],
                "response_time_seconds": duration,
                "timestamp": datetime.utcnow().isoformat(),
                "reasoning_type": "debug_assistance",
            }

        except Exception as e:
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}

    async def health_check(self) -> Dict[str, Any]:
        """Check if Groq API is accessible and working."""
        try:
            messages = [{"role": "user", "content": "Hello"}]
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 10,
                "temperature": 0.0,
                "stream": False,
            }

            start_time = time.time()
            response = await self._make_request(payload)
            duration = time.time() - start_time

            return {
                "status": "healthy",
                "response_time_seconds": duration,
                "model": response["model"],
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def close(self):
        """Close the HTTP session."""
        if self._session:
            await self._session.aclose()
            self._session = None


# Global instance for Cursor Agent to use
groq_reasoning_tool = GroqReasoningTool()


# Example usage functions for Cursor Agent
async def reason_about_code(code_context: str, question: str) -> Dict[str, Any]:
    """Reason about code using Groq's fast inference."""
    return await groq_reasoning_tool.reason_about_code(code_context, question)


async def analyze_problem(
    problem_description: str, context: str = None
) -> Dict[str, Any]:
    """Analyze complex problems using Groq's reasoning capabilities."""
    return await groq_reasoning_tool.analyze_problem(problem_description, context)


async def generate_implementation_plan(
    requirements: str, constraints: str = None
) -> Dict[str, Any]:
    """Generate implementation plans using Groq's planning capabilities."""
    return await groq_reasoning_tool.generate_implementation_plan(
        requirements, constraints
    )


async def debug_assistance(
    error_message: str, code_snippet: str, context: str = None
) -> Dict[str, Any]:
    """Get debugging assistance using Groq's analysis capabilities."""
    return await groq_reasoning_tool.debug_assistance(
        error_message, code_snippet, context
    )


async def check_groq_health() -> Dict[str, Any]:
    """Check if Groq API is healthy and accessible."""
    return await groq_reasoning_tool.health_check()


# Cleanup function
async def cleanup_groq_tool():
    """Clean up Groq tool resources."""
    await groq_reasoning_tool.close()


if __name__ == "__main__":
    # Example usage
    async def test_groq_tool():
        # Test health check
        health = await check_groq_health()
        print(f"Health check: {health}")

        # Test code reasoning
        code_context = """
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        """
        result = await reason_about_code(
            code_context,
            "What are the performance implications of this implementation?",
        )
        print(f"Code reasoning result: {result}")

        # Cleanup
        await cleanup_groq_tool()

    # Run test
    asyncio.run(test_groq_tool())

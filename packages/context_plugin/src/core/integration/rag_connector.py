"""
RAG System Integration for Claude Context Management
Connects with existing Claude RAG system for enhanced context and knowledge retrieval
"""

import asyncio
import logging
import json
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RAGQuery:
    """RAG query with metadata"""

    query: str
    max_results: int = 5
    min_similarity: float = 0.25
    collection_name: str = "claude_context_management"
    metadata_filters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls"""
        return {
            "query": self.query,
            "max_results": self.max_results,
            "min_similarity": self.min_similarity,
            "collection_name": self.collection_name,
            "metadata_filters": self.metadata_filters,
        }


@dataclass
class RAGResult:
    """RAG search result"""

    content: str
    similarity: float
    metadata: Dict[str, Any]
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "content": self.content,
            "similarity": self.similarity,
            "metadata": self.metadata,
            "source": self.source,
        }


@dataclass
class RAGDocument:
    """Document for RAG indexing"""

    id: str
    content: str
    metadata: Dict[str, Any]
    collection_name: str = "claude_context_management"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls"""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "collection_name": self.collection_name,
        }


class RAGConnector:
    """Connector for Claude RAG system integration"""

    def __init__(self, base_url: str = "http://127.0.0.1:8001", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "User-Agent": "claude-context-management/1.0",
            }
        )

        # Connection status
        self.is_connected = False
        self.last_health_check = None

    async def connect(self) -> bool:
        """Connect to RAG system and verify health"""
        try:
            # Check if RAG system is available
            response = await asyncio.to_thread(
                self.session.get, f"{self.base_url}/health", timeout=self.timeout
            )

            if response.status_code == 200:
                self.is_connected = True
                self.last_health_check = datetime.now()
                logger.info("✅ Connected to RAG system")
                return True
            else:
                logger.warning(
                    f"RAG system health check failed: {response.status_code}"
                )
                return False

        except Exception as e:
            logger.error(f"Failed to connect to RAG system: {e}")
            self.is_connected = False
            return False

    async def search(self, query: RAGQuery) -> List[RAGResult]:
        """Search RAG system for relevant content"""
        if not self.is_connected:
            await self.connect()

        if not self.is_connected:
            logger.warning("RAG system not available for search")
            return []

        try:
            response = await asyncio.to_thread(
                self.session.post,
                f"{self.base_url}/search",
                json=query.to_dict(),
                timeout=self.timeout,
            )

            if response.status_code == 200:
                data = response.json()
                results = []

                for item in data.get("results", []):
                    result = RAGResult(
                        content=item.get("content", ""),
                        similarity=item.get("similarity", 0.0),
                        metadata=item.get("metadata", {}),
                        source=item.get("source", ""),
                    )
                    results.append(result)

                logger.info(f"🔍 RAG search returned {len(results)} results")
                return results
            else:
                logger.error(
                    f"RAG search failed: {response.status_code} - {response.text}"
                )
                return []

        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return []

    async def index_document(self, document: RAGDocument) -> bool:
        """Index a document in RAG system"""
        if not self.is_connected:
            await self.connect()

        if not self.is_connected:
            logger.warning("RAG system not available for indexing")
            return False

        try:
            response = await asyncio.to_thread(
                self.session.post,
                f"{self.base_url}/index",
                json=document.to_dict(),
                timeout=self.timeout,
            )

            if response.status_code == 200:
                logger.info(f"📚 Document indexed: {document.id}")
                return True
            else:
                logger.error(
                    f"Document indexing failed: {response.status_code} - {response.text}"
                )
                return False

        except Exception as e:
            logger.error(f"Document indexing error: {e}")
            return False

    async def index_batch(self, documents: List[RAGDocument]) -> int:
        """Index multiple documents in batch"""
        if not documents:
            return 0

        successful_indexes = 0

        # Process in batches to avoid overwhelming the system
        batch_size = 10
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            # Index batch concurrently
            tasks = [self.index_document(doc) for doc in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if result is True:
                    successful_indexes += 1
                elif isinstance(result, Exception):
                    logger.error(f"Batch indexing error: {result}")

            # Brief pause between batches
            await asyncio.sleep(0.1)

        logger.info(
            f"📚 Batch indexing complete: {successful_indexes}/{len(documents)} documents indexed"
        )
        return successful_indexes

    async def get_collections(self) -> List[str]:
        """Get list of available collections"""
        if not self.is_connected:
            await self.connect()

        try:
            response = await asyncio.to_thread(
                self.session.get, f"{self.base_url}/collections", timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("collections", [])
            else:
                logger.error(f"Failed to get collections: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error getting collections: {e}")
            return []

    async def delete_document(
        self, document_id: str, collection_name: str = "claude_context_management"
    ) -> bool:
        """Delete a document from RAG system"""
        if not self.is_connected:
            await self.connect()

        try:
            response = await asyncio.to_thread(
                self.session.delete,
                f"{self.base_url}/document/{document_id}",
                json={"collection_name": collection_name},
                timeout=self.timeout,
            )

            if response.status_code == 200:
                logger.info(f"🗑️ Document deleted: {document_id}")
                return True
            else:
                logger.error(f"Document deletion failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Document deletion error: {e}")
            return False


class ContextRAGManager:
    """Manager for context-specific RAG operations"""

    def __init__(self, rag_connector: RAGConnector):
        self.rag = rag_connector
        self.indexed_sessions = set()

    async def index_session_context(
        self, session_id: str, context_data: Dict[str, Any]
    ) -> bool:
        """Index session context in RAG system"""

        # Create document from context data
        doc_id = f"session_{session_id}_{datetime.now().timestamp()}"

        # Generate content summary
        content_parts = []

        if "session_summary" in context_data:
            content_parts.append(f"Session Summary: {context_data['session_summary']}")

        if "key_implementations" in context_data:
            content_parts.append(
                f"Key Implementations: {context_data['key_implementations']}"
            )

        if "technical_decisions" in context_data:
            content_parts.append(
                f"Technical Decisions: {context_data['technical_decisions']}"
            )

        if "performance_metrics" in context_data:
            content_parts.append(
                f"Performance Metrics: {context_data['performance_metrics']}"
            )

        if "recommendations" in context_data:
            content_parts.append(f"Recommendations: {context_data['recommendations']}")

        content = "\n\n".join(content_parts)

        # Create metadata
        metadata = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "type": "session_context",
            "source": "claude_context_management",
        }

        # Add additional metadata from context
        if "session_type" in context_data:
            metadata["session_type"] = context_data["session_type"]

        if "duration" in context_data:
            metadata["duration"] = context_data["duration"]

        if "complexity_score" in context_data:
            metadata["complexity_score"] = context_data["complexity_score"]

        # Create document
        document = RAGDocument(id=doc_id, content=content, metadata=metadata)

        # Index document
        success = await self.rag.index_document(document)

        if success:
            self.indexed_sessions.add(session_id)

        return success

    async def search_similar_sessions(
        self, query: str, session_type: str = None, max_results: int = 5
    ) -> List[RAGResult]:
        """Search for similar sessions"""

        # Build query
        rag_query = RAGQuery(query=query, max_results=max_results, min_similarity=0.3)

        # Add session type filter if specified
        if session_type:
            rag_query.metadata_filters["session_type"] = session_type

        # Search RAG system
        results = await self.rag.search(rag_query)

        # Filter for session contexts
        session_results = [
            result
            for result in results
            if result.metadata.get("type") == "session_context"
        ]

        return session_results

    async def get_implementation_patterns(self, pattern_type: str) -> List[RAGResult]:
        """Get implementation patterns from indexed sessions"""

        query = f"implementation pattern {pattern_type} architecture design"

        rag_query = RAGQuery(query=query, max_results=10, min_similarity=0.25)

        results = await self.rag.search(rag_query)

        # Filter for implementation-related content
        pattern_results = [
            result
            for result in results
            if any(
                keyword in result.content.lower()
                for keyword in ["implementation", "pattern", "architecture", "design"]
            )
        ]

        return pattern_results

    async def get_optimization_insights(self, metric_type: str) -> List[RAGResult]:
        """Get optimization insights from historical data"""

        query = f"optimization performance improvement {metric_type}"

        rag_query = RAGQuery(query=query, max_results=8, min_similarity=0.2)

        results = await self.rag.search(rag_query)

        # Filter for optimization-related content
        optimization_results = [
            result
            for result in results
            if any(
                keyword in result.content.lower()
                for keyword in ["optimization", "performance", "improvement", "faster"]
            )
        ]

        return optimization_results

    async def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """Clean up old session data from RAG system"""

        # This would require querying by date and deleting old documents
        # For now, we'll return 0 as a placeholder
        logger.info(f"Cleanup old sessions older than {days_old} days")
        return 0


# Global RAG connector and manager
rag_connector = RAGConnector()
context_rag_manager = ContextRAGManager(rag_connector)


# Convenience functions
async def search_context(query: str, max_results: int = 5) -> List[RAGResult]:
    """Search context in RAG system"""
    rag_query = RAGQuery(query=query, max_results=max_results)
    return await rag_connector.search(rag_query)


async def index_session_data(session_id: str, data: Dict[str, Any]) -> bool:
    """Index session data in RAG system"""
    return await context_rag_manager.index_session_context(session_id, data)


async def get_similar_sessions(query: str, max_results: int = 5) -> List[RAGResult]:
    """Get similar sessions from RAG system"""
    return await context_rag_manager.search_similar_sessions(
        query, max_results=max_results
    )

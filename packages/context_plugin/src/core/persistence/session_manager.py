"""
Session Persistence and Recovery for Claude Context Management
Advanced session state management with automatic recovery and continuity
"""

import asyncio
import json
import logging
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
import hashlib
import aiofiles
import sqlite3
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class SessionStatus(Enum):
    """Session status types"""

    ACTIVE = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    RECOVERED = auto()


class PersistenceLevel(Enum):
    """Levels of session persistence"""

    MINIMAL = auto()  # Basic session info only
    STANDARD = auto()  # Session info + key metrics
    FULL = auto()  # Complete session state
    COMPREHENSIVE = auto()  # Full state + historical data


@dataclass
class SessionState:
    """Complete session state information"""

    session_id: str
    status: SessionStatus
    created_at: datetime
    updated_at: datetime

    # Core session data
    context_data: Dict[str, Any] = field(default_factory=dict)
    metrics_data: Dict[str, Any] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)

    # Analysis state
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, List[float]] = field(default_factory=dict)

    # Task tracking
    active_tasks: List[str] = field(default_factory=list)
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)

    # Recovery information
    recovery_points: List[datetime] = field(default_factory=list)
    checkpoint_data: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    session_type: str = "general"
    user_id: str = "default"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data["status"] = self.status.value
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        data["recovery_points"] = [rp.isoformat() for rp in self.recovery_points]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        """Create from dictionary"""
        data = data.copy()
        data["status"] = SessionStatus(data["status"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        data["recovery_points"] = [
            datetime.fromisoformat(rp) for rp in data["recovery_points"]
        ]
        return cls(**data)


class SessionPersistenceManager:
    """Manager for session persistence and recovery"""

    def __init__(
        self,
        storage_path: str = "~/.claude/sessions",
        persistence_level: PersistenceLevel = PersistenceLevel.STANDARD,
    ):

        self.storage_path = Path(storage_path).expanduser()
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.persistence_level = persistence_level
        self.db_path = self.storage_path / "sessions.db"

        # In-memory session cache
        self.session_cache: Dict[str, SessionState] = {}

        # Auto-save configuration
        self.auto_save_interval = 30  # seconds
        self.auto_save_enabled = True
        self.auto_save_task = None

        # Initialize database
        asyncio.create_task(self._initialize_database())

    async def _initialize_database(self):
        """Initialize SQLite database for session metadata"""
        async with aiofiles.open(self.db_path, "w+") as f:
            pass  # Create file if it doesn't exist

        # Create tables
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                status TEXT,
                created_at TEXT,
                updated_at TEXT,
                session_type TEXT,
                user_id TEXT,
                file_path TEXT,
                size_bytes INTEGER,
                checksum TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS session_metrics (
                session_id TEXT,
                metric_name TEXT,
                metric_value REAL,
                timestamp TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        """
        )

        conn.commit()
        conn.close()

        logger.info(f"📁 Session database initialized at {self.db_path}")

    async def save_session(self, session_state: SessionState) -> bool:
        """Save session state to persistent storage"""
        try:
            # Update timestamp
            session_state.updated_at = datetime.now()

            # Determine file path
            file_path = self._get_session_file_path(session_state.session_id)

            # Save based on persistence level
            if self.persistence_level == PersistenceLevel.MINIMAL:
                data = {
                    "session_id": session_state.session_id,
                    "status": session_state.status.value,
                    "created_at": session_state.created_at.isoformat(),
                    "updated_at": session_state.updated_at.isoformat(),
                    "session_type": session_state.session_type,
                }
            elif self.persistence_level == PersistenceLevel.STANDARD:
                data = {
                    "session_id": session_state.session_id,
                    "status": session_state.status.value,
                    "created_at": session_state.created_at.isoformat(),
                    "updated_at": session_state.updated_at.isoformat(),
                    "session_type": session_state.session_type,
                    "context_data": session_state.context_data,
                    "metrics_data": session_state.metrics_data,
                    "active_tasks": session_state.active_tasks,
                    "completed_tasks": session_state.completed_tasks,
                }
            else:  # FULL or COMPREHENSIVE
                data = session_state.to_dict()

            # Save to file
            async with aiofiles.open(file_path, "w") as f:
                await f.write(json.dumps(data, indent=2))

            # Update database
            await self._update_session_metadata(session_state, file_path)

            # Update cache
            self.session_cache[session_state.session_id] = session_state

            logger.debug(f"💾 Session saved: {session_state.session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save session {session_state.session_id}: {e}")
            return False

    async def load_session(self, session_id: str) -> Optional[SessionState]:
        """Load session state from persistent storage"""
        try:
            # Check cache first
            if session_id in self.session_cache:
                return self.session_cache[session_id]

            # Load from file
            file_path = self._get_session_file_path(session_id)

            if not file_path.exists():
                logger.warning(f"Session file not found: {session_id}")
                return None

            async with aiofiles.open(file_path, "r") as f:
                data = json.loads(await f.read())

            # Convert back to SessionState
            if self.persistence_level == PersistenceLevel.MINIMAL:
                # Create minimal session state
                session_state = SessionState(
                    session_id=data["session_id"],
                    status=SessionStatus(data["status"]),
                    created_at=datetime.fromisoformat(data["created_at"]),
                    updated_at=datetime.fromisoformat(data["updated_at"]),
                    session_type=data.get("session_type", "general"),
                )
            else:
                session_state = SessionState.from_dict(data)

            # Update cache
            self.session_cache[session_id] = session_state

            logger.debug(f"📂 Session loaded: {session_id}")
            return session_state

        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    async def create_checkpoint(self, session_id: str) -> bool:
        """Create recovery checkpoint for session"""
        try:
            session_state = await self.load_session(session_id)
            if not session_state:
                return False

            # Create checkpoint
            checkpoint_time = datetime.now()
            session_state.recovery_points.append(checkpoint_time)

            # Save checkpoint data
            checkpoint_data = {
                "timestamp": checkpoint_time.isoformat(),
                "context_data": session_state.context_data.copy(),
                "metrics_data": session_state.metrics_data.copy(),
                "active_tasks": session_state.active_tasks.copy(),
                "completed_tasks": session_state.completed_tasks.copy(),
            }

            session_state.checkpoint_data[checkpoint_time.isoformat()] = checkpoint_data

            # Save updated session
            await self.save_session(session_state)

            logger.info(f"🔄 Checkpoint created for session: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create checkpoint for {session_id}: {e}")
            return False

    async def recover_session(
        self, session_id: str, recovery_point: datetime = None
    ) -> Optional[SessionState]:
        """Recover session from checkpoint"""
        try:
            session_state = await self.load_session(session_id)
            if not session_state:
                return None

            # Find recovery point
            if recovery_point is None:
                # Use latest recovery point
                if not session_state.recovery_points:
                    logger.warning(
                        f"No recovery points found for session: {session_id}"
                    )
                    return session_state
                recovery_point = max(session_state.recovery_points)

            # Restore from checkpoint
            checkpoint_key = recovery_point.isoformat()
            if checkpoint_key in session_state.checkpoint_data:
                checkpoint_data = session_state.checkpoint_data[checkpoint_key]

                session_state.context_data = checkpoint_data["context_data"]
                session_state.metrics_data = checkpoint_data["metrics_data"]
                session_state.active_tasks = checkpoint_data["active_tasks"]
                session_state.completed_tasks = checkpoint_data["completed_tasks"]
                session_state.status = SessionStatus.RECOVERED
                session_state.updated_at = datetime.now()

                # Save recovered session
                await self.save_session(session_state)

                logger.info(f"🔄 Session recovered: {session_id} from {recovery_point}")
                return session_state
            else:
                logger.warning(
                    f"Checkpoint not found for recovery point: {recovery_point}"
                )
                return session_state

        except Exception as e:
            logger.error(f"Failed to recover session {session_id}: {e}")
            return None

    async def list_sessions(
        self,
        status: SessionStatus = None,
        session_type: str = None,
        user_id: str = None,
    ) -> List[Dict[str, Any]]:
        """List sessions with optional filtering"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Build query
            query = "SELECT * FROM sessions"
            conditions = []
            params = []

            if status:
                conditions.append("status = ?")
                params.append(status.value)

            if session_type:
                conditions.append("session_type = ?")
                params.append(session_type)

            if user_id:
                conditions.append("user_id = ?")
                params.append(user_id)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY updated_at DESC"

            cursor.execute(query, params)
            rows = cursor.fetchall()

            # Convert to list of dictionaries
            columns = [desc[0] for desc in cursor.description]
            sessions = [dict(zip(columns, row)) for row in rows]

            conn.close()

            return sessions

        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []

    async def delete_session(self, session_id: str) -> bool:
        """Delete session from persistent storage"""
        try:
            # Remove from cache
            if session_id in self.session_cache:
                del self.session_cache[session_id]

            # Remove file
            file_path = self._get_session_file_path(session_id)
            if file_path.exists():
                file_path.unlink()

            # Remove from database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                "DELETE FROM session_metrics WHERE session_id = ?", (session_id,)
            )
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

            conn.commit()
            conn.close()

            logger.info(f"🗑️ Session deleted: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    async def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """Clean up sessions older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)

            # Get old sessions
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                "SELECT session_id FROM sessions WHERE updated_at < ?",
                (cutoff_date.isoformat(),),
            )
            old_sessions = [row[0] for row in cursor.fetchall()]

            conn.close()

            # Delete old sessions
            deleted_count = 0
            for session_id in old_sessions:
                if await self.delete_session(session_id):
                    deleted_count += 1

            logger.info(f"🧹 Cleaned up {deleted_count} old sessions")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")
            return 0

    async def start_auto_save(self):
        """Start automatic session saving"""
        if self.auto_save_task is None:
            self.auto_save_enabled = True
            self.auto_save_task = asyncio.create_task(self._auto_save_loop())
            logger.info("🔄 Auto-save started")

    async def stop_auto_save(self):
        """Stop automatic session saving"""
        self.auto_save_enabled = False
        if self.auto_save_task:
            self.auto_save_task.cancel()
            self.auto_save_task = None
            logger.info("⏹️ Auto-save stopped")

    async def _auto_save_loop(self):
        """Auto-save loop for periodic session saving"""
        while self.auto_save_enabled:
            try:
                # Save all cached sessions
                for session_id, session_state in self.session_cache.items():
                    await self.save_session(session_state)

                # Wait for next interval
                await asyncio.sleep(self.auto_save_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-save error: {e}")
                await asyncio.sleep(self.auto_save_interval)

    def _get_session_file_path(self, session_id: str) -> Path:
        """Get file path for session"""
        return self.storage_path / f"{session_id}.json"

    async def _update_session_metadata(
        self, session_state: SessionState, file_path: Path
    ):
        """Update session metadata in database"""
        try:
            # Calculate file size and checksum
            file_size = file_path.stat().st_size

            with open(file_path, "rb") as f:
                checksum = hashlib.md5(f.read()).hexdigest()

            # Update database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO sessions 
                (session_id, status, created_at, updated_at, session_type, user_id, file_path, size_bytes, checksum)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    session_state.session_id,
                    session_state.status.value,
                    session_state.created_at.isoformat(),
                    session_state.updated_at.isoformat(),
                    session_state.session_type,
                    session_state.user_id,
                    str(file_path),
                    file_size,
                    checksum,
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to update session metadata: {e}")

    async def get_session_statistics(self) -> Dict[str, Any]:
        """Get session statistics"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Total sessions
            cursor.execute("SELECT COUNT(*) FROM sessions")
            total_sessions = cursor.fetchone()[0]

            # Sessions by status
            cursor.execute("SELECT status, COUNT(*) FROM sessions GROUP BY status")
            status_counts = dict(cursor.fetchall())

            # Sessions by type
            cursor.execute(
                "SELECT session_type, COUNT(*) FROM sessions GROUP BY session_type"
            )
            type_counts = dict(cursor.fetchall())

            # Storage usage
            cursor.execute("SELECT SUM(size_bytes) FROM sessions")
            total_size = cursor.fetchone()[0] or 0

            conn.close()

            return {
                "total_sessions": total_sessions,
                "status_counts": status_counts,
                "type_counts": type_counts,
                "total_storage_bytes": total_size,
                "cached_sessions": len(self.session_cache),
            }

        except Exception as e:
            logger.error(f"Failed to get session statistics: {e}")
            return {}


# Global session manager
session_manager = SessionPersistenceManager()


# Convenience functions
async def save_current_session(session_id: str, context_data: Dict[str, Any]) -> bool:
    """Save current session state"""
    session_state = SessionState(
        session_id=session_id,
        status=SessionStatus.ACTIVE,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        context_data=context_data,
    )
    return await session_manager.save_session(session_state)


async def load_session_state(session_id: str) -> Optional[SessionState]:
    """Load session state"""
    return await session_manager.load_session(session_id)


async def create_session_checkpoint(session_id: str) -> bool:
    """Create session checkpoint"""
    return await session_manager.create_checkpoint(session_id)


async def recover_from_checkpoint(session_id: str) -> Optional[SessionState]:
    """Recover session from latest checkpoint"""
    return await session_manager.recover_session(session_id)

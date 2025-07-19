from fastapi import FastAPI, HTTPException, Query, Header, Depends, status
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import asyncio
from datetime import datetime

# Import orchestration and session management
from core.orchestration.orchestrator_engine import OrchestrationEngine
from core.persistence.session_manager import SessionPersistenceManager, SessionStatus, SessionState

app = FastAPI(title="Claude Workflow Orchestration API", version="0.3.0")

# --- Authentication ---
API_KEY = os.getenv("CLAUDE_API_KEY", "changeme")
API_KEY_NAME = "X-API-Key"

def get_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
    return x_api_key

# --- Initialize core engine and session manager ---
orchestrator = OrchestrationEngine()
session_manager = SessionPersistenceManager()

# --- Models ---
class OrchestrateRequest(BaseModel):
    user_intent: Optional[str] = ""
    force_command: Optional[str] = None

class OrchestrateResponse(BaseModel):
    status: str
    command_executed: str
    parameters: Dict[str, Any]
    result: Dict[str, Any]
    recommendations: Optional[List[str]] = None
    metrics: Optional[Dict[str, Any]] = None

class SessionCreateRequest(BaseModel):
    session_type: Optional[str] = None
    user_id: Optional[str] = None
    tags: Optional[List[str]] = None
    context_data: Optional[Dict[str, Any]] = None

class SessionUpdateRequest(BaseModel):
    status: Optional[str] = None
    context_data: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None

# --- Endpoints ---
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/orchestrate", response_model=OrchestrateResponse, dependencies=[Depends(get_api_key)])
async def orchestrate(req: OrchestrateRequest):
    try:
        result = await orchestrator.orchestrate(
            user_intent=req.user_intent, force_command=req.force_command
        )
        return OrchestrateResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session", dependencies=[Depends(get_api_key)])
async def list_sessions(status: Optional[str] = Query(None)):
    try:
        status_enum = SessionStatus[status.upper()] if status else None
        sessions = await session_manager.list_sessions(status=status_enum)
        return {"sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/session", dependencies=[Depends(get_api_key)])
async def create_session(req: SessionCreateRequest):
    try:
        session_id = f"session_{os.urandom(4).hex()}"
        now = datetime.now()
        session_state = SessionState(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            session_type=req.session_type or "general",
            user_id=req.user_id or "default",
            tags=req.tags or [],
            context_data=req.context_data or {},
        )
        ok = await session_manager.save_session(session_state)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to save session")
        return {"session_id": session_id, "session": session_state.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}", dependencies=[Depends(get_api_key)])
async def get_session(session_id: str):
    try:
        session = await session_manager.load_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"session": session.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/session/{session_id}", dependencies=[Depends(get_api_key)])
async def update_session(session_id: str, req: SessionUpdateRequest):
    try:
        session = await session_manager.load_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        if req.status:
            session.status = SessionStatus[req.status.upper()]
        if req.context_data:
            session.context_data.update(req.context_data)
        if req.tags:
            session.tags = req.tags
        session.updated_at = datetime.now()
        ok = await session_manager.save_session(session)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to update session")
        return {"session": session.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/session/{session_id}", dependencies=[Depends(get_api_key)])
async def delete_session(session_id: str):
    try:
        ok = await session_manager.delete_session(session_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Session not found or could not be deleted")
        return {"deleted": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
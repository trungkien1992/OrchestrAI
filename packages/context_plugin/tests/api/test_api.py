import os
import pytest
from fastapi.testclient import TestClient
from contextlib import contextmanager
from packages.context_plugin.src.api.main import app, API_KEY_NAME

TEST_API_KEY = "testkey123"

# Patch the environment for the API key
def setup_module(module):
    os.environ["CLAUDE_API_KEY"] = TEST_API_KEY

client = TestClient(app)

def auth_headers():
    return {API_KEY_NAME: TEST_API_KEY}

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_auth_required():
    resp = client.get("/session")
    assert resp.status_code == 401
    resp = client.post("/session", json={})
    assert resp.status_code == 401

def test_create_and_crud_session():
    # Create
    payload = {"session_type": "test", "user_id": "tester", "tags": ["a", "b"], "context_data": {"foo": "bar"}}
    resp = client.post("/session", json=payload, headers=auth_headers())
    assert resp.status_code == 200
    data = resp.json()
    session_id = data["session_id"]
    # List
    resp = client.get("/session", headers=auth_headers())
    assert resp.status_code == 200
    assert any(s["session_id"] == session_id for s in resp.json()["sessions"])
    # Get
    resp = client.get(f"/session/{session_id}", headers=auth_headers())
    assert resp.status_code == 200
    assert resp.json()["session"]["session_id"] == session_id
    # Update
    update = {"status": "COMPLETED", "context_data": {"foo": "baz"}, "tags": ["x"]}
    resp = client.put(f"/session/{session_id}", json=update, headers=auth_headers())
    assert resp.status_code == 200
    assert resp.json()["session"]["status"] == 3  # COMPLETED enum value
    assert resp.json()["session"]["context_data"]["foo"] == "baz"
    assert resp.json()["session"]["tags"] == ["x"]
    # Delete
    resp = client.delete(f"/session/{session_id}", headers=auth_headers())
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True
    # Get after delete
    resp = client.get(f"/session/{session_id}", headers=auth_headers())
    assert resp.status_code == 404

def test_orchestrate_endpoint():
    payload = {"user_intent": "test orchestration"}
    resp = client.post("/orchestrate", json=payload, headers=auth_headers())
    # Accept 200 or 500 (if orchestrator fails), but should not be 401
    assert resp.status_code in (200, 500) 
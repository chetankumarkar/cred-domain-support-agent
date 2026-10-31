"""FastAPI HTTP and WebSocket service with redacted JSONL audit logging."""
from __future__ import annotations
import json, logging, time, uuid, os
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel, Field
# The deployed API wraps the same CrewAI crew exercised by the demo. MOCK_LLM
# remains default, so these agent calls are deterministic and keyless.
os.environ.setdefault("USE_CREWAI_CREW","1")
from .crew import run_support, RequestBudgetError
from .guardrails import validate_input, GuardrailError, mask_pii
from .rag import KB_DIR, build_chroma_indexes

LOG_PATH=Path(__file__).resolve().parents[1]/"evidence"/"requests.jsonl"
LOG_PATH.parent.mkdir(parents=True,exist_ok=True)
app=FastAPI(title="Cred Domain Support Agent",version="1.0.0")
class AskRequest(BaseModel): query:str=Field(min_length=1); session_id:str="default"
class AskResponse(BaseModel): answer:str; intent:str; sources:list[str]; application:dict|None=None; escalated:bool=False; trace_id:str
class AddDocumentRequest(BaseModel): document_id:str=Field(pattern=r"^[a-z0-9-]{2,64}$"); title:str=Field(min_length=2,max_length=100); content:str=Field(min_length=20,max_length=10000)

def log_request(trace_id:str, query:str, elapsed_ms:float, status:str):
    safe,_=mask_pii(query)
    row={"timestamp":datetime.now(timezone.utc).isoformat(),"trace_id":trace_id,"duration_ms":round(elapsed_ms,2),"status":status,"request_text":safe}
    with LOG_PATH.open("a",encoding="utf-8") as f: f.write(json.dumps(row,ensure_ascii=False)+"\n")

@app.post("/ask",response_model=AskResponse)
def ask(req:AskRequest):
    trace=uuid.uuid4().hex; started=time.perf_counter(); status="ok"
    try:
        # Guard before recording or orchestration; raw request text is never sent to the logger.
        result=run_support(req.query,req.session_id,trace)
        return AskResponse(**result.model_dump())
    except (GuardrailError,RequestBudgetError) as e:
        status=getattr(e,"code","BUDGET_REJECTED")
        raise HTTPException(status_code=400,detail={"code":status,"message":str(e),"trace_id":trace})
    finally: log_request(trace,req.query,(time.perf_counter()-started)*1000,status)

@app.post("/add-document")
def add_document(req:AddDocumentRequest):
    trace=uuid.uuid4().hex; started=time.perf_counter(); status="ok"
    try:
        path=KB_DIR/f"{req.document_id}.md"
        if path.exists():
            status="conflict"; raise HTTPException(409,"document id already exists")
        path.write_text(f"# {req.title}\n{req.content.strip()}\n",encoding="utf-8")
        indexed=None
        note="Available to subsequent lexical retrieval calls."
        if os.getenv("USE_VECTOR_STORE","0")=="1":
            try:
                build_chroma_indexes(); indexed=True
                note="Saved and upserted into both vector collections."
            except Exception:
                status="index_update_failed"; indexed=False
                note="Saved, but vector indexing failed. Check that the embedding model is installed locally and rerun python -m cred_agent.index."
        return {"added":True,"document_id":req.document_id,"trace_id":trace,"indexed":indexed,"note":note}
    finally: log_request(trace,req.title+" "+req.content,(time.perf_counter()-started)*1000,status)

@app.websocket("/ws/chat")
async def websocket_chat(ws:WebSocket):
    await ws.accept(); session_id="ws-"+uuid.uuid4().hex
    try:
        while True:
            payload=await ws.receive_json(); query=str(payload.get("query","")); trace=uuid.uuid4().hex; started=time.perf_counter(); status="ok"
            try:
                result=run_support(query,session_id,trace)
                await ws.send_json(result.model_dump())
            except (GuardrailError,RequestBudgetError) as e:
                status=getattr(e,"code","BUDGET_REJECTED")
                await ws.send_json({"error":status,"message":str(e),"trace_id":trace})
            finally: log_request(trace,query,(time.perf_counter()-started)*1000,status)
    except WebSocketDisconnect:
        # A single client disconnect ends only this handler; the ASGI server keeps running.
        return

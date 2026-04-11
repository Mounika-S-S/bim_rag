from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import shutil
from src.core.json_storage import JSONStorage
from src.app import query_vector_store_api, run_inference, run_l123, run_l125, run_l45, build_vector_store
from src.ingestion.ifc_parser import IFCParser
from src.l2.main_l2_pipeline import L2Pipeline
from src.l3.main_l3_pipeline import L3Pipeline
from src.ingestion.main_l4_pipeline import L4Pipeline
from src.l5.main_l5_pipeline import L5Pipeline

app = FastAPI()

# Allow local frontend (e.g. http://localhost:9000) to call backend
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HISTORY_DIR = "data/chat_history"
os.makedirs(HISTORY_DIR, exist_ok=True)

class ChatTurn(BaseModel):
    role: str
    text: str

class QueryRequest(BaseModel):
    project_id: str
    text: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    session_id: str
    answer: str
    history: List[ChatTurn]


def history_file(project_id: str, session_id: str) -> str:
    safe_project = project_id.replace("..", "")
    safe_session = session_id.replace("..", "")
    return os.path.join(HISTORY_DIR, f"{safe_project}__{safe_session}.json")


def load_history(project_id: str, session_id: str):
    path = history_file(project_id, session_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(project_id: str, session_id: str, history):
    path = history_file(project_id, session_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


@app.get("/ping")
def ping():
    return {"ok": True, "service": "BIM-RAG API"}


@app.get("/projects")
def list_projects():
    base = "data/processed"
    os.makedirs(base, exist_ok=True)
    return {"projects": [name for name in os.listdir(base) if os.path.isdir(os.path.join(base, name))]}


@app.post("/projects")
def create_project(project_id: str = Form(...)):
    path = os.path.join("data/processed", project_id)
    os.makedirs(path, exist_ok=True)
    return {"project_id": project_id, "path": path}


@app.post("/upload")
def upload_layer(
    project_id: str = Form(...),
    layer: str = Form(...),
    file: UploadFile = File(...)
):
    project_path = os.path.join("data/processed", project_id)
    os.makedirs(project_path, exist_ok=True)

    tmp_path = os.path.join(project_path, f"upload_{layer}_{file.filename}")
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    parsed = []
    if layer == "L1":
        parser = IFCParser()
        try:
            parsed = parser.parse_ifc(tmp_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid IFC file: {str(e)}")
        JSONStorage.save(project_id, "L1_ifc.json", parsed)
    elif layer == "L2":
        parser = L2Pipeline()
        try:
            parsed = parser.parse(tmp_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid L2 file: {str(e)}")
        JSONStorage.save(project_id, "L2_product.json", parsed)
    elif layer == "L3":
        parser = L3Pipeline()
        try:
            parsed = parser.parse(tmp_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid L3 file: {str(e)}")
        JSONStorage.save(project_id, "L3_process.json", parsed)
    elif layer == "L4":
        parser = L4Pipeline()
        try:
            parsed = parser.parse(tmp_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid L4 file: {str(e)}")
        JSONStorage.save(project_id, "L4_regulation.json", parsed)
    elif layer == "L5":
        parser = L5Pipeline()
        try:
            parsed = parser.parse(tmp_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid L5 file: {str(e)}")
        JSONStorage.save(project_id, "L5_requirement.json", parsed)
    else:
        raise HTTPException(status_code=400, detail="Unsupported layer")

    return {"project_id": project_id, "layer": layer, "count": len(parsed)}


@app.post("/run-inference")
def run_inference_endpoint(project_id: str = Form(...), inference_type: str = Form(...)):
    if inference_type == "l124":
        run_inference(project_id)
    elif inference_type == "l123":
        run_l123(project_id)
    elif inference_type == "l125":
        run_l125(project_id)
    elif inference_type == "l45":
        run_l45(project_id)
    else:
        raise HTTPException(status_code=400, detail="Unsupported inference type")

    return {"project_id": project_id, "inference": inference_type, "status": "done"}


@app.post("/build-vector-store")
def build_vector_store_endpoint(project_id: str = Form(...)):
    build_vector_store(project_id)
    return {"project_id": project_id, "status": "vector store built"}


@app.delete("/project")
def delete_project(project_id: str = Form(...)):
    path = os.path.join("data/processed", project_id)
    if os.path.exists(path):
        shutil.rmtree(path)
    return {"project_id": project_id, "status": "deleted"}


@app.post("/history/clear")
def clear_history(project_id: str = Form(...), session_id: Optional[str] = Form(None)):
    if session_id:
        path = history_file(project_id, session_id)
        if os.path.exists(path):
            os.remove(path)
        return {"status": "cleared", "project_id": project_id, "session_id": session_id}

    # clear all sessions for project
    prefix = f"{project_id}__"
    for filename in os.listdir(HISTORY_DIR):
        if filename.startswith(prefix):
            os.remove(os.path.join(HISTORY_DIR, filename))
    return {"status": "cleared_all", "project_id": project_id}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    session_id = request.session_id or os.urandom(8).hex()
    history = load_history(request.project_id, session_id)

    # Compose conversation history as string for LLM context
    # Limit to last 3 turns to stay within Groq token limits (free tier: 6000 TPM)
    chat_context = "\n".join(f"{h['role']}: {h['text']}" for h in history[-3:]) if history else ""

    try:
        answer = query_vector_store_api(request.project_id, request.text, chat_context)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    history.append({"role": "user", "text": request.text})
    history.append({"role": "assistant", "text": answer})
    save_history(request.project_id, session_id, history)

    recent_history = [ChatTurn(**turn) for turn in history[-16:]]

    return QueryResponse(
        session_id=session_id,
        answer=answer,
        history=recent_history,
    )

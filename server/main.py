from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
import logging

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
WEB_DIR = ROOT / "web"
SESSION_STORE = ROOT / os.getenv("SESSION_STORE_PATH", "data/sessions.json")
TRANSCRIPT_DIR = ROOT / os.getenv("TRANSCRIPT_STORE_PATH", "data/transcripts")

ENV_FILES = [
    ROOT / ".env",
]
load_dotenv(find_dotenv())
for env in ENV_FILES:
    if env.exists():
        load_dotenv(env, override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diligence")
logger.setLevel(logging.INFO)

TYPEFORM_FORM_ID = os.getenv("TYPEFORM_FORM_ID", "")
TYPEFORM_WEBHOOK_SECRET = os.getenv("TYPEFORM_WEBHOOK_SECRET", "")
TYPEFORM_API_TOKEN = os.getenv("TYPEFORM_API_TOKEN", "")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_AGENT_ID = os.getenv("ELEVENLABS_AGENT_ID", "")
ELEVENLABS_BASE_URL = os.getenv("ELEVENLABS_BASE_URL", "https://api.elevenlabs.io").rstrip("/")

app = FastAPI(title="DiligenceSquared Orchestrator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Serve the static frontend under /interview
if WEB_DIR.exists():
    app.mount("/interview", StaticFiles(directory=WEB_DIR, html=True), name="interview")


# --------- storage helpers ---------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    if not SESSION_STORE.exists():
        SESSION_STORE.write_text("{}", encoding="utf-8")


def _load_sessions() -> Dict[str, dict]:
    _ensure_dirs()
    try:
        return json.loads(SESSION_STORE.read_text(encoding="utf-8") or "{}")
    except json.JSONDecodeError:
        return {}


def _save_sessions(sessions: Dict[str, dict]) -> None:
    _ensure_dirs()
    SESSION_STORE.write_text(json.dumps(sessions, indent=2), encoding="utf-8")


# --------- models ---------

class SessionCreateResponse(BaseModel):
    respondent_id: str
    state: dict


class ConversationStartRequest(BaseModel):
    respondent_id: str


class ConversationCompleteRequest(BaseModel):
    respondent_id: str
    transcript: List[dict]


class TypeformWebhook(BaseModel):
    form_response: dict


# --------- business logic ---------

SEGMENT_CUSTOMER = "customer"
SEGMENT_POTENTIAL = "potential"
SEGMENT_SCREENED_OUT = "screened_out"


def _segment_from_answers(answers: List[dict]) -> str:
    """Derive segment from brand selection."""
    brands = []
    for ans in answers:
        if ans.get("type") == "choice":
            choice = ans.get("choice", {}).get("label")
            if choice:
                brands.append(choice.lower())
        if ans.get("type") == "choices":
            labels = ans.get("choices", {}).get("labels") or []
            brands.extend([l.lower() for l in labels])

    if any("bmw" in b for b in brands):
        return SEGMENT_CUSTOMER
    if any(b in {"mercedes-benz", "mercedes", "audi"} for b in brands):
        return SEGMENT_POTENTIAL
    return SEGMENT_SCREENED_OUT


def _new_session(respondent_id: Optional[str] = None) -> dict:
    return {
        "respondent_id": respondent_id or str(uuid.uuid4()),
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "segment": None,
        "status": "pending_screening",
        "typeform_response": None,
        "conversation_id": None,
        "progress": {"asked": 0, "answered": 0},
    }


def _extract_answers(form_response: Optional[dict]) -> Dict[str, Any]:
    answers = form_response.get("answers") if isinstance(form_response, dict) else []
    result: Dict[str, Any] = {}
    for ans in answers or []:
        qid = ans.get("field", {}).get("id") or ans.get("field", {}).get("ref")
        if ans.get("type") == "choice":
            result[qid] = ans.get("choice", {}).get("label")
        elif ans.get("type") == "choices":
            result[qid] = ans.get("choices", {}).get("labels")
        elif ans.get("type") == "text":
            result[qid] = ans.get("text")
    return result


def _session_context(session: dict) -> Dict[str, Any]:
    """Build a lightweight context blob to share with ElevenLabs."""
    answers = _extract_answers(session.get("typeform_response"))
    brands = answers.get("brand") or answers.get("car_brand")
    context = {
        "segment": session.get("segment"),
        "status": session.get("status"),
        "answers": answers,
    }
    # Add a readable summary for the prompt
    parts = []
    if session.get("segment"):
        parts.append(f"Segment: {session['segment']}")
    if brands:
        parts.append(f"Brands: {brands}")
    summary = "; ".join(parts) or "No screening summary available."
    context["summary"] = summary
    return context


def _update_session(respondent_id: str, **kwargs) -> dict:
    sessions = _load_sessions()
    session = sessions.get(respondent_id) or _new_session()
    session.update(kwargs)
    session["updated_at"] = _now_iso()
    sessions[respondent_id] = session
    _save_sessions(sessions)
    return session


def _get_session(respondent_id: str) -> dict:
    sessions = _load_sessions()
    if respondent_id not in sessions:
        raise HTTPException(status_code=404, detail="session not found")
    return sessions[respondent_id]


# --------- routes ---------

@app.post("/api/session", response_model=SessionCreateResponse)
def create_or_resume_session(respondent_id: Optional[str] = None) -> SessionCreateResponse:
    if respondent_id:
        sessions = _load_sessions()
        if respondent_id in sessions:
            return SessionCreateResponse(respondent_id=respondent_id, state=sessions[respondent_id])
    session = _new_session(respondent_id)
    session_copy = {k: v for k, v in session.items() if k != "respondent_id"}
    _update_session(session["respondent_id"], **session_copy)
    return SessionCreateResponse(respondent_id=session["respondent_id"], state=session)


@app.get("/api/session/{respondent_id}")
def fetch_session(respondent_id: str) -> dict:
    return _get_session(respondent_id)


def _verify_typeform_signature(raw_body: bytes, header_value: str) -> bool:
    """Validate Typeform HMAC signature per docs."""
    if not TYPEFORM_WEBHOOK_SECRET:
        return True
    if not header_value:
        return False
    try:
        provided = header_value.replace("sha256=", "").strip()
        digest = hmac.new(
            TYPEFORM_WEBHOOK_SECRET.encode("utf-8"),
            raw_body,
            hashlib.sha256,
        ).digest()
        computed = base64.b64encode(digest).decode()
        return hmac.compare_digest(provided, computed)
    except Exception:
        return False


@app.post("/api/typeform/webhook", status_code=204)
async def handle_typeform_webhook(request: Request):
    raw_body = await request.body()
    signature_header = request.headers.get("Typeform-Signature", "")
    if not _verify_typeform_signature(raw_body, signature_header):
        raise HTTPException(status_code=401, detail="invalid webhook signature")

    payload_dict = json.loads(raw_body.decode("utf-8"))
    payload = TypeformWebhook(**payload_dict)
    hidden = payload.form_response.get("hidden", {})
    respondent_id = hidden.get("respondent_id")
    if not respondent_id:
        raise HTTPException(status_code=400, detail="missing respondent_id in hidden fields")

    answers = payload.form_response.get("answers", [])
    segment = _segment_from_answers(answers)
    status_label = "qualified" if segment in {SEGMENT_CUSTOMER, SEGMENT_POTENTIAL} else "screened_out"

    _update_session(
        respondent_id,
        typeform_response=payload.form_response,
        segment=segment,
        status=status_label,
    )
    return {}


@app.post("/api/conversation/start")
async def start_conversation(req: ConversationStartRequest):
    session = _get_session(req.respondent_id)
    if session.get("status") == "screened_out":
        raise HTTPException(status_code=400, detail="respondent did not qualify")
    if not session.get("segment"):
        raise HTTPException(status_code=400, detail="screening not complete")

    conversation_id = session.get("conversation_id") or str(uuid.uuid4())
    # Create/ensure ElevenLabs conversation exists
    if not ELEVENLABS_API_KEY or not ELEVENLABS_AGENT_ID:
        raise HTTPException(status_code=500, detail="ElevenLabs credentials not configured")

    ws_url_override = None
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{ELEVENLABS_BASE_URL}/v1/convai/conversations",
                headers={"xi-api-key": ELEVENLABS_API_KEY},
                json={
                    "agent_id": ELEVENLABS_AGENT_ID,
                    "conversation_id": conversation_id,
                    "user_id": req.respondent_id,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            ws_url_override = data.get("websocket_url") or data.get("ws_url")
    except httpx.HTTPStatusError as exc:
        # Some accounts/agents may not require or allow the create call (405). If so, proceed anyway.
        status_code = exc.response.status_code if exc.response is not None else 500
        if status_code == 405:
            pass
        else:
            detail = exc.response.text if exc.response is not None else str(exc)
            raise HTTPException(status_code=status_code, detail=detail)
    except Exception as exc:  # pragma: no cover - external service
        raise HTTPException(status_code=500, detail=str(exc))

    session = _update_session(
        req.respondent_id,
        conversation_id=conversation_id,
        status="interview_in_progress",
    )
    questions = _question_set_for_segment(session["segment"])
    ws_base = ELEVENLABS_BASE_URL.replace("https://", "wss://").replace("http://", "ws://")
    if ws_url_override:
        ws_url = ws_url_override
    else:
        ws_url = (
            f"{ws_base}/v1/convai/conversation?"
            f"agent_id={ELEVENLABS_AGENT_ID}&conversation_id={conversation_id}&xi-api-key={ELEVENLABS_API_KEY}"
        )
    context = _session_context(session)
    try:
        logger.info("ElevenLabs context for %s: %s", req.respondent_id, json.dumps(context))
    except Exception:
        logger.info("ElevenLabs context for %s: %s", req.respondent_id, context)

    return {
        "conversation_id": conversation_id,
        "segment": session["segment"],
        "questions": questions,
        "progress": session.get("progress", {}),
        "history": session.get("history", []),
        "ws_url": ws_url,
        "context": context,
    }


def _question_set_for_segment(segment: str) -> List[str]:
    common = [
        "Are you ready to begin?",
        "How long have you owned your current vehicle?",
        "What were the main factors that influenced your decision to purchase this specific brand?",
        "On a scale of 1 to 10, how satisfied are you with your current vehicle?",
        "What features or aspects of your car do you value most?",
        "Have you experienced any issues or concerns with your vehicle?",
    ]
    closing = [
        "Is there anything else you'd like to share about your vehicle ownership experience?",
    ]
    if segment == SEGMENT_CUSTOMER:
        specific = [
            "What made you choose BMW over other luxury brands like Mercedes or Audi?",
            "How would you rate BMW's customer service and dealership experience?",
            "Which BMW model do you own, and what do you love most about it?",
            "How likely are you to purchase another BMW in the future? What would make you consider switching brands?",
            "What could BMW improve to make your ownership experience even better?",
        ]
    else:
        specific = [
            "Have you ever considered purchasing a BMW? Why or why not?",
            "What perceptions or impressions do you have of the BMW brand?",
            "What would it take for you to switch to BMW for your next vehicle purchase?",
            "Compared to BMW, what do you think your current brand does better?",
            "If you were to recommend a luxury car brand to a friend, which would you choose and why?",
        ]
    return common + specific + closing


@app.post("/api/conversation/complete", status_code=204)
async def complete_conversation(req: ConversationCompleteRequest):
    session = _get_session(req.respondent_id)
    if not session.get("conversation_id"):
        raise HTTPException(status_code=400, detail="conversation not started")

    transcript_path = TRANSCRIPT_DIR / f"{req.respondent_id}.json"
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(json.dumps(req.transcript, indent=2), encoding="utf-8")

    _update_session(
        req.respondent_id,
        status="interview_complete",
        progress={"asked": len(req.transcript), "answered": len(req.transcript)},
        history=req.transcript,
    )
    return {}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

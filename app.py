"""Local FastAPI demo for the PDF RAG chat widget."""
import json
import logging
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, selectinload
from database import get_db
from models import Conversation, Message, MessageSource, utc_now
from conversation_memory import select_history

from prepare_documents import evidence_path

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="Animal Kingdom PDF Chat", version="0.1.0")
app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")
CHAT_LOCK = threading.Lock()


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    content_type: Literal["image", "text", "table"] | None = None
    conversation_id: str | None = Field(default=None, min_length=36, max_length=36)


@app.post("/api/conversations", status_code=201)
def create_conversation(db: Session = Depends(get_db)):
    conversation = Conversation()
    db.add(conversation)
    try:
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(503, "Chat storage is unavailable.") from None
    return {"id": conversation.id, "title": conversation.title}


@app.get("/api/conversations/{conversation_id}")
def conversation_history(conversation_id: str, db: Session = Depends(get_db)):
    try:
        conversation = db.get(Conversation, conversation_id)
        if conversation is None:
            raise HTTPException(404, "Conversation not found.")
        messages = db.scalars(select(Message).where(Message.conversation_id == conversation_id)
                             .options(selectinload(Message.sources)).order_by(Message.created_at, Message.id)).all()
        history = []
        for message in messages:
            sources = [{"label": s.citation_label, "citation": f"{s.source_filename}, PDF page {s.page_number}",
                        "page_url": f"/api/media/{s.record_id}/page"} for s in message.sources]
            images = [{"label": s.citation_label, "citation": f"{s.source_filename}, PDF page {s.page_number}",
                       "url": f"/api/media/{s.record_id}/image"} for s in message.sources if s.is_selected_image]
            history.append({"id": message.id, "role": message.role, "content": message.content,
                            "status": message.status, "sources": sources, "images": images,
                            "created_at": message.created_at.isoformat() + "Z"})
        return {"id": conversation.id, "title": conversation.title, "messages": history}
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(503, "Chat storage is unavailable.") from None


def document():
    """Select one prepared PDF and whitelist its evidence files."""
    paths = list((ROOT / "storage").glob("*/prepared_documents_captioned.jsonl"))
    if len(paths) != 1:
        raise HTTPException(503, "The demo needs exactly one prepared PDF in storage.")
    prepared = paths[0]
    rows = [json.loads(line) for line in prepared.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise HTTPException(503, "The document has no prepared records.")
    records = {row["record_id"]: row for row in rows}
    return prepared, records


@app.get("/")
def home():
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/api/document")
def document_info():
    prepared, records = document()
    manifest = json.loads((prepared.parent / "manifest.json").read_text(encoding="utf-8"))
    return {"filename": manifest["source_filename"], "pages": manifest["page_count"], "records": len(records)}


@app.get("/api/media/{record_id}/{kind}")
def media(record_id: str, kind: Literal["image", "page"]):
    prepared, records = document()
    row = records.get(record_id)
    if row is None:
        raise HTTPException(404, "Evidence not found.")
    key = "image_path" if kind == "image" else "page_image_path"
    if key not in row:
        raise HTTPException(404, "Evidence image not found.")
    try:
        relative = evidence_path(prepared.parent.resolve(), row[key])
    except (ValueError, OSError):
        raise HTTPException(404, "Evidence unavailable.") from None
    return FileResponse(prepared.parent / relative, media_type="image/png")


def answer(question, content_type, prepared, history=None):
    """Keep the existing CLI usable; run it with an isolated output per request."""
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "answer.json"
        command = [sys.executable, str(ROOT / "answer_question.py"), question,
                   "--prepared", str(prepared), "--output", str(output), "--top-k", "3"]
        if content_type:
            command.extend(["--type", content_type])
        if history:
            history_path = Path(directory) / "history.json"
            history_path.write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")
            command.extend(["--history", str(history_path)])
        try:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=240)
        except subprocess.TimeoutExpired:
            raise HTTPException(504, "The answer took too long. Please try again.") from None
        if result.returncode:
            # Do not send provider responses, credentials, or filesystem paths to browsers.
            logging.warning("RAG command failed with exit code %s", result.returncode)
            raise HTTPException(502, "The AI service could not answer right now. Check provider quota or connectivity and try again.")
        return json.loads(output.read_text(encoding="utf-8"))


@app.post("/api/chat")
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    question = request.question.strip()
    if not question:
        raise HTTPException(422, "Please enter a question.")
    # A single demo request at a time avoids a burst of provider quota usage.
    if not CHAT_LOCK.acquire(blocking=False):
        raise HTTPException(429, "An answer is already being prepared. Please try again shortly.")
    assistant_message = None
    try:
        prepared, records = document()
        conversation = db.get(Conversation, request.conversation_id) if request.conversation_id else Conversation(title=question[:255])
        if conversation is None:
            raise HTTPException(404, "Conversation not found.")
        if not request.conversation_id:
            db.add(conversation)
            db.flush()
        elif conversation.title == "New conversation":
            conversation.title = question[:255]
        conversation.updated_at = utc_now()
        previous = db.scalars(select(Message).where(Message.conversation_id == conversation.id)
                              .order_by(Message.created_at, Message.id)).all()
        history = select_history([{"role": m.role, "content": m.content, "status": m.status}
                                  for m in previous], question)
        user_message = Message(conversation_id=conversation.id, role="user", content=question)
        assistant_message = Message(conversation_id=conversation.id, role="assistant", content="", status="pending")
        db.add(user_message)
        db.flush()
        db.add(assistant_message)
        db.commit()
        try:
            result = answer(question, request.content_type, prepared, history)
        except HTTPException as exc:
            assistant_message.status = "failed"
            assistant_message.content = str(exc.detail)
            db.commit()
            raise
        sources = []
        label_records = {}
        for source in result["sources"]:
            record_id = source["record_id"]
            if record_id not in records:
                raise HTTPException(502, "The answer returned unavailable evidence.")
            label_records[source["label"]] = record_id
            sources.append({"label": source["label"], "citation": source["citation"],
                            "page_url": f"/api/media/{record_id}/page"})
        images = []
        for image in result["images"]:
            record_id = label_records.get(image["label"])
            if record_id and "image_path" in records[record_id]:
                images.append({"label": image["label"], "citation": image["citation"],
                               "url": f"/api/media/{record_id}/image"})
        selected = {image["label"] for image in result["images"]}
        for source in result["sources"]:
            row = records[source["record_id"]]
            db.add(MessageSource(message_id=assistant_message.id, record_id=source["record_id"],
                                 citation_label=source["label"], document_id=row["document_id"],
                                 source_filename=row["source_filename"], page_number=row["page_number"],
                                 content_type=row["content_type"], is_selected_image=source["label"] in selected))
        assistant_message.content = result["answer"]
        assistant_message.status = "completed"
        conversation.updated_at = utc_now()
        db.commit()
        return {"answer": result["answer"], "sources": sources, "images": images,
                "conversation_id": conversation.id, "message_id": assistant_message.id}
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(503, "Chat storage is unavailable. Please try again.") from None
    except Exception as exc:
        db.rollback()
        if assistant_message is not None:
            try:
                saved = db.get(Message, assistant_message.id)
                if saved and saved.status == "pending":
                    saved.status = "failed"
                    saved.content = "The answer could not be completed. Please try again."
                    db.commit()
            except SQLAlchemyError:
                db.rollback()
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(502, "The answer could not be completed. Please try again.") from None
    finally:
        CHAT_LOCK.release()

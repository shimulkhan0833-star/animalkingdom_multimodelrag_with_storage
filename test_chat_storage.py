"""Test API persistence without calling AI services or modifying MySQL."""
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import app
from database import get_db
from models import Base, Message, MessageSource


class ChatStorageTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(self.engine)

        def db():
            with Session(self.engine, expire_on_commit=False) as session:
                yield session

        app.app.dependency_overrides[get_db] = db
        self.client = TestClient(app.app)

    def tearDown(self):
        app.app.dependency_overrides.clear()
        self.engine.dispose()

    def test_save_and_reload(self):
        prepared, records = app.document()
        row = next(r for r in records.values() if r["content_type"] == "image")
        result = {"answer": "Frog [S1]", "sources": [{"label": "S1", "record_id": row["record_id"],
                  "citation": "animal_kingdom.pdf, PDF page 12"}], "images": [{"label": "S1", "citation": "PDF page 12"}]}
        conversation = self.client.post("/api/conversations").json()["id"]
        with patch("app.answer", return_value=result):
            response = self.client.post("/api/chat", json={"question": "frog", "conversation_id": conversation})
        self.assertEqual(response.status_code, 200)
        history = self.client.get(f"/api/conversations/{conversation}").json()["messages"]
        self.assertEqual([m["role"] for m in history], ["user", "assistant"])
        self.assertEqual(history[1]["content"], "Frog [S1]")
        self.assertEqual(len(history[1]["images"]), 1)
        with Session(self.engine) as db:
            self.assertTrue(db.scalar(select(MessageSource)).is_selected_image)

    def test_failed_answer_is_saved(self):
        conversation = self.client.post("/api/conversations").json()["id"]
        with patch("app.answer", side_effect=HTTPException(502, "Provider unavailable")):
            response = self.client.post("/api/chat", json={"question": "frog", "conversation_id": conversation})
        self.assertEqual(response.status_code, 502)
        history = self.client.get(f"/api/conversations/{conversation}").json()["messages"]
        self.assertEqual(history[1]["status"], "failed")
        with Session(self.engine) as db:
            self.assertEqual(len(db.scalars(select(Message)).all()), 2)


if __name__ == "__main__":
    unittest.main()

import unittest

from rag.conversation_memory import select_history, is_identity_question
from rag.answer_question import build_messages


class MemoryTests(unittest.TestCase):
    def test_old_name_survives_recent_chatter(self):
        turns = [{"role": "user", "content": "My name is Alex", "status": "completed"}]
        turns += [{"role": "user", "content": "Explain frogs", "status": "completed"} for _ in range(30)]
        history = select_history(turns, "What is my name?")
        self.assertTrue(any("Alex" in m["content"] for m in history))
        messages, sources, attached = build_messages("What is my name?", [], history)
        self.assertIn("Alex", messages[-1]["content"][0]["text"])
        self.assertEqual(sources, {})

    def test_newer_correction_and_failed_turns(self):
        history = select_history([
            {"role": "user", "content": "My name is Alex", "status": "completed"},
            {"role": "assistant", "content": "Your name is Wrong", "status": "failed"},
            {"role": "user", "content": "My name is Sam now", "status": "completed"}], "What is my name?")
        self.assertEqual(history[-1]["content"], "My name is Sam now")
        self.assertEqual(len(history), 2)
        self.assertTrue(is_identity_question("What is my name?"))
        self.assertTrue(is_identity_question("My name is Alex"))
        self.assertFalse(is_identity_question("What are amphibians?"))


if __name__ == "__main__":
    unittest.main()

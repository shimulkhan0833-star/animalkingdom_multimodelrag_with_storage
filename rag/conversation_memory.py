"""Select recent turns and older relevant user facts from one conversation."""
import re


def select_history(messages, question, max_chars=5000):
    """Prefer recent turns, then older user messages matching the current query."""
    turns = [m for m in messages if m["status"] == "completed" and m["role"] in ("user", "assistant")]
    query_words = set(re.findall(r"\w+", question.casefold())) - {"is", "the", "a", "what", "my", "i", "you", "me", "of", "to"}
    selected = set(range(max(0, len(turns) - 8), len(turns)))
    older = [(len(query_words & set(re.findall(r"\w+", m["content"].casefold()))), i)
             for i, m in enumerate(turns) if i not in selected and m["role"] == "user"]
    candidates = [i for score, i in sorted(older, reverse=True) if score > 0][:6]
    # Relevant older facts get budget priority, so recent chatter cannot hide a name.
    chosen, remaining = [], max_chars
    for i in candidates + sorted(selected, reverse=True):
        text = turns[i]["content"][:1200]
        if len(text) <= remaining:
            chosen.append(i)
            remaining -= len(text)
    return [{"role": turns[i]["role"], "content": turns[i]["content"][:1200]} for i in sorted(chosen)]


def is_identity_question(question):
    """Personal name questions do not need document retrieval or PDF citations."""
    text = question.casefold().strip()
    return bool(re.search(r"\bmy name is\b|\b(?:what(?:'s| is)|remember|know|tell me)\s+(?:what\s+)?my name\b", text))

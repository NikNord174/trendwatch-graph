"""Retrieval is the Chat section's whole behavior in keyless mode, so its
ranking logic gets tested like any other pure function. st.cache_data
functions run fine outside the Streamlit runtime."""

import chat


def test_tokenize_keeps_tech_tokens():
    assert chat.tokenize("C++ and node.js beat f#") == ["c++", "and", "node.js", "beat", "f#"]
    assert chat.tokenize("A B c") == []  # single letters carry no signal


def test_score_topic_prefers_label_match_over_big_vocabulary():
    idf = {"agents": 2.5, "happening": 3.8, "ai": 1.0}
    q = {"what", "is", "happening", "with", "ai", "agents"}
    small_relevant = chat.score_topic(q, {"ai", "agents", "coding"}, set("abcdefgh"), idf)
    big_generic = chat.score_topic(
        q, {"ai", "hate", "slop"}, {f"w{i}" for i in range(2000)} | {"happening"}, idf
    )
    assert small_relevant > big_generic


def test_retrieve_returns_matching_topic_first():
    topics = chat.data.load_topics()
    target = topics.iloc[5]
    question = "what about " + " ".join(target["terms"][:3])
    hits = chat.retrieve(question)
    assert not hits.empty
    assert int(hits.iloc[0]["id"]) == int(target["id"])


def test_retrieve_no_match_returns_empty():
    assert chat.retrieve("xyzzy quux frobnicate").empty

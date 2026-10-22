from conftest import FakeEmbeddings

from news_mcp import chunking
from news_mcp.chunking import make_splitter, split_article

BODY = " ".join(f"Sentence number {i} talks about the reactor." for i in range(100))  # 700 words


def test_chunks_respect_token_limit_and_overlap(monkeypatch):
    monkeypatch.setattr(chunking, "CHUNK_TOKENS", 50)
    monkeypatch.setattr(chunking, "CHUNK_OVERLAP_TOKENS", 10)
    count_words = FakeEmbeddings().count_tokens
    chunks = split_article({"id": "a1", "title": "T", "body": BODY}, make_splitter(count_words))

    assert len(chunks) > 10
    assert all(count_words(c["text"]) <= 50 for c in chunks)
    # consecutive chunks share some text (overlap keeps context across the boundary)
    assert chunks[1]["text"].split()[0] in chunks[0]["text"]


def test_chunks_end_at_sentence_boundaries(monkeypatch):
    monkeypatch.setattr(chunking, "CHUNK_TOKENS", 50)
    chunks = split_article({"id": "a1", "title": "T", "body": BODY}, make_splitter(FakeEmbeddings().count_tokens))
    assert all(c["text"].endswith(".") for c in chunks)


def test_chunks_are_traceable_to_the_article(monkeypatch):
    monkeypatch.setattr(chunking, "CHUNK_TOKENS", 50)
    chunks = split_article({"id": "a1", "title": "T", "body": BODY}, make_splitter(FakeEmbeddings().count_tokens))
    for i, chunk in enumerate(chunks):
        assert chunk["article_id"] == "a1" and chunk["chunk_index"] == i
        assert BODY[chunk["start_char"]:chunk["end_char"]] == chunk["text"]


def test_short_article_is_one_chunk():
    chunks = split_article({"id": "a1", "title": "T", "body": "Short text."}, make_splitter(len))
    assert [c["text"] for c in chunks] == ["Short text."]


def test_title_is_added_as_header_for_embedding():
    assert chunking.text_to_embed("Title", "Chunk text.") == "Title\n\nChunk text."
